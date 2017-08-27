from __future__ import division
import random
import sys
import itertools
from copy import deepcopy
import macarico
import numpy as np
import dynet as dy

from macarico.lts.lols import EpisodeRunner, one_step_deviation

# helpful functions

def reseed(seed=90210):
    random.seed(seed)
    #torch.manual_seed(seed)
    np.random.seed(seed)

def break_ties_by_policy(reference, policy, state, force_advance_policy=True):
    costs = np.zeros(state.n_actions)
    try:
        reference.set_min_costs_to_go(state, costs)
    except NotImplementedError:
        ref = reference(state)
        if force_advance_policy:
            policy(state)
        return ref
    # otherwise we successfully got costs
    old_actions = state.actions
    min_cost = min((costs[a] for a in old_actions))
    state.actions = [a for a in old_actions if costs[a] <= min_cost]
    a = policy(state)  # advances policy
    #print costs, old_actions, state.actions, a
    #a = state.actions[0]
    assert a is not None, 'got action None in %s, costs=%s, old_actions=%s' % (state.actions, costs, old_actions)
    state.actions = old_actions
    return a


def evaluate(data, policy, losses, verbose=False):
    "Compute average `loss()` of `policy` on `data`"
    was_list = True
    if not isinstance(losses, list):
        losses = [losses]
        was_list = False
    for loss in losses:
        loss.reset()
    for example in data:
        env = example.mk_env()
        res = env.run_episode(policy)
        if verbose:
            print res, example
        for loss in losses:
            loss(example, env)
    scores = [loss.get() for loss in losses]
    if not was_list:
        scores = scores[0]
    return scores


def should_print(print_freq, last_print, N):
    if print_freq is None:
        return False
    if last_print is None:
        return True
    next_print = last_print + print_freq if isinstance(print_freq, int) else \
                 last_print * print_freq
    return N >= next_print


def minibatch(data, minibatch_size, reshuffle):
    """
    >>> list(minibatch(range(8), 3, 0))
    [[0, 1, 2], [3, 4, 5], [6, 7]]

    >>> list(minibatch(range(0), 3, 0))
    []

    """
    # TODO this can prob be made way more efficient
    if reshuffle:
        random.shuffle(data)
    for n in xrange(0, len(data), minibatch_size):
        yield data[n:n+minibatch_size]


def padto(s, l):
    if isinstance(s, list):
        s = ' '.join(map(str, s))
    elif not isinstance(s, str):
        s = str(s)
    n = len(s)
    if n > l:
        return s[:l-2] + '..'
    return s + (' ' * (l - n))

def learner_to_alg(Learner, loss):
    def learning_alg(ex):
        env = ex.mk_env()
        learner = Learner()
        env.run_episode(learner)
        loss_val = loss.evaluate(ex, env)
        learner.update(loss_val)
        return loss_val, getattr(learner, 'squared_loss', 0)
    return learning_alg

def trainloop(training_data,
              dev_data=None,
              policy=None,
              Learner=None,
              learning_alg=None,
              optimizer=None,
              losses=None,      # one or more losses, first is used for early stopping
              n_epochs=10,
              minibatch_size=1,
              run_per_batch=[],
              run_per_epoch=[],
              print_freq=2.0,   # int=additive, float=multiplicative
              print_per_epoch=True,
              quiet=False,
              train_eval_skip=100,
              reshuffle=True,
              print_dots=True,
              returned_parameters='best',  # { best, last, none }
              save_best_model_to=None,
              hogwild_rank=None,
              bandit_evaluation=False,
             ):

    assert (Learner is None) != (learning_alg is None), \
        'trainloop expects exactly one of Learner / learning_alg'

    assert losses is not None, \
        'must specify at least one loss function'

    if bandit_evaluation and n_epochs > 1 and not quiet:
        print >>sys.stderr, 'warning: running bandit mode with n_epochs>1, this is weird!'
    
    if not isinstance(losses, list):
        losses = [losses]

    if learning_alg is None:
        learning_alg = learner_to_alg(Learner, losses[0])

    extra_loss_format = ''
    if not quiet:
        extra_loss_header = ''
        if len(losses) > 1:
            extra_loss_header += ' ' * 9
        for evaluator in losses[1:]:
            extra_loss_header += padto('  tr_' + evaluator.name, 10)
            extra_loss_header += padto('  de_' + evaluator.name, 10)
            extra_loss_format += '  %-8.5f  %-8.5f'
        print >>sys.stderr, '%s %s %8s  %5s  rand_dev_truth          rand_dev_pred%s' % \
            ('tr_' + padto(losses[0].name, 8),
             'de_' + padto(losses[0].name, 8),
             'N', 'epoch', extra_loss_header)

    last_print = None
    best_de_err = float('inf')
    final_parameters = None
    error_history = []
    num_batches = len(training_data) // minibatch_size
    bandit_loss, bandit_count = 0., 0.

    if hogwild_rank is not None:
        reseed(20009 + 4837 * hogwild_rank)

    squared_loss, squared_loss_cnt = 0., 0.
        
    N = 0  # total number of examples seen
    for epoch in xrange(1, n_epochs+1):
        M = 0  # total number of examples seen this epoch
        for batch in minibatch(training_data, minibatch_size, reshuffle):
            #if optimizer is not None:
                #optimizer.zero_grad()
            dy.renew_cg()
            # TODO: minibatching is really only useful if we can
            # preprocess in a useful way
            for ex in batch:
                N += 1
                M += 1
                bl, sq = learning_alg(ex)
                bandit_loss += bl
                bandit_count += 1
                squared_loss += sq
                squared_loss_cnt += 1
                if print_dots and (len(training_data) <= 40 or M % (len(training_data)//40) == 0):
                    sys.stderr.write('.')
                              
            if optimizer is not None:
                optimizer.update()

            if not quiet and (should_print(print_freq, last_print, N) or \
                              (print_per_epoch and M >= len(training_data))):
                tr_err = [0] * len(losses)
                if bandit_evaluation:
                    tr_err[0] = bandit_loss/bandit_count
                elif train_eval_skip is not None:
                    tr_err = evaluate(training_data[::train_eval_skip], policy, losses)
                de_err = [0] * len(losses) if dev_data is None else \
                         evaluate(dev_data, policy, losses)

                if not isinstance(tr_err, list): tr_err = [tr_err]
                if not isinstance(de_err, list): de_err = [de_err]
                
                extra_loss_scores = list(itertools.chain(*zip(tr_err[1:], de_err[1:])))
                error_history.append((tr_err, de_err))

                random_dev_truth, random_dev_pred = '', ''
                if dev_data is not None:
                    ex = random.choice(dev_data)
                    random_dev_truth = ex
                    random_dev_pred  = ex.mk_env().run_episode(policy)

                if print_dots:
                    sys.stderr.write('\r')

                fmt = '%-10.6f  %-10.6f  %8s  %5s  [%s]  [%s]' + extra_loss_format
                is_best = de_err[0] < best_de_err
                if is_best:
                    fmt += '  *'
                fmt_vals = [tr_err[0],
                            de_err[0], N, epoch,
                            padto(random_dev_truth, 20), padto(random_dev_pred, 20)] + \
                           extra_loss_scores
                #print >>sys.stderr, '%g |' % (squared_loss / squared_loss_cnt),
                print >>sys.stderr, fmt % tuple(fmt_vals)
                
                last_print = N
                if is_best:
                    best_de_err = de_err[0]
                    if save_best_model_to is not None:
                        if print_dots:
                            print >>sys.stderr, 'saving model to %s...' % save_best_model_to,
                        torch.save(policy.state_dict(), save_best_model_to)
                        if print_dots:
                            sys.stderr.write('\r' + (' ' * (21 + len(save_best_model_to))) + '\r')
                    if returned_parameters == 'best':
                        final_parameters = None # deepcopy(policy)

            for x in run_per_batch: x()
        for x in run_per_epoch: x()

    if returned_parameters == 'last':
        final_parameters = None # deepcopy(policy)
        
    return error_history, final_parameters

########################################################
# synthetic data construction

def make_sequence_reversal_data(num_ex, ex_len, n_types):
    data = []
    for _ in xrange(num_ex):
        x = [random.choice(range(n_types)) for _ in xrange(ex_len)]
        y = list(reversed(x))
        data.append((x,y))
    return data

def make_sequence_mod_data(num_ex, ex_len, n_types, n_labels):
    data = []
    for _ in xrange(num_ex):
        x = np.random.randint(n_types, size=ex_len)
        y = (x+1) % n_labels
        data.append((x,y))
    return data

def test_reference_on(ref, loss, ex, verbose=True, test_values=False):
    from macarico import Policy
    from macarico.policies.linear import LinearPolicy
    
    env = ex.mk_env()
    policy = LinearPolicy(dy.ParameterCollection(), None, env.n_actions)
    
    def run(run_strategy):
        env.rewind()
        runner = EpisodeRunner(policy, run_strategy, ref)
        env.run_episode(runner)
        cost = loss()(ex, env)
        return cost, runner.trajectory, runner.limited_actions, runner.costs, runner.ref_costs
    
    # generate the backbone by REF
    loss0, traj0, limit0, costs0, refcosts0 = run(lambda t: EpisodeRunner.REF)
    if verbose:
        print 'loss0', loss0, 'traj0', traj0
    
    backbone = lambda t: (EpisodeRunner.ACT, traj0[t])
    n_actions = env.n_actions
    any_fail = False
    for t in xrange(len(traj0)):
        costs = np.zeros(n_actions)
        traj1_all = [None] * n_actions
        for a in limit0[t]:
            #if a == traj0[t]: continue
            l, traj1, _, _, _ = run(one_step_deviation(backbone, lambda _: EpisodeRunner.REF, t, a))
            if verbose:
                print t, a, l
            costs[a] = l
            traj1_all[a] = traj1
            if l < loss0 or (a == traj0[t] and l != loss0):
                print 'local opt failure, ref loss=%g, loss=%g on deviation (%d, %d), traj0=%s traj\'=%s [ontraj=%s]' % \
                    (loss0, l, t, a, traj0, traj1, a == traj0[t])
                any_fail = True
                raise Exception()
        if test_values:
            for a in limit0[t]:
                if refcosts0[t][a] != costs[a]:
                    print 'cost failure, t=%d, a=%d, traj0=%s, traj1=%s, ref_costs=%s, observed costs=%s' % \
                        (t, a, traj0, traj1_all[a], \
                         [refcosts0[t][a0] for a0 in limit0[t]], \
                         [costs[a0] for a0 in limit0[t]])
                    raise Exception()
            
    if not any_fail:
        print 'passed!'

def test_reference(ref, loss, data, verbose=False, test_values=False):
    for n, ex in enumerate(data):
        print '# example %d ' % n,
        test_reference_on(ref, loss, ex, verbose, test_values)

def sample_from_probs(probs):
    r = np.random.rand()
    a = 0
    for i, v in enumerate(probs.npvalue()):
        r -= v
        if r <= 0:
            a = i
            break
    return a, probs[a]
