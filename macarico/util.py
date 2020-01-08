import sys
from copy import deepcopy
import macarico
import numpy as np
import torch
import torch.nn as nn
import progressbar
import time
import os
import random as py_random

from macarico.lts.lols import EpisodeRunner, one_step_deviation
from macarico.annealing import Averaging


# helpful functions
Var = torch.autograd.Variable


def feature_vector_to_vw_string(feature_vector):
    feature_vector = feature_vector.reshape(-1)
    ex = ' | '
    for i, value in enumerate(feature_vector):
        ex += ' ' + str(i) + ':' + str(value.item())
    return ex

def feature_vector_to_vw_string_adf(feature_vector, n_actions, act=None, prob=None, cost=None):
    feature_vector = feature_vector.reshape(-1)
    examples = []
    ex = 'shared |'
    for i, value in enumerate(feature_vector):
        ex += ' ' + str(value.item())
    examples.append(ex)
    if act == None:
        for action in range(n_actions):
            ex = ' |'
            for a in range(n_actions):
                if a == action:
                    ex += ' ' + str(a) + ':1'
                else:
                    ex += ' ' + str(a) + ':0'
            examples.append(ex)
    else:
        for action in range(n_actions):
            if action == act:
                ex = '0:' + str(cost) + ':' + str(prob) + ' |'
            else:
                ex = ' |'
            for a in range(n_actions):
                if a == action:
                    ex += ' ' + str(a) + ':1'
                else:
                    ex += ' ' + str(a) + ':0'
            examples.append(ex)
    return examples


def Varng(*args, **kwargs):
    return torch.autograd.Variable(*args, requires_grad=False, **kwargs)


def getnew(param):
    return param.new if hasattr(param, 'new') else \
        param.data.new if hasattr(param, 'data') else \
        param.weight.data.new if hasattr(param, 'weight') else \
        param._typememory.param.data.new if hasattr(param, '_typememory') else \
        None


def zeros(param, *dims):
    return getnew(param)(*dims).zero_()


def longtensor(param, *dims):
    return getnew(param)(*dims).long()


def onehot(param, i):
    return Varng(longtensor(param, [int(i)]))


def argmin(vec, allowed=None, dim=0):
    if isinstance(vec, Var): vec = vec.data
    if allowed is None or len(allowed) == 0 or len(allowed) == vec.shape[dim]:
        return vec.min(dim)[1].item()
    i = None
    for a in allowed:
        if i is None or \
           (dim == 0 and vec[a] < vec[i]) or \
           (dim == 1 and vec[0,a] < vec[0,i]) or \
           (dim == 2 and vec[0,0,a] < vec[0,0,i]):
            i = a
    return i
           

def getattr_deep(obj, field):
    for f in field.split('.'):
        obj = getattr(obj, f)
    return obj


def reseed(seed=90210, gpu_id=None):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if gpu_id is not None:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def break_ties_by_policy(reference, policy, state, force_advance_policy=True):
    costs = torch.zeros(state.n_actions)
    set_costs = False
    try:
        reference.set_min_costs_to_go(state, costs)
        set_costs = True
    except NotImplementedError:
        pass
    if not set_costs:
        ref = reference(state)
        if force_advance_policy:
            policy(state)
        return ref
    # otherwise we successfully got costs
    old_actions = state.actions
    min_cost = min((costs[a] for a in old_actions))
    state.actions = [a for a in old_actions if costs[a] <= min_cost]
    a = policy(state)  # advances policy
    assert a is not None, 'got action None in %s, costs=%s, old_actions=%s' % (state.actions, costs, old_actions)
    state.actions = old_actions
    return a
    

def evaluate(mk_env, data, policy, losses, verbose=False):
    "Compute average `loss()` of `policy` on `data`"
    was_list = True
    if not isinstance(losses, list):
        losses = [losses]
        was_list = False
    for loss in losses:
        loss.reset()
    for example in data:
        policy.new_minibatch()
        mk_env(example).run_episode(policy)
        if verbose:
            print(example)
        for loss in losses:
            loss(example)
    scores = [loss.get() for loss in losses]
    if not was_list:
        scores = scores[0]
    return scores


def minibatch(data, minibatch_size):
    """
    >>> list(minibatch(range(8), 3, 0))
    [[0, 1, 2], [3, 4, 5], [6, 7]]

    >>> list(minibatch(range(0), 3, 0))
    []
    """
    mb = []
    data = iter(data)
    try:
        prev_x = next(data)
    except StopIteration:
        # there are no examples
        return
    while True:
        mb.append(prev_x)
        try:
            prev_x = next(data)
        except StopIteration:
            break
        if len(mb) >= minibatch_size:
            yield mb, False
            mb = []
    if len(mb) > 0:
        yield mb, True


def padto(s, l, right=False):
    if isinstance(s, list):
        s = ' '.join(map(str, s))
    elif not isinstance(s, str):
        s = str(s)
    n = len(s)
    if l is not None and n > l:
        return s[:l-2] + '..'
    if l is not None:
        if right:
            s = ' ' * (l - n) + s
        else:
            s += ' ' * (l - n)
    return s


class LearnerToAlg(macarico.LearningAlg):
    def __init__(self, learner, policy, loss):
        macarico.LearningAlg.__init__(self)
        self.learner = learner
        self.policy = policy
        self.loss = loss()

    def __call__(self, env):
        env.rewind(self.policy)
        env.run_episode(self.learner)
        loss = self.loss.evaluate(env.example)
        obj = self.learner.get_objective(loss, final_state=env)
        return obj


class LossMatrix(object):
    def __init__(self, n_ex, losses):
        if not isinstance(losses, list):
            losses = [losses]
        self.losses = [loss() for loss in losses]
        self.A = torch.zeros(3, len(losses)+2)
        self.i = 0
        self.cur_count = 0
        for loss in self.losses:
            loss.reset()
        self.n_ex = n_ex
        self.examples = []

    def append(self, example):
        self.cur_count += 1
        for loss in self.losses:
            loss(example)
        if len(self.examples) < self.n_ex:
            self.examples.append(example)
        elif np.random.random() < self.n_ex/(self.n_ex+self.cur_count):
            self.examples[np.random.randint(0, self.n_ex)] = example

    def run_and_append(self, env, policy):
        out = env.run_episode(policy)
        self.append(env.example)
        return out

    def names(self):
        return [loss.name for loss in self.losses]

    def next(self, n_ex, epoch):
        M, N = self.A.shape
        if self.i >= M:
            B = torch.zeros(self.i*2, N)
            B[:self.i, :] = self.A
            self.A = B
        for n, loss in enumerate(self.losses):
            self.A[self.i, n] = loss.get()
        for n, loss in enumerate(self.losses):
            loss.reset()
        self.A[self.i, -2] = n_ex
        self.A[self.i, -1] = epoch
        self.i += 1
        self.cur_count = 0
        return self.row(self.i-1)

    def row(self, i):
        assert (0 <= i) and (i < self.i)
        return self.A[i, :-2]

    def last_row(self):
        return self.row(self.i-1)

    def col(self, n):
        assert (0 <= n) and (n < len(self.losses))
        return self.A[:, n]


class ShortFormatter(object):
    def __init__(self, has_dev, losses, ex_width=20):
        self.start_time = time.time()
        self.has_dev = has_dev
        self.ex_width = ex_width
        self.loss_names = [loss().name for loss in losses]
        self.fmt = '%10.6f  %10.6f' + ('  %10.6f' if has_dev else '') + '  %8s  %5s'
        self.fmt += '  [%s]  [%s]'
        self.fmt += '  %8.2f'
        self.last_N = 0
        extra_loss_header = '    ex/sec'
        if len(losses) > 1:
            for name in self.loss_names[1:]:
                extra_loss_header += padto('  tr_' + name, 10, right=True)
                self.fmt += '  %8.5f'
                if has_dev:
                    extra_loss_header += padto('  de_' + name, 10, right=True)
                    self.fmt += '  %8.5f'
                
        self.header = '%s %s%s  %8s  %5s%s%s' % \
              (padto(' objective', 11),
               padto('tr_' + self.loss_names[0], 10, right=True),
               ('  ' + padto('de_' + self.loss_names[0], 10, right=True)) if has_dev else '',
               'N', 'epoch',
               '  ' + padto('rand_' + ('de' if has_dev else 'tr') + '_truth', ex_width+2) +
               '  ' + padto('rand_' + ('de' if has_dev else 'tr') + '_pred',  ex_width+2),
               extra_loss_header)

    def __call__(self, obj, tr_mat, de_mat, N, epoch, is_best):
        now = time.time()
        tr_err = tr_mat.last_row()
        de_err = de_mat.last_row()
        vals = [obj, tr_err[0]]
        if self.has_dev: vals.append(de_err[0])
        vals += [N, epoch]
        examples = de_mat.examples if self.has_dev else tr_mat.examples
        vals += [padto(examples[0].output_str(), self.ex_width),
                 padto(examples[0].prediction_str(), self.ex_width)]
        vals.append((N - self.last_N) / (now - self.start_time))
        self.last_N = N
        self.start_time = now
        for i in range(1, len(self.loss_names)):
            vals.append(tr_err[i])
            if self.has_dev: vals.append(de_err[i])
        s = self.fmt % tuple(vals)
        if is_best: s += ' *'
        return s


class LongFormatter(object):
    def __init__(self, has_dev, losses, ex_width=None):
        self.start_time = time.time()
        self.has_dev = has_dev
        self.ex_width = ex_width
        self.loss_names = [loss().name for loss in losses]
        self.header = None
        self.last_N = 0

    def __call__(self, obj, tr_mat, de_mat, N, epoch, is_best):
        now = time.time()
        tr_err = tr_mat.last_row()
        de_err = de_mat.last_row()
        s = ''
        s += '-' * 80 + '\n'
        s += '  example %11s%s\n    epoch %11s\nobjective  %10.6f\n   ex/sec  %10.6f' % (N, ' *' if is_best else '', epoch, obj, (N - self.last_N) / (now - self.start_time))
        self.start_time = now
        self.last_N = N
        s += '\n'
        s += '    train'
        for i in range(len(self.loss_names)):
            if i > 0: s += '  |'
            s += '  %10.6f %s' % (tr_err[i], self.loss_names[i])
        s += '\n'
        if self.has_dev:
            s += '      dev'
            for i in range(len(self.loss_names)):
                if i > 0: s += '  |'
                s += '  %10.6f %s' % (de_err[i], self.loss_names[i])
            s += '\n'

        s += '\nTRAIN EXAMPLES\n'
        for i, (ex, inp, out) in enumerate(tr_mat.examples):
            ii = ' ' * max(0, 3+len(str(len(tr_mat.examples)-1))-len(str(i)))
            if inp is not None:
                s += ii + 'input%d  %s\n' % (i, padto(inp, None))
            s += ii + 'truth%d  %s\n' % (i, padto(ex, None))
            s += ii + ' pred%d  %s\n' % (i, padto(out, None))
            s += '\n'
            
        if self.has_dev:
            s += '\nDEV EXAMPLES\n'
            for i, (ex, inp, out) in enumerate(de_mat.examples):
                ii = ' ' * max(0, 3+len(str(len(de_mat.examples)-1))-len(str(i)))
                if inp is not None:
                    s += ii + 'input%d  %s\n' % (i, padto(inp, None))
                s += ii + 'truth%d  %s\n' % (i, padto(ex, None))
                s += ii + ' pred%d  %s\n' % (i, padto(out, None))
                s += '\n'
        return s


class TrainLoop(object):
    # int k = checkpoint after every k batches
    def __init__(self,
                 mk_env,
                 policy,
                 learner,
                 optimizer,
                 losses,      # one or more losses, first is used for early stopping
                 minibatch_size=1,
                 run_per_batch=(),
                 run_per_epoch=(),
                 print_freq=2.0,   # int=additive, float=multiplicative
                 print_per_epoch=True,
                 gradient_clip=None,
                 quiet=False,
                 reshuffle=True,
                 returned_parameters='best',  # { best, last, none }
                 save_best_model_to=None,
                 bandit_evaluation=False,
                 n_random_train=5,
                 n_random_dev=5,
                 mk_formatter=ShortFormatter,
                 progress_bar=True,
                 checkpoint_per_batch=None,):
        assert mk_env is not None, 'trainloop expects an mk_env'
        assert policy is not None, 'trainloop expects a policy'
        assert learner is not None, 'trainloop expects a learner'
        assert losses is not None, 'must specify at least one loss function'
        assert optimizer is not None, 'need an optimizer'
        if not isinstance(losses, list):
            losses = [losses]

        self.mk_env = mk_env
        self.policy = policy
        self.optimizer = optimizer
        self.losses = losses
        self.minibatch_size = minibatch_size
        self.run_per_batch = run_per_batch
        self.run_per_epoch = run_per_epoch
        self.print_freq = print_freq
        self.print_per_epoch = print_per_epoch
        self.gradient_clip = gradient_clip
        self.quiet = quiet
        self.reshuffle = reshuffle
        self.returned_parameters = returned_parameters
        self.save_best_model_to = save_best_model_to
        self.bandit_evaluation = bandit_evaluation
        self.n_random_train = n_random_train
        self.n_random_dev = n_random_dev
        self.mk_formatter = mk_formatter
        self.progress_bar = progress_bar
        self.checkpoint_per_batch = checkpoint_per_batch
        self.learning_alg = learner if isinstance(learner, macarico.LearningAlg) else LearnerToAlg(learner, policy,
                                                                                                   losses[0])
        
        self.tr_loss_matrix = LossMatrix(n_random_train, losses)
        self.de_loss_matrix = LossMatrix(n_random_dev, losses)
        self.print_history = []

        self.max_n_eval_train = 50

        self.last_print = None
        self.best_de_err = float('inf')
        self.final_parameters = None

        self.objective_average = Averaging()

        self.optimizer_parameters = None
        if optimizer is not None:
            self.optimizer_parameters = []
            for pg in optimizer.param_groups:
                if 'params' in pg:
                    self.optimizer_parameters += pg['params']

        self.N = 0  # total number of examples seen
        self.N_last = 0
        self.erasable = None

    def setup_minibatching(self, batch):
        # find all static features in the policy; TODO cache this list in the policy
        # TODO there's gonna be an issue with multitask policies where only some features run on certain examples :(
        computed_modules = set()
        for module in self.policy.modules():
            if isinstance(module, macarico.StaticFeatures) and id(module) not in computed_modules:
                computed_modules.add(id(module))
                module.forward_batch(batch)

    def print_it(self, string, *args, **kwargs):
        if not self.quiet:
            assert self.erasable is None
            print(string, *args, file=sys.stderr, **kwargs)
        self.print_history.append((string, args, kwargs))

    def print_it_erasable(self, string):
        if self.quiet: return
        assert self.erasable is None
        self.erasable = len(string)
        print(string, file=sys.stderr, end='')

    def erase_it(self):
        if self.quiet: return
        assert self.erasable is not None
        print('\r' + (' ' * self.erasable) + '\r', file=sys.stderr, end='')
        self.erasable = None
        
    def next_print(self):
        if self.print_freq is None:
            return None
        if self.N < 1:
            return 1
        N2 = (self.N + self.print_freq) if isinstance(self.print_freq, int) else (self.N * self.print_freq)
        if self.n_training_ex is not None and self.print_per_epoch:
            N2 = min(N2, self.N + self.n_training_ex)
        return N2

    def train(self,
              training_data,
              dev_data=None,
              n_epochs=1,
              resume_from_checkpoint=None,
             ):

        if self.bandit_evaluation and n_epochs > 1 and not self.quiet:
            self.print_it('warning: running bandit mode with n_epochs>1, this is weird!')
        self.erasable = None
        if dev_data is not None and len(dev_data) == 0:
            dev_data = None

        self.formatter = self.mk_formatter(dev_data is not None, self.losses)
        if self.formatter.header is not None:
            self.print_it(self.formatter.header)
            
        # TODO: handle RL-like things with training_data=None and n_epoch=num reps
        self.n_training_ex = len(training_data) if isinstance(training_data, list) else None

        self.n_epochs = n_epochs
        self.N_print = self.next_print()
        
        bar = None if not self.progress_bar else progressbar.ProgressBar(max_value=int(self.N_print))

        first_epoch_restored = False
        self.example_order = None
        if resume_from_checkpoint is not None:
            self.restore_checkpoint(resume_from_checkpoint)
            first_epoch_restored = True
        
        low_epoch = self.epoch if first_epoch_restored else 1
        for self.epoch in range(low_epoch, self.n_epochs+1):
            if first_epoch_restored:
                first_epoch_restored = False
                if self.example_order is not None:
                    inv_example_order = list(range(len(self.example_order)))
                    for n,i in enumerate(self.example_order):
                        inv_example_order[i] = n
                    tr_with_num = list(zip(training_data, inv_example_order))
                    tr_with_num.sort(key=lambda o: o[1])
                    training_data, eo = zip(*tr_with_num)
                minibatches = minibatch(training_data, self.minibatch_size)
                M = 0
                for batch, _ in minibatches:
                    M += len(batch)
                    if M >= self.M:
                        break
            else:
                self.M = 0  # total number of examples seen this epoch
                if self.reshuffle:
                    assert not self.bandit_evaluation
                    if self.example_order is None:
                        self.example_order = range(len(training_data))
                    tr_with_num = list(zip(training_data, self.example_order))
                    np.random.shuffle(tr_with_num)
                    training_data, self.example_order = zip(*tr_with_num)
                minibatches = minibatch(training_data, self.minibatch_size)
            for batch_id, (batch, is_last_batch) in enumerate(minibatches):
                if self.checkpoint_per_batch is not None and ((batch_id+1) % self.checkpoint_per_batch[0]) == 0:
                    self.save_checkpoint()

                self.optimizer.zero_grad()

                # when we don't know n_training_ex, we'll just be optimistic that there are
                # still >= max_n_eval_train remaining, which may cause one of the printouts to be
                # erroneous; we can correct for this later in principle if we must
                tr_eval_threshold = self.N_print
                if is_last_batch and self.n_training_ex is None:
                    self.n_training_ex = self.N + len(batch)
                if self.n_training_ex is not None:
                    tr_eval_threshold = min(tr_eval_threshold, self.n_training_ex)
                tr_eval_threshold -= self.max_n_eval_train

                # preprocess if we're minibatching
                self.policy.new_minibatch()
                batch = [self.mk_env(example) for example in batch]
                if len(batch) > 1:
                    self.setup_minibatching(batch)

                # run over each example
                total_obj = 0
                for env in batch:
                    self.N += 1
                    self.M += 1
                    if bar is not None and self.N <= self.N_print:
                        bar.update(max(1, self.N-self.N_last))

                    self.policy.new_example()
                    if not self.bandit_evaluation and self.N > tr_eval_threshold:
                        self.tr_loss_matrix.run_and_append(env, self.policy)

                    obj = self.learning_alg(env)
                    if self.bandit_evaluation:
                        self.tr_loss_matrix.append(env.example)

                    self.objective_average.update(obj if isinstance(obj, float) else obj.item())
                    total_obj += obj

                # do a gradient update/optimizer step
                if not isinstance(total_obj, float):
                    total_obj /= len(batch)
                    total_obj.backward()

                    if self.gradient_clip is not None:
                        total_norm = nn.utils.clip_grad_norm(self.optimizer_parameters, self.gradient_clip)
                    self.optimizer.step()

                # print stuff to screen and/or save the current model
                if (self.N_print is not None and self.N >= self.N_print) or \
                   (is_last_batch and (self.print_per_epoch or (self.epoch==self.n_epochs))):
                    self.do_printable_update(bar, dev_data)

                    # update the progress bar
                    if bar is not None:
                        # TODO there seems to be an off-by-k error or something on the progress bar
                        bar = progressbar.ProgressBar(max_value=int(self.N_print - self.N_last))

                # run the per_batch stuff
                for x in self.run_per_batch: x()
            # run the per_epoch stuff
            for x in self.run_per_epoch: x()

            if self.n_training_ex is None:
                self.n_training_ex = self.N

        if self.returned_parameters == 'last':
            self.final_parameters = deepcopy(self.policy.state_dict())

        return self.tr_loss_matrix, self.de_loss_matrix, self.final_parameters

    def save_checkpoint(self):
        self.print_it_erasable('checkpointing model to %s...' % self.checkpoint_per_batch[1])
        torch.save({ 'TrainLoopStatus': (self.tr_loss_matrix,
                                         self.de_loss_matrix,
                                         self.epoch,
                                         self.n_training_ex,
                                         self.example_order,
                                         self.M,
                                         self.N,
                                         self.N_print,
                                         self.N_last,
                                         self.objective_average,
                                         self.final_parameters,
                                         self.best_de_err,
                                         self.print_history,
                                         ),
                     'np_random_state': np.random.get_state(),
                     'torch_random_state': torch.random.get_rng_state(),
                     'py_random_state': py_random.getstate(),
                     'policy': self.policy.state_dict(),
                     'optimizer': self.optimizer.state_dict()
                     },
                   self.checkpoint_per_batch[1] + '.writing')
        os.rename(self.checkpoint_per_batch[1] + '.writing', self.checkpoint_per_batch[1])
        # also need to save the current optimizer state and model parameters
        self.erase_it()

    def restore_checkpoint(self, filename):
        self.print_it('restoring model from %s...' % filename)
        checkpoint = torch.load(filename)
        'TrainLoopStatus'
        (self.tr_loss_matrix,
         self.de_loss_matrix,
         self.epoch,
         self.n_training_ex,
         self.example_order,
         self.M,
         self.N,
         self.N_print,
         self.N_last,
         self.objective_average,
         self.final_parameters,
         self.best_de_err,
         self.print_history,
        ) = checkpoint['TrainLoopStatus']
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.random.set_rng_state(checkpoint['torch_random_state'])
        np.random.set_state(checkpoint['np_random_state'])
        py_random.setstate(checkpoint['py_random_state'])
        max_len = max((len(s) for s,_,_ in self.print_history))
        for string, args, kwargs in self.print_history:
            print(string + (' ' * (max_len - len(string))) + '  o_o', *args, **kwargs)
    
    def do_printable_update(self, bar, dev_data):
        self.N_last = int(self.N)
        self.N_print = self.next_print()
        if dev_data is not None:
            # TODO minibatch this
            for example in dev_data[:self.N]:
                self.policy.new_minibatch()
                self.de_loss_matrix.run_and_append(self.mk_env(example), self.policy)

        tr_err = self.tr_loss_matrix.next(self.N, self.epoch)
        de_err = self.de_loss_matrix.next(self.N, self.epoch)

        #import ipdb; ipdb.set_trace()
        #extra_loss_scores = list(itertools.chain(*zip(tr_err[1:], de_err[1:])))

        is_best = de_err[0] < self.best_de_err
        if bar is not None:
            self.print_it('\r' + ' ' * (bar.term_width) + '\r', end='')

        self.print_it(self.formatter(self.objective_average(),
                                     self.tr_loss_matrix,
                                     self.de_loss_matrix,
                                     self.N,
                                     self.epoch,
                                     is_best))
        self.objective_average.reset()

        self.last_print = self.N
        if is_best:
            self.best_de_err = de_err[0]
            if self.save_best_model_to is not None:
                self.print_it_erasable('saving model to %s...' % self.save_best_model_to)
                torch.save(self.policy.state_dict(), self.save_best_model_to)
                self.erase_it()
            if self.returned_parameters == 'best':
                self.final_parameters = deepcopy(self.policy.state_dict())
        
    
def test_reference_on(mk_env, ref, loss, example, verbose=True, test_values=False, except_on_failure=True):

    def run(run_strategy):
        env = mk_env(example)
        #env.rewind(None)
        runner = EpisodeRunner(None, run_strategy, ref, store_ref_costs=True)
        env.run_episode(runner)
        cost = loss()(example)
        return cost, runner.trajectory, runner.limited_actions, runner.costs, runner.ref_costs, example.Yhat

    # generate the backbone by REF
    loss0, traj0, limit0, costs0, refcosts0, ref_parsetree = run(lambda t: EpisodeRunner.REF)
    if verbose:
        print('loss0', loss0, 'traj0', traj0)

    n_actions = mk_env(example).n_actions
    backbone = lambda t: (EpisodeRunner.ACT, traj0[t])
    any_fail = False
    pred_trees = {}
    for t in range(len(traj0)):
        costs = torch.zeros(n_actions)
        traj1_all = [None] * n_actions
        for a in limit0[t]:
            l, traj1, _, _, _, pt1 = run(one_step_deviation(len(traj0), backbone, lambda _: EpisodeRunner.REF, t, a))
            pred_trees[t,a] = pt1
            costs[a] = l
            traj1_all[a] = traj1
            if l < loss0 or (a == traj0[t] and l != loss0):
                print('local opt failure, ref loss=%g, loss=%g on deviation (%d, %d), traj0=%s traj\'=%s [ontraj=%s, is_proj=%s]' % \
                    (loss0, l, t, a, traj0, traj1, a == traj0[t], not example.is_non_projective))
                any_fail = True
                if except_on_failure:
                    raise Exception()
        if test_values:
            for a in limit0[t]:
                if refcosts0[t][a] != costs[a]:
                    print('cost failure, t=%d, a=%d, traj0=%s, traj1=%s, ref_costs=%s, observed costs=%s [is_proj=%s]' % \
                          (t, a, traj0, traj1_all[a], [refcosts0[t][a0] for a0 in limit0[t]],
                           [costs[a0] for a0 in limit0[t]], not example.is_non_projective))
                    if except_on_failure:
                        assert False

    if not any_fail:
        print('passed!')


def test_reference(mk_env, ref, loss, data, verbose=False, test_values=False, except_on_failure=True):
    for n, example in enumerate(data):
        print('# example %d ' % n,)
        test_reference_on(mk_env, ref, loss, example, verbose, test_values, except_on_failure)


def sample_action_from_probs(r, probs):
    r0 = r
    for i, v in enumerate(probs):
        r -= v
        if r <= 0:
            return i
    _, mx = probs.max(0)
    print('warning: sampling from %s failed! returning max item %d; (r=%g r0=%g sum=%g)' %
          (str(probs), mx, r, r0, probs.sum()), file=sys.stderr)
    return len(probs)-1


def sample_from_np_probs(np_probs):
    r = np.random.rand()
    a = sample_action_from_probs(r, np_probs)
    return a, np_probs[a]


def sample_from_probs(probs):
    r = np.random.rand()
    a = sample_action_from_probs(r, probs.data)
    return a, probs[a]
