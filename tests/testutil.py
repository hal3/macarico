from __future__ import division
import random
import sys
import itertools
from copy import deepcopy
import numpy as np
import torch

def reseed(seed=90210):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

class CustomEvaluator(object):
    def __init__(self, name, corpus_level=False, maximize=False):
        self.name = name
        self.corpus_level = corpus_level
        self.maximize = maximize
        self.count = 0
        self.total = 0

    def evaluate(self, truth, prediction):
        raise NotImplementedError('abstract')

    def reset(self):
        self.count = 0
        self.total = 0

    def __call__(self, truth, prediction):
        val = self.evaluate(truth, prediction)
        if self.corpus_level:
            self.total = val
            self.count = 1
        else:
            self.total += val
            self.count += 1
        return self.get()

    def get(self):
        return self.total / self.count

def evaluate(data, policy, custom_evaluators=[], verbose=False):
    "Compute average `loss()` of `policy` on `data`"
    for evaluator in custom_evaluators:
        evaluator.reset()
    scores = [0] * (1 + len(custom_evaluators))
    N = len(data)
    for example in data:
        env = example.mk_env()
        res = env.run_episode(policy)
        if verbose: print res, example
        scores[0] += env.loss() / N
        for evaluator in custom_evaluators:
            evaluator(example, res)
    for i, evaluator in enumerate(custom_evaluators):
        scores[i+1] = evaluator.get()
    if len(custom_evaluators) == 0:
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


def trainloop(training_data,
              dev_data=None,
              policy=None,
              Learner=None,
              learning_alg=None,
              optimizer=None,
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
              custom_evaluators=[],
             ):

    assert (Learner is None) != (learning_alg is None), \
        'trainloop expects exactly one of Learner / learning_alg'

    if learning_alg is None:
        def learning_alg(X):
            env = X.mk_env()
            learner = Learner(env.reference())
            env.run_episode(learner)
            loss = env.loss()
            learner.update(loss)
            return loss

    custom_format = ''
    if not quiet:
        custom_header = ''
        if len(custom_evaluators) > 0:
            custom_header += ' ' * 9
        for evaluator in custom_evaluators:
            custom_header += padto('  tr_' + evaluator.name, 10)
            custom_header += padto('  de_' + evaluator.name, 10)
            custom_format += '  %-8.5f  %-8.5f'
        print >>sys.stderr, '%s      %s      %8s  %5s  rand_dev_truth          rand_dev_pred%s' % \
            ('tr_err', 'de_err', 'N', 'epoch', custom_header)

    last_print = None
    best_de_err = float('inf')
    final_parameters = None
    error_history = []
    num_batches = len(training_data) // minibatch_size
    N = 0  # total number of examples seen
    total_loss = 0 # total training loss so far
    maximize = False
    if len(custom_evaluators) > 0:
        maximize = custom_evaluators[-1].maximize
        if maximize:
            best_de_err = -best_de_err
    for epoch in xrange(1,n_epochs+1):
        M = 0  # total number of examples seen this epoch
        for batch in minibatch(training_data, minibatch_size, reshuffle):
            if optimizer is not None:
                optimizer.zero_grad()
            # TODO: minibatching is really only useful if we can
            # preprocess in a useful way
            for ex in batch:
                N += 1
                M += 1
                if print_dots and (num_batches <= 40 or M % (num_batches//40) == 0):
                    sys.stderr.write('.')
                total_loss += learning_alg(ex)
            if optimizer is not None:
                optimizer.step()

            if not quiet and (should_print(print_freq, last_print, N) or \
                              (print_per_epoch and M >= len(training_data))):
                tr_err = [0] * (len(custom_evaluators)+1) if train_eval_skip is None else \
                         evaluate(training_data[::train_eval_skip], policy, custom_evaluators)
                de_err = [0] * (len(custom_evaluators)+1) if dev_data is None else \
                         evaluate(dev_data, policy, custom_evaluators)

                if not isinstance(tr_err, list): tr_err = [tr_err]
                if not isinstance(de_err, list): de_err = [de_err]
                
                custom_scores = list(itertools.chain(*zip(tr_err[1:], de_err[1:])))
                error_history.append((tr_err, de_err))

                random_dev_truth, random_dev_pred = '', ''
                if dev_data is not None:
                    ex = random.choice(dev_data)
                    random_dev_truth = ex
                    random_dev_pred  = ex.mk_env().run_episode(policy)

                if print_dots:
                    sys.stderr.write('\r')

                fmt = '%-10.6f  %-10.6f  %8s  %5s  [%s]  [%s]' + custom_format
                is_best = (de_err[-1] < best_de_err and not maximize) or \
                          (de_err[-1] > best_de_err and maximize)
                if is_best:
                    fmt += '  *'
                fmt_vals = [tr_err[0], de_err[0], N, epoch,
                            padto(random_dev_truth, 20), padto(random_dev_pred, 20)] + \
                           custom_scores
                print >>sys.stderr, fmt % tuple(fmt_vals)
                
                last_print = N
                if is_best:
                    best_de_err = de_err[-1]
                    if returned_parameters == 'best':
                        final_parameters = deepcopy(optimizer.param_groups)

            for x in run_per_batch: x()
        for x in run_per_epoch: x()

    if returned_parameters == 'last':
        final_parameters = deepcopy(optimizer.param_groups)
        
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
