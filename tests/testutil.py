from __future__ import division
import random
import sys
import numpy as np
import torch

def reseed(seed=90210):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def evaluate(data, policy, verbose=False):
    "Compute average `loss()` of `policy` on `data`"
    L = 0.0
    for example in data:
        env = example.mk_env()
        res = env.run_episode(policy)
        if verbose: print res, example
        L += env.loss()
    return L / len(data)


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
    if not isinstance(s, str):
        if isinstance(s, list):
            s = ' '.join(map(str, s))
        else:
            s = str(s)
    n = len(s)
    if n > l:
        return s[:l-2] + '..'
    return s + (' ' * (l - len(s)))


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
              ):

    assert (Learner is None) != (learning_alg is None), \
        'trainloop expects exactly one of Learner / learning_alg'

    if learning_alg is None:
        def learning_alg(X):
            env = X.mk_env()
            learner = Learner(env.reference())
            env.run_episode(learner)
            learner.update(env.loss())

    if not quiet:
        print >>sys.stderr, '%s      %s      %8s  %5s  rand_dev_truth          rand_dev_pred' % \
            ('tr_err', 'de_err', 'N', 'epoch')

    last_print = None
    best_de_err = float('inf')
    error_history = []
    if training_data is not None:
        N = 0  # total number of examples seen
        for epoch in xrange(1,n_epochs+1):
            M = 0  # total number of examples seen this epoch
            num_batches = len(training_data) // minibatch_size
            for batch in minibatch(training_data, minibatch_size, reshuffle):
                if optimizer is not None:
                    optimizer.zero_grad()
                # TODO: minibatching is really only useful if we can
                # preprocess in a useful way
                for X in batch:
                    N += 1
                    M += 1
                    if print_dots and (num_batches <= 40 or M % (num_batches//40) == 0):
                        sys.stderr.write('.')
                    learning_alg(X)
                if optimizer is not None:
                    optimizer.step()

                if not quiet and (should_print(print_freq, last_print, N) or \
                                  (print_per_epoch and M >= len(training_data))):
                    tr_err = evaluate(training_data[::train_eval_skip], policy)
                    de_err = 0. if dev_data is None else \
                             evaluate(dev_data, policy)

                    error_history.append((tr_err, de_err))

                    random_dev_truth, random_dev_pred = '', ''
                    if dev_data is not None:
                        X = random.choice(dev_data)
                        random_dev_truth = X
                        random_dev_pred  = X.mk_env().run_episode(policy)

                    if print_dots:
                        sys.stderr.write('\r')

                    print >>sys.stderr, '%-10.6f  %-10.6f  %8s  %5s  [%s]  [%s]%s' % \
                        (tr_err, de_err, N, epoch, \
                         padto(random_dev_truth, 20), padto(random_dev_pred, 20),
                         '  *' if de_err < best_de_err else '',
                         )
                    last_print = N
                    if de_err < best_de_err:
                        best_de_err = de_err

                for x in run_per_batch: x()
            for x in run_per_epoch: x()

    return error_history

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
