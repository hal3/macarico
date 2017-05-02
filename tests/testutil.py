import random
import sys
import torch

def reseed(seed=90210):
    random.seed(seed)
    torch.manual_seed(seed)

def evaluate(mk_env, data, policy, verbose=False):
    errors = 0.0
    count  = 0.0
    for words, labels in data:
        env = mk_env(words)
        loss = env.loss_function(labels)
        res = env.run_episode(loss.reference if policy is None else policy)
        if verbose: print res, labels
        errors += loss()
        count  += env.N
    return errors / count

def should_print(update_freq, last_print, N):
    if last_print is None:
        return True
    next_print = last_print + update_freq if isinstance(update_freq, int) else \
                 last_print * update_freq
    return N >= next_print

def minibatch(data, minibatch_size, reshuffle):
    # TODO this can prob be made way more efficient
    if reshuffle:
        random.shuffle(data)
    for n in range(0, len(data), minibatch_size):
        yield data[n:n+minibatch_size]

def padto(s, l):
    n = len(s)
    if n > l:
        return s[:n-2] + '..'
    return s + (' ' * (l - len(s)))

def trainloop(Env,
              training_data=None,
              dev_data=None,
              policy=None,
              Learner=None,
              optimizer=None,
              n_epochs=20,
              minibatch_size=1,
              run_per_batch=[],
              run_per_epoch=[],
              print_freq=2.0,   # int=additive, float=multiplicative
              print_per_epoch=True,
              quiet=False,
              train_eval_skip=100,
              reshuffle=True,
              ):
    last_print = None
    best_de_err = float('inf')
    error_history = []
    if training_data is not None:
        N = 0  # total number of examples seen
        for epoch in xrange(n_epochs):
            M = 0  # total number of examples seen this epoch
            for batch in minibatch(training_data, minibatch_size, reshuffle):
                if optimizer is not None:
                    optimizer.zero_grad()
                # TODO: minibatching is really only useful if we can
                # preprocess in a useful way
                for X,Y in batch:
                    env = Env(X)
                    loss = env.loss_function(Y)
                    learner = policy if Learner is None else \
                              Learner(loss.reference)
                    env.run_episode(learner)
                    learner.update(loss())
                if optimizer is not None:
                    optimizer.step()

                N += len(batch)
                M += len(batch)
                if not quiet and (should_print(print_freq, last_print, N) or \
                                  (print_per_epoch and M >= len(training_data))):
                    tr_err = evaluate(Env, training_data[::train_eval_skip], policy)
                    de_err = 0. if dev_data is None else \
                             evaluate(Env, dev_data, policy)

                    error_history.append( (tr_err, de_err) )
                    
                    if last_print is None:
                        print >>sys.stderr, '%s      %s      %8s  %5s  rand_dev_truth          rand_dev_pred' % \
                            ('tr_err', 'de_err', 'N', 'epoch')
                    
                    random_dev_truth, random_dev_pred = '', ''
                    if dev_data is not None:
                        X,Y = random.choice(dev_data)
                        random_dev_truth = str(Y)
                        random_dev_pred  = str(Env(X).run_episode(policy))
                        
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

