from __future__ import division
import numpy as np
import random
import torch
from macarico.annealing import ExponentialAnnealing

from macarico.lts.reinforce import Reinforce
from macarico.lts.dagger import DAgger
from macarico.annealing import EWMA
from macarico.tasks.sequence_labeler import SequenceLabeling, BiLSTMFeatures
from macarico import LinearPolicy


def re_seed(seed=90210):
    random.seed(seed)
    torch.manual_seed(seed)

re_seed()


def evaluate(data, policy):
    errors = 0.0
    for words, labels in data:
        env = SequenceLabeling(words)
        loss = env.loss_function(labels)
        env.run_episode(policy)
        errors += loss() / env.T
    return errors / len(data)


def test1():
    print
    print 'Running test 1'
    print '=============='

    task = 1
    if task == 0:
        # Sequence reversal task
        T = 5
        data = []
        for _ in range(100):
            x = [random.choice(range(5)) for _ in range(T)]
            y = list(reversed(x))
            data.append((x,y))

    elif task == 1:
        # Memoryless task, y[t] = (x[t]+1) % K
        T = 5
        K = 3
        data = []
        for _ in range(50):
            x = np.random.randint(K, size=T)
            y = (x+1) % K
            data.append((x,y))

    random.shuffle(data)
    m = int(np.ceil(len(data)/2))
    train = data[:m]
    dev = data[m:]

    n_words = len({x for X, _ in data for x in X})
    n_labels = len({y for _, Y in data for y in Y})

    print 'n_train: %s, n_dev: %s' % (len(train), len(dev))
    print 'n_words: %s, n_labels: %s' % (n_words, n_labels)

    policy = LinearPolicy(BiLSTMFeatures(n_words, n_labels), n_labels)

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)

    _p_rollin_ref = ExponentialAnnealing(0.99)

    baseline = EWMA(0.8)

    for epoch in range(500):
        for words,labels in train:
            env = SequenceLabeling(words)
            loss = env.loss_function(labels)

            if 1:
                p_rollin_ref = lambda: epoch==0 #random.random() <= _p_rollin_ref(epoch)
                learner = DAgger(loss.reference, policy, p_rollin_ref)
            else:
                learner = Reinforce(policy, baseline)

            optimizer.zero_grad()
            env.run_episode(learner)

#            from arsenal.viz import lc
#            lc['learner_obj'].update(None, log_objective=np.log(learner.objective.data[0]))

            learner.update(loss() / env.T)
            optimizer.step()

        if epoch % 1 == 0:
            if dev:
                a = evaluate(train, policy)
                b = evaluate(dev, policy)
#                from arsenal.viz import lc
#                lc['learning'].update(None, train=a, dev=b)
                print 'error rate: train %g, dev: %g' % (a,b)
            else:
                print 'error rate: train %g' % evaluate(train, policy)


if __name__ == '__main__':
    test1()



#def hash_list(*l):
#    x = 431801
#    for y in l:
#        x = int((x + y) * 849107)
#    return x
#
#
#def noisy_label(y, n_labels, noise_level):
#    if random.random() < noise_level:
#        return random.randint(0,n_labels-1)
#    return y % n_labels
#
#
#def make_xor_data(n_words, n_labels, n_ex, sent_len, history_length, noise_level=0.1):
#    training_data = []
#    for _ in range(n_ex):
#        words,labels = [],[]
#        words.append(random.randint(0,n_words-1))
#        labels.append(noisy_label(words[-1], n_labels, noise_level))
#        for _ in range(sent_len-1):
#            words.append(random.randint(0,n_words-1))
#            hist = hash_list(words[-1], *labels[-history_length:])
#            labels.append(noisy_label(hist, n_labels, noise_level))
#        training_data.append((words,labels))
#    return training_data
#
#
#def train_test(n_words, n_labels, training_data, dev_data, test_data, n_epochs, batch_size,
#               d_emb, d_rnn, d_actemb, d_hid, lr, mk_lts, mk_lts_args={}):
#    task = SequenceLabeler(n_words,
#                           n_labels,
#                           HammingReference,
#                           d_emb = d_emb,
#                           d_rnn = d_rnn,
#                           d_actemb = d_actemb,
#                           d_hid = d_hid)
#
#    #lts = DAgger(p_rollin_ref=NoAnnealing(1.))
#    lts = mk_lts(**mk_lts_args)
#    optimizer  = optim.Adam(task.parameters(), lr=lr)
#
#    def eval_on(data):
#        err = 0.
#        for words,labels in data:
#            torch_words = Variable(torch.LongTensor(words))
#            pred = task.forward(torch_words)  # no labels ==> test mode
#            this_err = sum([a!=b for a,b in zip(pred,labels)])
#            #this_err2 = task.ref_policy.final_loss()
#            #print task.ref_policy.truth, task.ref_policy.prediction
#            #assert this_err2 == this_err, 'mismatch %g != %g' % (this_err, this_err2)
#            #print this_err
#            err += this_err
#        return err / len(data)
#
#    # train
#    best = None
#    for epoch in range(n_epochs):
#        obj_value = 0.
#        for n in range(0, len(training_data), batch_size):
#            optimizer.zero_grad()
#            lts.zero_objective()
#            for words,labels in training_data[n:n+batch_size]:
#                torch_words = Variable(torch.LongTensor(words))
#                output = task.forward(torch_words, labels, lts)
#                lts.backward()
#                #obj = lts.get_objective()
#                #obj_value += obj.data[0]
#                #obj /= batch_size
#                #obj.backward()#retain_variables=True)
##            obj = lts.get_objective()
##            obj_value += obj.data[0]
##            obj /= batch_size
##            obj.backward()#retain_variables=True)
#            optimizer.step()
#        lts.new_pass()
#        if epoch % 10 == 0:
#            [tr,de,te] = map(eval_on, [training_data, dev_data, test_data])
#            if best is None or de < best[0]: best = (de,te)
#            print 'ep %d\ttr %g\tde %g\tte %g\tte* %g\tob %g' % (epoch, tr, de, te, best[1],
#                                                                 obj_value / len(training_data))
#
#
#def test2(n_words = 20,
#          n_labels = 5,
#          n_ex = 100,
#          sent_len = 5,
#          history_length = 1,
#          noise_level = 0.,
#          n_epochs = 1000,
#          batch_size = 1,
#          d_emb = 15,
#          d_rnn = 15,
#          d_actemb = 15,
#          d_hid = 15,
#          lr = 1e-2,
#          lts = MaximumLikelihood,
#          lts_args = {},
#          reseed=True):
#
#    print
#    print 'Running test 2'
#    print '=============='
#    if reseed: re_seed()
#
#    all_data = make_xor_data(n_words, n_labels, n_ex*3,
#                             sent_len, history_length, noise_level)
#
#    training_data = all_data[:n_ex]
#    dev_data      = all_data[n_ex:n_ex*2]
#    test_data     = all_data[2*n_ex:]
#
#    train_test(n_words, n_labels, training_data, dev_data, test_data, n_epochs, batch_size,
#               d_emb, d_rnn, d_actemb, d_hid, lr, lts, lts_args)


if __name__ == '__main__':
    test1()
#    test2()
