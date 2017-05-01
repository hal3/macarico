from __future__ import division
import random
import torch

from macarico.lts.maximum_likelihood import MaximumLikelihood
from macarico.tasks.sequence_labeler import BiLSTMFeatures
from macarico.tasks.seq2seq import Seq2Seq, Seq2SeqFoci
from macarico import LinearPolicy

from test_sequence_labeler import evaluate


def test1():
    print 'Sequence reversal task'
    # Sequence reversal task
    T = 5
    data = []
    for _ in range(1000):
        x = [random.choice(range(5)) for _ in range(T)]
        y = list(reversed([i+1 for i in x])) + [0]
        data.append((x,y))

    random.shuffle(data)
    m = len(data) // 2
    train = data[:m]
    dev = data[m:]

    n_words = len({x for X, _ in data for x in X})
    n_labels = 1+max({y for _, Y in data for y in Y})
    Env = lambda x: Seq2Seq(x, n_labels)

    print 'n_train: %s, n_dev: %s' % (len(train), len(dev))
    print 'n_words: %s, n_labels: %s' % (n_words, n_labels)
    print 'eval ref: %s' % evaluate(Env, train, None)
    print

    policy = LinearPolicy(BiLSTMFeatures(Seq2SeqFoci(), n_words, n_labels), n_labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)


    for epoch in range(500):
        for inputs,outputs in train:
            env = Env(inputs)
            loss = env.loss_function(outputs)
            learner = MaximumLikelihood(loss.reference, policy)
            optimizer.zero_grad()
            env.run_episode(learner)
            learner.update(loss())
            optimizer.step()

        if epoch % 1 == 0:
            a = evaluate(Env, train, policy)
            b = evaluate(Env, dev, policy)
            print 'error rate: train %g, dev: %g' % (a,b)


if __name__ == '__main__':
    test1()
#    test2()
