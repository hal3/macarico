from __future__ import division
import numpy as np
import random
import dynet as dy
import sys
import json
import macarico.util
from collections import Counter
import pickle
import glob

macarico.util.reseed()

import macarico.lts.lols
reload(macarico.lts.lols)

from macarico.data import nlp_data
from macarico.annealing import ExponentialAnnealing, NoAnnealing, stochastic, EWMA
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.dagger import DAgger
from macarico.lts.lols import BanditLOLS, BanditLOLSMultiDev, BanditLOLSRewind
from macarico.tasks.sequence_labeler import Example, HammingLoss, HammingLossReference
from macarico.tasks.seq2seq import EditDistance, EditDistanceReference
from macarico.features.sequence import RNNFeatures, BOWFeatures, AttendAt, SoftmaxAttention, FrontBackAttention
from macarico.features.actor import TransitionRNN, TransitionBOW
from macarico.policies.linear import LinearPolicy
from macarico.policies.active import CSActive
from macarico.lts.dagger import DAgger
from macarico.lts.reinforce import Reinforce, AdvantageActorCritic, LinearValueFn
from macarico.tasks.dependency_parser import DependencyAttention, AttachmentLoss, AttachmentLossReference


names = 'blols_1 blols_2 blols_3 blols_4 blols_1_learn blols_2_learn blols_3_learn blols_4_learn blols_1_bl blols_3_bl blols_4_bl blols_1_pref blols_2_pref blols_3_pref blols_4_pref blols_1_pref_os blols_2_pref_os blols_3_pref_os blols_4_pref_os blols_1_pref_learn blols_2_pref_learn blols_3_pref_learn blols_4_pref_learn blols_1_pref_learn_os blols_2_pref_learn_os blols_3_pref_learn_os blols_4_pref_learn_os reinforce reinforce_nobl reinforce_md1 reinforce_uni reinforce_md1_uni reinforce_md1_nobl reinforce_uni_nobl reinforce_md1_uni_nobl'.split()

def dumpit():
    pickle.dump([(name, globals()[name]) for name in names if name in globals()], open('big_banditlols.new.saved','w'))

##############################################################################
## SETUP UP DATASETS
##############################################################################

def setup_mod(dy_model, n_train=50, n_dev=100, n_types=10, n_labels=4, length=6):
    data = macarico.util.make_sequence_mod_data(n_train+n_dev, length, n_types, n_labels)
    data = [Example(x, y, n_labels) for x, y in data]
    train = data[n_dev:]
    dev = data[:n_dev]
    attention = lambda _: [AttendAt()]
    reference = HammingLossReference()
    losses = [HammingLoss()]
    mk_feats = lambda fb: [fb(dy_model, n_types)]
    return train, dev, attention, reference, losses, mk_feats, n_labels

def setup_sequence(dy_model, filename, n_train, n_dev):
    train, dev, test, token_vocab, label_id = nlp_data.read_wsj_pos(filename, n_tr=n_train, n_de=n_dev, n_te=0, min_freq=2)
    attention = lambda _: [AttendAt()]
    reference = HammingLossReference()
    losses = [HammingLoss()]
    n_labels = len(label_id)
    n_types = len(token_vocab)
    mk_feats = lambda fb: [fb(dy_model, n_types)]
    return train, dev, attention, reference, losses, mk_feats, n_labels

def setup_deppar(dy_model, filename, n_train, n_dev):
    train, dev, test, token_vocab, pos_vocab, rel_id = nlp_data.read_wsj_deppar(filename, n_tr=n_train, n_de=n_dev, n_te=0, min_freq=2)
    attention = lambda _: [DependencyAttention(),
                           DependencyAttention(field='pos_rnn')]
    reference = AttachmentLossReference()
    losses = [AttachmentLoss()]
    n_types = len(token_vocab)
    n_labels = 3 + len(rel_id)
    mk_feats = lambda fb: [fb(dy_model, n_types),
                           fb(dy_model, n_types, input_field='pos', output_field='pos_rnn')]
    return train, dev, attention, reference, losses, mk_feats, n_labels

def setup_translit(dy_model, filename, n_dev):
    [filename_src, filename_tgt] = filename.split(':')
    train, dev, src_voc, tgt_voc = nlp_data.read_parallel_data(filename_src, filename_tgt, n_de=n_dev, min_src_freq=2, shuffle=True)
    attention = lambda features: [SoftmaxAttention(dy_model, features, 50)]
    n_types = len(src_voc)
    n_labels = len(tgt_voc)
    reference = EditDistanceReference()
    losses = [EditDistance()]
    mk_feats = lambda fb: [fb(dy_model, n_types)]
    return train, dev, attention, reference, losses, mk_feats, n_labels

##############################################################################
## SETUP UP LEARNING ALGORITHMS
##############################################################################

def setup_banditlols(dy_model, learning_method):
    learning_method = learning_method.split('::')
    update_method = \
      BanditLOLS.LEARN_IPS    if 'ips'    in learning_method else \
      BanditLOLS.LEARN_BIASED if 'biased' in learning_method else \
      BanditLOLS.LEARN_DR     if 'dr'     in learning_method else \
      BanditLOLS.LEARN_MTR    if 'mtr'    in learning_method else \
      BanditLOLS.LEARN_MTR_ADVANTAGE if 'mtra' in learning_method else \
      None
    exploration_method = \
      BanditLOLS.EXPLORE_UNIFORM if 'uniform' in learning_method else \
      BanditLOLS.EXPLORE_BOLTZMANN if 'boltzmann' in learning_method else \
      BanditLOLS.EXPLORE_BOLTZMANN_BIASED if 'biasedboltz' in learning_method else \
      None
    temperature = 1.0
    use_prefix_costs = 'upc' in learning_method
    offset_t = 'oft' in learning_method
    p_rin = 0.
    p_rout = 0.
    exploit = 1.
    for x in learning_method:
        if   x.startswith('p_rin='): p_rin = float(x[5:])
        elif x.startswith('p_rout='): p_rout = float(x[6:])
        elif x.startswith('temp='): temperature = float(x[5:])
        elif x.startswith('exploit='): exploit = float(x[8:])
        else: assert '=' not in x, 'unknown arg: ' + x
    
    p_rollin_ref  = stochastic(ExponentialAnnealing(p_rin))
    p_rollout_ref = stochastic(NoAnnealing(p_rout))
    run_per_batch = [p_rollout_ref.step, p_rollin_ref.step]
    if 'annealeps' in learning_method:
        exploit = stochastic(ExponentialAnnealing(exploit))
        run_per_batch.append(exploit.step)

    BLOLS = BanditLOLSMultiDev if 'multidev' in learning_method else \
            BanditLOLS
    builder = lambda reference, policy: \
        BLOLS(reference, policy, p_rollin_ref, p_rollout_ref,
              update_method, exploration_method,
              temperature=temperature,
              use_prefix_costs=use_prefix_costs, exploit=exploit,
              offset_t=offset_t)
        
    return builder, run_per_batch

def setup_reinforce(dy_model, learning_method):
    learning_method = learning_method.split('::')
    baseline = 0.8
    max_deviations = None
    for x in learning_method:
        if   x.startswith('baseline='): baseline = float(x[9:])
        elif x.startswith('maxd='): max_deviations = int(x[5:])
        else: assert '=' not in x, 'unknown arg: ' + x        
    baseline = EWMA(baseline)
    return lambda _, policy: \
        Reinforce(policy, baseline, max_deviations=max_deviations), \
        []

def setup_aac(dy_model, learning_method):
    def builder(reference, policy):
        baseline = LinearValueFn(dy_model, policy.features)
        policy.vfa = baseline
        return AdvantageActorCritic(policy, baseline)

    return builder, []

def setup_dagger(dy_model, learning_method):
    learning_method = learning_method.split('::')
    p_rin = 0.
    for x in learning_method:
        if x.startswith('p_rin='): p_rin = float(x[5:])
        else: assert '=' not in x, 'unknown arg: ' + x
    p_rollin_ref  = stochastic(ExponentialAnnealing(p_rin))
    return lambda reference, policy: \
        DAgger(reference, policy, p_rollin_ref), \
        [p_rollin_ref.step]

def setup_aggrevate(dy_model, learning_method):
    learning_method = learning_method.split('::')
    p_rin = 0.
    for x in learning_method:
        if x.startswith('p_rin='): p_rin = float(x[5:])
        else: assert '=' not in x, 'unknown arg: ' + x
    p_rollin_ref  = stochastic(ExponentialAnnealing(p_rin))
    return lambda reference, policy: \
        AggreVaTe(reference, policy, p_rollin_ref), \
        [p_rollin_ref.step]


##############################################################################
## RUN EXPERIMENTS
##############################################################################

#def test1(learning_method, exploration, N=50, n_types=10, n_labels=4, length=6, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=None, use_prefix_costs=False, epsilon=1.0, offset_t=False, learning_rate=0.001, loss_fn='squared', task='mod'):
def run(task='mod::160::4::20', \
        learning_method='blols::',
        opt_method='adadelta',
        learning_rate=0.01,
        bow=False,
        active=False,
        supervised=False,
       ):
    print >>sys.stderr, ''
    #print >>sys.stderr, '# testing learning_method=%d exploration=%d' % (learning_method, exploration)
    print >>sys.stderr, '# %s' % locals()
    print >>sys.stderr, ''

    dy_model = dy.ParameterCollection()

    task_args = task.split('::')
    task = task_args[0]
    task_args = task_args[1:]

    train, dev, attention, reference, losses, mk_feats, n_labels = \
      setup_mod(dy_model, 65536, 100, int(task_args[0]), int(task_args[1]), int(task_args[2])) if task == 'mod' else \
      setup_sequence(dy_model, task_args[0], int(task_args[1]), int(task_args[2])) if task == 'seq' else \
      setup_deppar(dy_model, task_args[0], int(task_args[1]), int(task_args[2])) if task == 'dep' else \
      setup_translit(dy_model, task_args[0], int(task_args[1])) if task == 'trn' else \
      None

    feature_builder = BOWFeatures if bow else RNNFeatures
    transition_builder = TransitionBOW if bow else TransitionRNN
    
    features = mk_feats(feature_builder)
    transition = transition_builder(dy_model, features, attention(features), n_labels)
    policy = LinearPolicy(dy_model, transition, n_labels, loss_fn='huber')
    if active:
        policy = CSActive(policy)

    mk_learner, run_per_batch = \
      setup_banditlols(dy_model, learning_method) if learning_method.startswith('blols') else \
      setup_reinforce(dy_model, learning_method) if learning_method.startswith('reinforce') else \
      setup_aac(dy_model, learning_method) if learning_method.startswith('aac') else \
      setup_dagger(dy_model, learning_method) if learning_method.startswith('dagger') else \
      setup_aggrevate(dy_model, learning_method) if learning_method.startswith('aggrevate') else \
      None

    Learner = lambda: mk_learner(reference, policy)

    optimizer = \
      dy.AdadeltaTrainer(dy_model) if opt_method == 'adadelta' else \
      dy.AdamTrainer(dy_model, alpha=learning_rate) if opt_method == 'adam' else \
      dy.AdagradTrainer(dy_model, learning_rate=learning_rate) if opt_method == 'adagrad' else \
      dy.MomentumSGDTrainer(dy_model, learning_rate=learning_rate) if opt_method == 'sgdmom' else \
      dy.RMSPropTrainer(dy_model, learning_rate=learning_rate) if opt_method == 'rmsprop' else \
      dy.SimpleSGDTrainer(dy_model, learning_rate=learning_rate) if opt_method == 'sgd' else \
      None

    if hasattr(policy, 'set_optimizer'):
        policy.set_optimizer(optimizer)

    def printit():
        #print optimizer.status()
        #if random.random() < 0.01:
        #    from arsenal import ip; ip()
        pass

    history, _ = macarico.util.trainloop(
        training_data     = train,
        dev_data          = dev,
        policy            = policy,
        Learner           = Learner,
        losses            = losses,
        optimizer         = optimizer,
        run_per_batch     = run_per_batch + [printit],
        train_eval_skip   = None,
        bandit_evaluation = not supervised,
        n_epochs          = 20 if supervised else 1,
    )
    #print json.dumps(dict(learning_method=learning_method, exploration=exploration, N=N, n_types=n_types, n_labels=n_labels, length=length, random_seed=random_seed, history=history))
    #print history
    return history

    

if __name__ == '__main__' and len(sys.argv) >= 4:
    print sys.argv
    res = run(sys.argv[1],  # task
              sys.argv[2],  # learning_method
              sys.argv[3],  # opt_method
              float(sys.argv[4]),  # learning_rate
              'bow' in sys.argv,
              'active' in sys.argv,
              'supervised' in sys.argv)
    print res
    sys.exit(0)

# if __name__ == '__main__' and len(sys.argv) > 2:
#     print sys.argv

#     learning_method = int(sys.argv[1])
#     exploration = int(sys.argv[2])
#     temperature = float(sys.argv[3])
#     baseline = float(sys.argv[4])
#     uniform = sys.argv[5] == 'True'
#     max_deviations = None if sys.argv[6] == 'None' else int(sys.argv[6])
#     use_prefix_costs = sys.argv[7] == 'True'
#     offset_t = sys.argv[8] == 'True'
#     method = sys.argv[9]
#     learning_rate = float(sys.argv[10])
#     loss_fn = sys.argv[11]
#     random_seed = int(sys.argv[12])
#     num_rep = int(sys.argv[13])

#     for _ in xrange(num_rep):
#         test1(learning_method=learning_method,
#               exploration=exploration,
#               N=65536,
#               n_types=160,
#               n_labels=4,
#               length=20,
#               random_seed=random_seed,
#               bow=False,
#               method=method,
#               temperature=temperature,
#               p_ref=0,
#               baseline=baseline,
#               uniform=uniform,
#               max_deviations=max_deviations,
#               use_prefix_costs=use_prefix_costs,
#               offset_t=offset_t,
#               learning_rate=learning_rate,
#               loss_fn=loss_fn,
#         )
#     #bow = 'bow' in sys.argv
#     #method = 'banditlols'
#     #if 'reinforce' in sys.argv: method = 'reinforce'
#     #if 'aac' in sys.argv: method = 'aac'
#     #test1(learning_method=int(sys.argv[1]),
#     #      exploration=int(sys.argv[2]),
#     #      N=int(sys.argv[3]),
#     #      n_types=int(sys.argv[4]),
#     #      n_labels=int(sys.argv[5]),
#     #      length=int(sys.argv[6]),
#     #      random_seed=int(sys.argv[7]),
#     #      bow=bow,
#     #      method=method,
#     #      temperature=float(sys.argv[8]),
#     #      p_ref=float(sys.argv[9]),
#     #)
#     sys.exit()


# for name, res in pickle.load(open('big_banditlols.new.saved')): globals()[name] = res

# if False:
#     blols_1 = test1(1, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.0, uniform=False, max_deviations=None)
#     blols_2 = test1(2, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.0, uniform=False, max_deviations=None)
#     blols_3 = test1(3, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.0, uniform=False, max_deviations=None)
#     blols_4 = test1(4, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.0, uniform=False, max_deviations=None)
    
#     blols_1_bl = test1(1, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=None)
#     blols_2_bl = test1(2, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=None)
#     blols_3_bl = test1(3, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=None)
#     blols_4_bl = test1(4, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=None)

# test1(2, 1, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='banditlols', temperature=0.2, p_ref=0, baseline=0, uniform=False, max_deviations=None, use_prefix_costs=True, offset_t=True)


# if False:
#     reinforce = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=None)
#     reinforce_nobl = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.0, uniform=False, max_deviations=None)
#     reinforce_md1 = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.8, uniform=False, max_deviations=1)
#     reinforce_uni = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.8, uniform=True, max_deviations=None)
#     reinforce_md1_uni = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.8, uniform=True, max_deviations=1)
#     reinforce_md1_nobl = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.0, uniform=False, max_deviations=1)
#     reinforce_uni_nobl = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0.0, uniform=True, max_deviations=None)
#     reinforce_md1_uni_nobl = test1(0, 0, N=65536, n_types=160, n_labels=4, length=20, random_seed=20001, bow=True, method='reinforce', temperature=1, p_ref=1, baseline=0, uniform=True, max_deviations=1)
#     dumpit()

def print_one(name, name2, X=None):
    if X is None:
        X = globals()[name]
        
    if isinstance(X, list):
        X = np.array(X).reshape(len(X),2)
    
    tail_score  = X[-1,0]
    tail_score2 = (2 ** (np.arange(17) + 1) * X[:,1] / sum(2 ** (np.arange(17) + 1))).sum()
    tail_score3 = X[-1,1]
    tail_score  = str(int(100 * tail_score ) / 100)
    tail_score2 = str(int(100 * tail_score2) / 100)
    tail_score3 = str(int(100 * tail_score3) / 100)

    name = name.replace('_1', '_ips').replace('_2', '_dir').replace('_3', '_mtr').replace('_4', '_mtA')
    
    print name, ' ' * (22-len(name)), \
        name2, ' ' * (22-len(name2)), \
        tail_score , ' ' * (5 - len(tail_score)), \
        tail_score2, ' ' * (5 - len(tail_score2)), \
        tail_score3, ' ' * (5 - len(tail_score3)), \
        ''

def read_bbl_out(fname):
    D = []
    me = None
    args = None
    for l in open(fname, 'r'):
        if 'dynet' in l: continue
        if not l.startswith('['): continue
        l = eval(l.strip())
        if l[0] == 'big_banditlols.py':
            learning_method = int(l[1])
            exploration = int(l[2])
            temperature = float(l[3])
            baseline = float(l[4])
            uniform = l[5] == 'True'
            max_deviations = None if l[6] == 'None' else int(l[6])
            use_prefix_costs = l[7] == 'True'
            offset_t = l[8] == 'True'
            method = l[9]
            learning_rate = float(l[10])
            loss_fn = l[11]
            random_seed = int(l[12])
            num_rep = int(l[13])

            learning_method = 'ips' if learning_method == 0 else 'dir' if learning_method == 1 else 'mtr' if learning_method == 2 else 'mtA'
            
            method = 'rnfrc' if method == 'reinforce' else 'blols'
            exploration = 'uni' if exploration == 0 else 'btz' if exploration == 1 else 'bzb'
            temperature = 'temp:' + str(temperature)
            baseline = 'bl' if baseline > 0 else ''
            uniform = 'uni' if uniform else ''
            max_deviations = 'md1' if max_deviations is not None else ''
            use_prefix_costs = 'pre' if use_prefix_costs else ''
            offset_t = 'oft' if offset_t else ''
            loss_fn = 'hub' if loss_fn == 'huber' else 'sqr'
            learning_rate = 'lr:' + str(learning_rate)

            if method == 'blols':
                baseline = ''
                uniform = ''
                max_deviations = ''
            else:
                learning_method = ''
                exploration = ''
                use_prefix_costs = ''
                offset_t = ''
                loss_fn = ''
            
            me = ' '.join(map(str, [x for x in [method, learning_method, exploration, baseline, uniform, max_deviations, use_prefix_costs, offset_t] if x != '']))
            args = ' '.join(map(str, [temperature, learning_rate, loss_fn]))
        else:
            l = np.array(l)[:,:,0]
            D.append(l)
    #print_one('me', x/n)
    return me, args, D
            


d = {}
for fname in glob.glob('bbl_out/*'):
    me, args, D = read_bbl_out(fname)
    l = len(D)
    if l == 0: continue
    D = np.array(D)
    X = D.mean(axis=0)
    V = D.std(axis=0)
    score = (2 ** (np.arange(17) + 1) * X[:,1] / sum(2 ** (np.arange(17) + 1))).sum()
    if me not in d or score < d[me][0]:
        d[me] = (score, args, X, V / np.sqrt(l))
        
    #d[me][args] 
#read_bbl_out('bbl_out/lols.1068')


count = {}
names0 = []
col = 1

from matplotlib.pyplot import *

fig, ax = subplots(1)

V_all = []
colors = 'brg'
print
d = sorted([(score, me, args, X, V) for me, (score, args, X, V) in d.iteritems()])
for (score, name, args, X, V) in d:
    print_one(name, args, X)

    if name not in ['rnfrc bl uni', 'rnfrc bl', 'blols mtA btz oft', 'rnfrc bl md1', 'rnfrc bl uni md1']:
        continue
    
#for name in names:
    #if 'blols' in name and '_learn' not in name: continue
    #data = globals()[name]
    #X = np.array(data).reshape(len(data),2)

    line_style = ''

    if 'rnfrc' in name: line_style = ':'
    elif 'pre' in name and 'oft' in name: line_style = '-'
    elif 'pre' in name: line_style = '--'
    else: line_style = '-.'
    
    #if 'blols'     in name and '_bl' not in name: line_style = '-'
    #if 'blols'     in name and '_bl'     in name: line_style = ':'
    #if 'blols' not in name and '_nobl'   in name: line_style = '--'
    #if 'blols' not in name and '_nobl' not in name: line_style = '-.'
    #if   'blols' in name and '_pref' in name and '_os' in name: line_style = '-'
    #elif 'blols' in name and '_pref' in name: line_style = '--'
    #elif 'blols' in name: line_style = ':'
    #else: line_style = '-.'

    #if 'reinforce' in name: continue
    #if 'blols' in name: continue

    color_id = count.get(line_style, 0)
    if color_id < len(colors):
        color = colors[color_id]
        count[line_style] = 1 + count.get(line_style, 0)

        T = np.arange(17)+1
        ax.plot(T, X[:,col], color + line_style, linewidth=4)
        ax.fill_between(T, X[:,col]+V[:,col], X[:,col]-V[:,col], facecolor=color, alpha=0.2)
        names0.append(name)

    V_all += list(V.flatten())
    #print_one(name, X)

#print sum(V_all) / len(V_all)

    
legend(names0, fontsize='xx-large')
show(True)
