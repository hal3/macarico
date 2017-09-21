from __future__ import division

import random
import sys
#import torch
#from torch.autograd import Variable
import macarico
#import torch.nn.functional as F
import numpy as np
import dynet as dy
import macarico.util
from collections import Counter
import scipy.optimize
from macarico.annealing import Averaging

certainty_tracker = Averaging()
num_offsets = Counter()
global_times = None

def opt_alpha_kl(beta):
    k, k2 = beta.shape
    assert k == k2
    def f(alpha):
        p = beta.dot(alpha)
        return p.dot(np.log(p + 1e-6))

    def g(alpha):
        return alpha.sum() - 1
    
    return scipy.optimize.minimize(f,
                                   x0=np.ones(k) / k,
                                   bounds=[(0,1)] * k, \
                                   constraints={'type': 'eq', 'fun': g}).x

def compute_time_distribution(T, delays):
    beta = np.zeros((T,T))
    for i in xrange(T):
        for j in xrange(i, T):
            beta[j,i] = delays.get(j-i, 0)
        beta[i,i] += 1e-6
        beta[:,i] /= sum(beta[:,i])
    alpha = opt_alpha_kl(beta)
    return alpha, beta.dot(alpha)

class BanditLOLS(macarico.Learner):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1
    LEARN_BIASED, LEARN_IPS, LEARN_DR, LEARN_MTR, LEARN_MTR_ADVANTAGE, _LEARN_MAX = 0, 1, 2, 3, 4, 5
    EXPLORE_UNIFORM, EXPLORE_BOLTZMANN, EXPLORE_BOLTZMANN_BIASED, _EXPLORE_MAX = 0, 1, 2, 3

    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=LEARN_IPS,
                 exploration=EXPLORE_UNIFORM, baseline=None,
                 epsilon=1.0, mixture=MIX_PER_ROLL, use_prefix_costs=False,
                 temperature=1., offset_t=False):
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.learning_method in range(BanditLOLS._LEARN_MAX), \
            'unknown learning_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        
        self.use_prefix_costs = use_prefix_costs
        if mixture == BanditLOLS.MIX_PER_ROLL:
            use_in_ref  = p_rollin_ref()
            use_out_ref = p_rollout_ref()
            self.rollin_ref  = lambda: use_in_ref
            self.rollout_ref = lambda: use_out_ref
        else:
            self.rollin_ref  = p_rollin_ref
            self.rollout_ref = p_rollout_ref
        self.baseline = baseline
        self.epsilon = epsilon
        self.t = None
        self.dev_t = None
        self.dev_a = None
        self.dev_actions = None
        self.dev_imp_weight = None
        self.dev_costs = None
        self.squared_loss = 0.
        self.pred_act_cost = None
        #self.pred_cost_total = 0.
        #self.pred_cost_until_dev = 0.
        #self.pred_cost_dev = 0.
        self.deviated = False
        self.offset_t = offset_t
        self.this_num_offsets = 0
        
        super(BanditLOLS, self).__init__()

    def __call__(self, state):
        global global_times, certainty_tracker, num_offsets
        if self.t is None:
            self.t = 0
            if global_times is None or np.random.random() < 0.01:
                global_times, _ = compute_time_distribution(state.T, num_offsets)
            self.dev_t = np.random.choice(range(len(global_times)), p=global_times) + 1
            #self.dev_t = np.random.randint(0, state.T) + 1
            self.pred_act_cost = []

        self.t += 1

        costs = self.policy.predict_costs(state).npvalue()
        costs_idx = costs.argsort()
        certainty = costs[costs_idx[1]] - costs[costs_idx[0]]
        certainty_tracker.update(certainty)
        #if np.random.random() < 0.001: print certainty_tracker()
        #certainty = costs.max() - costs.min()
        #certainty_threshold = 100.5
        certainty_threshold = certainty_tracker()
        #if np.random.random() < 0.001: print self.certainty_tracker()
        #if np.random.random() < 0.001: print num_offsets()
        if self.offset_t and self.t == self.dev_t and certainty > certainty_threshold and self.t < state.T:
            self.dev_t += 1
            self.this_num_offsets += 1

        a_ref = self.reference(state)
        a_pol = self.policy(state)
        if self.t == self.dev_t: # or (self.deviated and certainty < self.dev_certainty and np.random.random() < 0.5):
            self.deviated = True
            #self.dev_certainty = certainty
            a = None
            num_offsets[self.this_num_offsets] += 1
            if random.random() > self.epsilon: # exploit
                a = self.policy(state)
            else:
                self.dev_costs = self.policy.predict_costs(state)
                self.dev_actions = list(state.actions)[:]
                self.dev_a, self.dev_imp_weight = self.explore(self.dev_costs, self.dev_actions)
                a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.npvalue()[0,0]
            
        else:
            a = a_ref
            if not (self.rollin_ref() if not self.deviated else self.rollout_ref()):
                a = a_pol
                
        self.pred_act_cost.append(costs[a])
        return a

    def explore(self, costs, dev_actions):
        # returns action and importance weight
        if self.exploration == BanditLOLS.EXPLORE_UNIFORM:
            return random.choice(list(dev_actions)), len(dev_actions)
        if self.exploration in [BanditLOLS.EXPLORE_BOLTZMANN, BanditLOLS.EXPLORE_BOLTZMANN_BIASED]:
            if len(dev_actions) != self.policy.n_actions:
                disallow = np.zeros(self.policy.n_actions)
                for i in xrange(self.policy.n_actions):
                    if i not in dev_actions:
                        disallow[i] = 1e10
                costs += dy.inputTensor(disallow)
            probs = dy.softmax(- costs / self.temperature)
            a, p = macarico.util.sample_from_probs(probs)
            p = p.npvalue()[0]
            if self.exploration == BanditLOLS.EXPLORE_BOLTZMANN_BIASED:
                p = max(p, 1e-4)
            return a, 1 / p
        assert False, 'unknown exploration strategy'
    
        
    def update(self, loss):
        #self.pred_cost_without_dev = self.pred_cost_total - self.pred_cost_dev
        if self.use_prefix_costs:
            #loss -= self.pred_cost_until_dev
            #loss -= self.pred_cost_without_dev
            #loss -= sum(self.pred_act_cost)
            loss -= sum(self.pred_act_cost) - self.pred_act_cost[self.dev_t-1]
            #loss -= sum(self.pred_act_cost) - sum(self.pred_act_cost[self.dev_t-1:])
            
        if self.dev_a is not None:
            baseline = 0 if self.baseline is None else self.baseline()
            truth = self.build_cost_vector(baseline, loss, self.dev_a, self.dev_imp_weight, self.dev_costs)
            importance_weight = 1
            old_dev_actions = self.dev_actions[:]
            if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                self.dev_actions = [self.dev_a if isinstance(self.dev_a, int) else self.dev_a.npvalue()[0,0]]
                importance_weight = self.dev_imp_weight
            #print 'diff = %s, a = %s' % (self.dev_costs.npvalue() - truth, self.dev_actions)
            #print (self.dev_costs.npvalue() - truth)[0,self.dev_actions[0]]
            loss_var = self.policy.forward_partial_complete(self.dev_costs, truth, self.dev_actions)
            self.dev_actions = old_dev_actions # TODO remove?
            loss_var *= importance_weight
            loss_var.backward()
            
            a = self.dev_a if isinstance(self.dev_a, int) else self.dev_a.npvalue()[0,0]
            self.squared_loss = (loss - self.dev_costs.npvalue()[a]) ** 2
            
            if self.baseline is not None:
                self.baseline.update(loss)

    def build_cost_vector(self, baseline, loss, dev_a, imp_weight, dev_costs):
        costs = np.zeros(self.policy.n_actions)
        a = dev_a
        if not isinstance(a, int):
            a = a.npvalue()[0,0]
        if self.learning_method == BanditLOLS.LEARN_BIASED:
            costs -= baseline
            costs[a] = loss - baseline
        elif self.learning_method == BanditLOLS.LEARN_IPS:
            costs -= baseline
            costs[a] = (loss - baseline) * imp_weight
        elif self.learning_method == BanditLOLS.LEARN_DR:
            costs += dev_costs.npvalue() # now costs = \hat c
            costs[a] = dev_costs.npvalue()[a] + imp_weight * (loss - dev_costs.npvalue()[a])
        elif self.learning_method == BanditLOLS.LEARN_MTR:
            costs[a] = loss - baseline
        elif self.learning_method == BanditLOLS.LEARN_MTR_ADVANTAGE:
            costs[a] = loss - dev_costs.npvalue().min()
        else:
            assert False, self.learning_method
        return costs


class BanditLOLSMultiDev(BanditLOLS):
    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=BanditLOLS.LEARN_IPS,
                 exploration=BanditLOLS.EXPLORE_UNIFORM, epsilon=1.0,
                 mixture=BanditLOLS.MIX_PER_ROLL,
                 use_prefix_costs=False, temperature=1.):
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        self.exploration = exploration
        self.temperature = temperature
        assert self.learning_method in range(BanditLOLS._LEARN_MAX), \
            'unknown learning_method, must be one of BanditLOLS.LEARN_*'
        assert self.exploration in range(BanditLOLS._EXPLORE_MAX), \
            'unknown exploration, must be one of BanditLOLS.EXPLORE_*'
        
        self.use_prefix_costs = use_prefix_costs
        if mixture == BanditLOLS.MIX_PER_ROLL:
            use_in_ref  = p_rollin_ref()
            use_out_ref = p_rollout_ref()
            self.rollin_ref  = lambda: use_in_ref
            self.rollout_ref = lambda: use_out_ref
        else:
            self.rollin_ref  = p_rollin_ref
            self.rollout_ref = p_rollout_ref
        self.epsilon = epsilon
        self.t = None
        self.dev_t = []
        self.dev_a = []
        self.dev_actions = []
        self.dev_imp_weight = []
        self.dev_costs = []
        self.squared_loss = 0.
        self.pred_act_cost = []
        self.this_num_offsets = 0
        
        super(BanditLOLS, self).__init__()

    def __call__(self, state):
        global certainty_tracker, num_offsets
        if self.t is None:
            self.t = 0

        self.t += 1

        costs = self.policy.predict_costs(state).npvalue()
        costs_idx = costs.argsort()
        certainty = costs[costs_idx[1]] - costs[costs_idx[0]]
        certainty_tracker.update(certainty)
        #if np.random.random() < 0.001: print certainty_tracker()
        #certainty = costs.max() - costs.min()
        #certainty_threshold = 100.5
        certainty_threshold = certainty_tracker()
        #if np.random.random() < 0.001: print self.certainty_tracker()
        #if np.random.random() < 0.001: print num_offsets()

        a_ref = self.reference(state)
        a_pol = self.policy(state)
        
        if certainty < certainty_tracker:
            # deviate
            a = None
            if random.random() > self.epsilon: # exploit
                a = self.policy(state)
                dev_costs = None
                iw = 0.
            else:
                dev_costs = self.policy.predict_costs(state)
                dev_a, iw = self.explore(dev_costs, state.actions)
                a = dev_a if isinstance(dev_a, int) else dev_a.npvalue()[0,0]
                
            self.dev_t.append(self.t)
            self.dev_a.append(a)
            self.dev_actions.append(list(state.actions)[:])
            self.dev_imp_weight.append(iw)
            self.dev_costs.append(dev_costs)
            
        else:
            a = a_ref
            if not (self.rollin_ref() if not self.deviated else self.rollout_ref()):
                a = a_pol
                
        self.pred_act_cost.append(costs[a])
        return a
    
        
    def update(self, loss0):
        #self.pred_cost_without_dev = self.pred_cost_total - self.pred_cost_dev

        for dev_t, dev_a, dev_actions, dev_imp_weight, dev_costs in zip(self.dev_t, self.dev_a, self.dev_actions, self.dev_imp_weight, self.dev_costs):
            if dev_costs is None or dev_imp_weight == 0.:
                continue
            
            loss = loss0
            if self.use_prefix_costs:
                loss -= sum(self.pred_act_cost) - self.pred_act_cost[dev_t-1]
            
            truth = self.build_cost_vector(0, loss, dev_a, dev_imp_weight, dev_costs)
            importance_weight = 1
            if self.learning_method in [BanditLOLS.LEARN_MTR, BanditLOLS.LEARN_MTR_ADVANTAGE]:
                dev_actions = [dev_a if isinstance(dev_a, int) else dev_a.npvalue()[0,0]]
                importance_weight = dev_imp_weight
            loss_var = self.policy.forward_partial_complete(dev_costs, truth, dev_actions)
            loss_var *= importance_weight
            loss_var.backward()
            
            a = dev_a if isinstance(dev_a, int) else dev_a.npvalue()[0,0]
            self.squared_loss = (loss - dev_costs.npvalue()[a]) ** 2

    
class BanditLOLSRewind(BanditLOLS):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1
    LEARN_BIASED, LEARN_IPS, LEARN_DR, LEARN_MTR, LEARN_MTR_ADVANTAGE, _LEARN_MAX = 0, 1, 2, 3, 4, 5
    EXPLORE_UNIFORM, EXPLORE_BOLTZMANN, EXPLORE_BOLTZMANN_BIASED, _EXPLORE_MAX = 0, 1, 2, 3

    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=LEARN_IPS,
                 exploration=EXPLORE_UNIFORM, baseline=None,
                 epsilon=1.0, mixture=MIX_PER_ROLL, use_prefix_costs=False,
                 temperature=1., offset_t=False):
        super(BanditLOLSRewind, self).__init__(reference, policy, p_rollin_ref, p_rollout_ref, learning_method, exploration, baseline, epsilon, mixture, use_prefix_costs, temperature, False)
        self.cur_run = 0
        self.certainty = []
        self.backbone = []

    def __call__(self, state):
        if self.cur_run == 0:
            return self.call_backbone(state)
        else:
            return self.call_rollout(state)

    def call_backbone(self, state):
        if self.t is None:
            self.t = 0

        self.t += 1

        costs = self.policy.predict_costs(state).npvalue()
        costs_idx = costs.argsort()
        certainty = costs[costs_idx[1]] - costs[costs_idx[0]]
        self.certainty.append(certainty)
            
        a_ref = self.reference(state)
        a_pol = self.policy(state)
        a = a_ref if self.rollout_ref() else a_pol
        self.backbone.append(a)

        return a
        

    def call_rollout(self, state):
        if self.t is None:
            self.t = 0
            #self.dev_t = np.random.randint(0, state.T) + 1
            j = 0 # lowest certainty
            for i, v in enumerate(self.certainty):
                if v < self.certainty[j]:
                    j = i
            self.dev_t = j + 1
            self.pred_act_cost = []

        return super(BanditLOLSRewind, self).__call__(state)

    def run_again(self):
        self.cur_run += 1
        if self.cur_run > 1:
            return False
        # reset
        self.t = None
        return True


    
class EpisodeRunner(macarico.Learner):
    REF, LEARN, ACT = 0, 1, 2

    def __init__(self, policy, run_strategy, reference=None):
        self.policy = policy
        self.run_strategy = run_strategy
        self.reference = reference
        self.t = 0
        self.total_loss = 0.
        self.trajectory = []
        self.limited_actions = []
        self.costs = []
        self.ref_costs = []

    def __call__(self, state):
        a_type = self.run_strategy(self.t)
        pol = self.policy(state)
        ref_costs_t = np.zeros(self.policy.n_actions)
        self.reference.set_min_costs_to_go(state, ref_costs_t)
        self.ref_costs.append(ref_costs_t)
        if a_type == EpisodeRunner.REF:
            a = self.reference(state)
        elif a_type == EpisodeRunner.LEARN:
            a = pol
        elif isinstance(a_type, tuple) and a_type[0] == EpisodeRunner.ACT:
            a = a_type[1]
        else:
            raise ValueError('run_strategy yielded an invalid choice %s' % a_type)

        assert a in state.actions, \
            'EpisodeRunner strategy insisting on an illegal action :('

        self.limited_actions.append(state.actions[:])
        self.trajectory.append(a)
        cost = self.policy.predict_costs(state)
        self.costs.append(cost)
        self.t += 1

        return a
    
def one_step_deviation(T, rollin, rollout, dev_t, dev_a):
    def run(t):
        if   t == dev_t: return (EpisodeRunner.ACT, dev_a)
        elif t <  dev_t: return rollin(t)
        else:            return rollout(t)
    return run

class TiedRandomness(object):
    def __init__(self, rng=random.random):
        self.tied = {}
        self.rng = rng

    def reset(self):
        self.tied = {}

    def __call__(self, t):
        if t not in self.tied:
            self.tied[t] = self.rng()
        return self.tied[t]
    
def lols(ex, loss, ref, policy, p_rollin_ref, p_rollout_ref,
         mixture=BanditLOLS.MIX_PER_ROLL):
    # construct the environment
    env = ex.mk_env()
    # set up a helper function to run a single trajectory
    def run(run_strategy):
        env.rewind()
        runner = EpisodeRunner(policy, run_strategy, ref)
        env.run_episode(runner)
        cost = loss()(ex, env)
        return cost, runner.trajectory, runner.limited_actions, runner.costs

    n_actions = env.n_actions
    
    # construct rollin and rollout policies
    if mixture == BanditLOLS.MIX_PER_STATE:
        # initialize tied randomness for both rollin and rollout
        # TODO THIS IS BORKEN!
        rng = TiedRandomness()
        rollin_f  = lambda t: EpisodeRunner.REF if rng(t) <= p_rollin_ref  else EpisodeRunner.LEARN
        rollout_f = lambda t: EpisodeRunner.REF if rng(t) <= p_rollout_ref else EpisodeRunner.LEARN
    else:
        rollin  = EpisodeRunner.REF if p_rollin_ref()  else EpisodeRunner.LEARN
        rollout = EpisodeRunner.REF if p_rollout_ref() else EpisodeRunner.LEARN
        rollin_f  = lambda t: rollin
        rollout_f = lambda t: rollout

    # build a back-bone using rollin policy
    loss0, traj0, limit0, costs0 = run(rollin_f)
    T = env.T

    # start one-step deviations
    objective = 0. # Variable(torch.zeros(1))
    traj_rollin = lambda t: (EpisodeRunner.ACT, traj0[t])
    for t, costs_t in enumerate(costs0):
        costs = np.zeros(n_actions)
        # collect costs for all possible actions
        for a in limit0[t]:
            l, traj, _, _ = run(one_step_deviation(T, traj_rollin, rollout_f, t, a))
            costs[a] = l
        # accumulate update
        costs -= costs.min()
        objective += policy.forward_partial_complete(costs_t, costs, limit0[t])

    # run backprop
    v = objective.npvalue()
    objective.backward()

    return v, v

