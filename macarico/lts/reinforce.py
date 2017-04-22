from __future__ import division

import torch
#import torch.nn.functional as F
#from torch import autograd
#from torch.autograd import Variable

import macarico


class Reinforce(macarico.LearningAlg):
    "REINFORCE with a scalar baseline function."

    def __init__(self, policy, baseline):
        self.trajectory = []
        self.baseline = baseline
        self.policy = policy
        super(Reinforce, self).__init__()

    def update(self, loss):
        b = self.baseline()
        for a in self.trajectory:
            a.reinforce(b - loss)
        self.baseline.update(loss)
        torch.autograd.backward(self.trajectory[:], [None]*len(self.trajectory))

    def __call__(self, state):
        action = self.policy.stochastic(state)
        # log actions (and values for actor critic) taken along current trajectory
        self.trajectory.append(action)
        return action.data[0,0]   # return an integer



#class AdvantageActorCritic(LTS):
#
#    def __init__(self):
#        self.out_nodes = []
#        self.gradients = []
#        self.saved_actions = []
#        self.saved_values = []
#        self.saved_rewards = []
#        self.task = None
#        self.gamma = 0.9
#        super(AdvantageActorCritic, self).__init__()
#
#    def train(self, task, input):
#        # remember the task and run it
#        self.task = task
#        task._run(input)
#
#        # get the total loss
#        loss = task.ref_policy.final_loss()
#
#        # TODO: eventually we should support partial loss feedback for reduced variance.
#        self.saved_rewards = [-loss for _ in self.saved_values]
#
#        # Compute discounted rewards
#        rewards = []
#        R = 0.0
#        for r in self.saved_rewards[::-1]:    # reverse
#            R = r + self.gamma * R
#            rewards.append(R)
#        rewards = torch.Tensor(rewards[::-1]) # un-reverse
#        rewards = (rewards - rewards.mean()) / rewards.std()  # step is a little weird. got it from example in torch repo.
#
#        value_loss = 0   # for training value function regression
#        for a, v, r in zip(self.saved_actions, self.saved_values, rewards):
#            [vv] = v.data.squeeze()
#            a.reinforce(r - vv)
#            value_loss += F.smooth_l1_loss(v, Variable(torch.Tensor([r])))
#
#        self.out_nodes = [value_loss] + self.saved_actions
#        self.gradients = [torch.ones(1)] + [None] * len(self.saved_actions)
#
#        return loss
#
#    def zero_objective(self):
#        del self.out_nodes[:]
#        del self.gradients[:]
#        del self.saved_actions[:]
#        del self.saved_values[:]
#        del self.saved_rewards[:]
#
#    def get_objective(self):
#        return (self.out_nodes, self.gradients)
#
#    def act(self, state, a_ref=None):
#        pred_costs = self.task._lts_csoaa_predict(state)
#        probs = F.softmax(-pred_costs)
#        action = probs.multinomial()
#
#        # TODO: create a reasonable estimator of the value function
#        #pred_value = Variable(torch.Tensor([0.0]))
#        pred_value = probs.dot(pred_costs)
#
#        # log actions and values taken along current trajectory
#        self.saved_actions.append(action)
#        self.saved_values.append(pred_value)
#
#        return action.data[0,0]
