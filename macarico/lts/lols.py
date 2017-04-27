from __future__ import division

import random
import torch
import macarico

class BanditLOLS(macarico.LearningAlg):
    MIX_PER_STATE, MIX_PER_ROLL = 0, 1
    LEARN_REINFORCE, LEARN_IMPORTANCE = 0, 1

    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref,
                 learning_method=LEARN_REINFORCE, baseline=None,
                 epsilon=1.0, mixture=MIX_PER_ROLL):
        self.reference = reference
        self.policy = policy
        self.learning_method = learning_method
        if mixture == BanditLOLS.MIX_PER_STATE:
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
        self.dev_weight = None
        self.dev_state = None
        self.dev_limit_actions = None
        super(BanditLOLS, self).__init__()

    def __call__(self, state, limit_actions=None):
        if self.t is None:
            self.t = 0
            self.dev_t = random.randint(1, state.T)

        self.t += 1
        if self.t == self.dev_t:
            if random.random() > self.epsilon: # exploit
                return self.policy(state, limit_actions)
            elif self.learning_method == BanditLOLS.LEARN_REINFORCE:
                self.dev_state = self.policy.stochastic(state, limit_actions)
                self.dev_a = self.dev_state.data[0, 0]
                return self.dev_a
            elif self.learning_method == BanditLOLS.LEARN_IMPORTANCE:
                if limit_actions is None:
                    self.dev_a = random.randint(0, self.policy.n_actions-1)
                    self.dev_weight = self.policy.n_actions
                else:
                    self.dev_a = random.choice(limit_actions)
                    self.dev_weight = len(limit_actions)
                self.dev_state = self.policy.forward_partial(state)
                self.dev_limit_actions = limit_actions
                return self.dev_a
        elif self.rollin_ref() if self.t < self.dev_t else self.rollout_ref():
            self.policy(state, limit_actions) # must call this to get updates
            return self.reference(state, limit_actions=limit_actions)
        else:
            return self.policy(state, limit_actions=limit_actions)

    def update(self, loss):
        if self.dev_a is not None:
            baseline = 0. if self.baseline is None else self.baseline()
            if self.learning_method == BanditLOLS.LEARN_REINFORCE:
                self.dev_state.reinforce(baseline - loss)
                torch.autograd.backward(self.dev_state, [None])
            elif self.learning_method == BanditLOLS.LEARN_IMPORTANCE:
                truth = self.build_cost_vector(baseline, loss)
                self.policy.forward_partial_complete(self.dev_state, truth).backward()
        if self.baseline is not None:
            self.baseline.update(loss)

    def build_cost_vector(self, baseline, loss):
        # TODO: handle self.dev_limit_actions
        # TODO: doubly robust
        costs = torch.zeros(self.policy.n_actions) + baseline
        costs[self.dev_a] = loss * self.dev_weight
        return costs
