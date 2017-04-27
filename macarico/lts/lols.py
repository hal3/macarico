from __future__ import division

import macarico

class BanditLOLS(macarico.LearningAlg):
    MIX_PER_STATE, MIX_PER_ROLL = 0,1
    
    def __init__(self, reference, policy, p_rollin_ref, p_rollout_ref, baseline, epsilon=0.5, mixture=MIX_PER_ROLL):
        self.reference = reference
        self.policy = policy
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

    def __call__(self, state, limit_actions=None):
        if self.T is None:
            self.t = 0
            self.dev_t = random.randint(1, state.T)

        self.t += 1
        
        if self.t == self.dev_t:
            if random.random() < self.epsilon:
                return self.policy(state, limit_actions)
            else: # exploring
                self.dev_a = self.policy.stochastic(state, limit_actions)
                return self.dev_a
        elif self.rollin_ref() if self.t < self.dev_t else self.rollout_ref():
            return self.reference(state, limit_actions=limit_actions)
        else:
            return self.policy(state, limit_actions=limit_actions)
            
    def update(self, loss):
        b = self.baseline()
        if self.dev_a is not None:
            self.dev_a.reinforce(b - loss)
            torch.autograd.backward(self.dev_a, [None])
        self.baseline.update(loss)

