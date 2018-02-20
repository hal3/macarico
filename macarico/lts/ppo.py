import macarico
import dynet as dy
import numpy as np
import copy

class PPO(macarico.Learner):
    "Proximal Policy Optimization"
    def __init__(self, policy, baseline, epsilon, temperature=1.0, only_one_deviation=False):
        super(PPO, self).__init__()
        self.policy = policy
        self.baseline = baseline
        self.epsilon = epsilon
        self.temperature = temperature
        self.trajectory = []
        self.only_one_deviation = only_one_deviation

    def update(self, loss):
        if len(self.trajectory) > 0:
            b = self.baseline()
            total_loss = 0
            dev_t = np.random.choice(range(len(self.trajectory)))
            for t, (a, p_a, s) in enumerate(self.trajectory):
                if self.only_one_deviation and t != dev_t:
                    continue
                p_a_value = p_a.npvalue()
                ratio = p_a * (1/p_a_value[0])
                ratio_by_adv = (b - loss) * ratio
                lower_bound = dy.constant(1, 1 - self.epsilon)
                clipped_ratio = dy.bmax(ratio, lower_bound)
                upper_bound = dy.constant(1, 1 + self.epsilon)
                clipped_ratio = dy.bmin(clipped_ratio, upper_bound)
                clipped_ratio_by_adv = (b - loss) * clipped_ratio
                increment = dy.bmin(ratio_by_adv, clipped_ratio_by_adv)
                total_loss -= increment
            self.baseline.update(loss)
            if not isinstance(total_loss, float):
                total_loss.forward()
                total_loss.backward()

    def update_ppo(self, loss):
        if len(self.trajectory) > 0:
            b = self.baseline()
            total_loss = 0
            dev_t = np.random.choice(range(len(self.trajectory)))
            for t, (a, p_a_old, s) in enumerate(self.trajectory):
                if self.only_one_deviation and t != dev_t:
                    continue
                p_a_value = p_a_old.npvalue()
                # Compute p_a, the new probablity
                p_a = self.policy.stochastic_probability(
                    s, temperature=self.temperature)[a]
#                print('p_a_value: ', p_a_value, ' new: ', p_a.npvalue())
                ratio = p_a * (1/p_a_value[0])
#                print('ratio: ', ratio.npvalue()[0])
                ratio_by_adv = (b - loss) * ratio
                lower_bound = dy.constant(1, 1 - self.epsilon)
                clipped_ratio = dy.bmax(ratio, lower_bound)
                upper_bound = dy.constant(1, 1 + self.epsilon)
                clipped_ratio = dy.bmin(clipped_ratio, upper_bound)
                clipped_ratio_by_adv = (b - loss) * clipped_ratio
                increment = dy.bmin(ratio_by_adv, clipped_ratio_by_adv)
                total_loss -=  increment
            self.baseline.update(loss)
            if not isinstance(total_loss, float):
                total_loss.forward()
                total_loss.backward()

    def __call__(self, state):
        state_copy = copy.deepcopy(state)
        action, p_action = self.policy.stochastic_with_probability(state, temperature=self.temperature)
        self.trajectory.append((action, p_action, state_copy))
        return action
