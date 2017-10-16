import macarico
import dynet as dy


class PPO(macarico.Learner):
    "Proximal Policy Optimization"
    def __init__(self, policy, baseline, epsilon, temperature=1.0):
        super(PPO, self).__init__()
        self.policy = policy
        self.baseline = baseline
        self.epsilon = epsilon
        self.temperature = temperature
        self.trajectory = []

    def update(self, loss):
        if len(self.trajectory) > 0:
            b = self.baseline()
            total_loss = 0
            for a, p_a in self.trajectory:
                p_a_value = p_a.npvalue()
                ratio = p_a * (1/p_a_value[0])
                ratio_by_adv = (b - loss) * ratio
                lower_bound = dy.constant(1, 1 - self.epsilon)
                clipped_ratio = dy.bmax(ratio, lower_bound)
                upper_bound = dy.constant(1, 1 + self.epsilon)
                clipped_ratio = dy.bmin(clipped_ratio, upper_bound)
                clipped_ratio_by_adv = (b - loss) * clipped_ratio
                increment = dy.bmin(ratio_by_adv, clipped_ratio_by_adv)
                total_loss -=  increment
            self.baseline.update(loss)
            total_loss.forward()
            total_loss.backward()

    def __call__(self, state):
        # TODO tune temp
        action, p_action = self.policy.stochastic_with_probability(state, temperature=self.temperature)
        self.trajectory.append((action, p_action))
        return action
