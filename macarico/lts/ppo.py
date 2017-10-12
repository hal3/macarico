import macarico
import dynet as dy


class PPO(macarico.Learner):
    "Proximal Policy Optimization"
    # TODO use a baseline
    def __init__(self, policy, baseline, epsilon):
        super(PPO, self).__init__()
        self.policy = policy
        self.baseline = baseline
        self.epsilon = epsilon
        self.trajectory = []

    def update(self, loss):
        if len(self.trajectory) > 0:
            b = self.baseline()
            total_loss = 0
            for a, p_a in self.trajectory:
                total_loss += (loss - b) * dy.log(p_a)
            self.baseline.update(loss)
            total_loss.forward()
            total_loss.backward()

    def __call__(self, state):
        # TODO tune temp
        temp = 1
        action, p_action = self.policy.stochastic_with_probability(state, temperature=temp)
        self.trajectory.append((action, p_action))
        return action
