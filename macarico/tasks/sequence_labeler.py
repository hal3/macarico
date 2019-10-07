import macarico


class SequenceLabeler(macarico.Env):
    """Basic sequence labeling environment (input and output sequences have the same
    length). Loss is evaluated with Hamming distance, which has an optimal
    reference policy.
    """

    def __init__(self, example):
        self.N = example.N
        macarico.Env.__init__(self, example.n_labels, self.N, example)
        self.X = example.X
        # TODO default this
        self.actions = set(range(example.n_labels))

    def _run_episode(self, policy):
#        print('self._losses: ', self._losses)
        a_string = ''
        total_reward = 0
        for self.n in range(self.horizon()):
            a = policy(self)
            a_string += str(a) + '_'
            self._losses.append(0)
            if a == self.example.Y[self.n]:
                total_reward += 1
            else:
                total_reward += 0
#            self._losses.append(1 if a == self.example.Y[self.n] else 0)
#        print('self._losses: ', self._losses)
#        print('example: ', self.example)
#        print('a_string: ', a_string)
        self._losses[-1] = total_reward
        return self._trajectory

    def _rewind(self):
        # Erase previous predictions on example, if any
        self.example.Yhat = None


class HammingLossReference(macarico.Reference):
    def __call__(self, state):
        return int(state.example.Y[state.n])

    def set_min_costs_to_go(self, state, cost_vector):
        cost_vector *= 0
        cost_vector += 1
        cost_vector[state.example.Y[state.n]] = 0.


class HammingLoss(macarico.Loss):
    def __init__(self, Yname=None, Yhatname=None):
        self.Yname = Yname
        self.Yhatname = Yhatname
        super(HammingLoss, self).__init__('hamming')

    def evaluate(self, example):
        Y = getattr(example, self.Yname or 'Y')
        Yhat = getattr(example, self.Yhatname or 'Yhat')
        assert len(Y) == len(Yhat)
        return sum(y != p for p,y in zip(Y, Yhat))
