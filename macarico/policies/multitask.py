import macarico
from macarico.util import learner_to_alg

class MultitaskLoss(macarico.Loss):
    def __init__(self, dispatchers, only_one=None):
        self.dispatchers = dispatchers
        self.only_one = only_one
        assert all((not d.loss.corpus_level for d in dispatchers)), \
            'sorry, MultitaskLoss currently cannot handle corpus level losses'
        super(MultitaskLoss, self).__init__('mtl' if only_one is None else \
                                            dispatchers[only_one].loss.name)

    def evaluate(self, truth, state):
        if self.only_one is None:
            for d in self.dispatchers:
                if d.check(state.example):
                    return d.loss.evaluate(truth, state)
            raise Exception('MultitaskLoss dispatching failed on %s' % state)
            #return None
        else:
            d = self.dispatchers[self.only_one]
            if d.check(state.example):
                return d.loss.evaluate(truth, state)
            else:
                #raise Exception('MultitaskLoss dispatching failed on %s' % state)
                return None

class Dispatcher(object):
    def __init__(self, check, loss, policy, learn):
        self.check = check
        self.loss = loss
        self.policy = policy
        self.learn = learner_to_alg(learn, loss) \
                     if isinstance(learn(), macarico.Learner) \
                     else learn

class MultitaskPolicy(macarico.Policy):
    def __init__(self, dispatchers):
        self.dispatchers = dispatchers
        super(MultitaskPolicy, self).__init__()

    def __call__(self, state):
        for d in self.dispatchers:
            if d.check(state.example):
                return d.policy(state)
        raise Exception('MultitaskLearningAlg dispatching failed on %s' % state.example)
        #return random.choice(list(state.actions))
        
    def state_dict(self):
        return [d.policy.state_dict() for d in self.dispatchers]

    def load_state_dict(self, models):
        assert False
        #self.dispatchers[0][2].load_state_dict(models[0])
        #for (_, _, policy, _), model in zip(self.dispatchers, models):
        #    policy.load_state_dict(model)
        
class MultitaskLearningAlg(object):
    def __init__(self, dispatchers):
        self.dispatchers = dispatchers
        
    def policy(self):
        return MultitaskPolicy(self.dispatchers)

    def losses(self):
        return [MultitaskLoss(self.dispatchers)] + \
               [MultitaskLoss(self.dispatchers, i)
                for i in xrange(len(self.dispatchers))]
    
    def __call__(self, ex):
        for d in self.dispatchers:
            if d.check(ex):
                return d.learn(ex)
        raise Exception('MultitaskLearningAlg dispatching failed on %s' % ex)
        #return 0
