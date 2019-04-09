import macarico
from macarico.util import LearnerToAlg
import torch.nn

class MultitaskLoss(macarico.Loss):
    def __init__(self, dispatchers, only_one=None):
        self.dispatchers = dispatchers
        self.only_one = only_one
        assert all((not d.loss().corpus_level for d in dispatchers)), \
            'sorry, MultitaskLoss currently cannot handle corpus level losses'
        super().__init__('mtl' if only_one is None else \
                         dispatchers[only_one].loss().name)

    def evaluate(self, example):
        if self.only_one is None:
            for d in self.dispatchers:
                if d.check(example):
                    return d.loss().evaluate(example)
            raise Exception('MultitaskLoss dispatching failed on %s' % example)
            #return None
        else:
            d = self.dispatchers[self.only_one]
            if d.check(example):
                return d.loss().evaluate(example)
            else:
                #raise Exception('MultitaskLoss dispatching failed on %s' % example)
                return None

class Dispatcher(object):
    def __init__(self, check, mk_env, loss, policy, learn):
        assert isinstance(loss(), macarico.Loss)
        assert isinstance(policy, macarico.Policy)
        #assert isinstance(learn(), macarico.Learner)
        self.mk_env = mk_env
        self.check = check
        self.loss = loss
        self.policy = policy
        self.learn = LearnerToAlg(learn(), policy, loss) \
                     if isinstance(learn(), macarico.Learner) \
                     else learn

class MultitaskPolicy(macarico.Policy):
    def __init__(self, dispatchers):
        self.dispatchers = dispatchers
        # register policy parameters so that things like .parameters() and .modules() and ._reset_some() work
        super(MultitaskPolicy, self).__init__()
        all_modules = []
        for d in dispatchers:
            all_modules += d.policy.modules()
        self._dispatcher_module_list = torch.nn.ModuleList(all_modules)

    def forward(self, state):
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
        
class MultitaskLearningAlg(macarico.LearningAlg):
    def __init__(self, dispatchers):
        self.dispatchers = dispatchers

    def mk_env(self, example):
        for d in self.dispatchers:
            if d.check(example):
                return d.mk_env(example)
        raise Exception('MultitaskLearningAlg.mk_env dispatching failed on %s' % example)

    def policy(self):
        return MultitaskPolicy(self.dispatchers)

    def losses(self):
        return [lambda: MultitaskLoss(self.dispatchers)] + \
               [(lambda i: lambda: MultitaskLoss(self.dispatchers, i))(ii) #ugh stupid python scoping
                for ii in range(len(self.dispatchers))]

    def __call__(self, env):
        for d in self.dispatchers:
            if d.check(env.example):
                return d.learn(env)
        raise Exception('MultitaskLearningAlg.__call__ dispatching failed on %s' % ex)
        #return 0
