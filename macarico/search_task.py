import torch.nn as nn

class SearchTask(nn.Module):
    def __init__(self, state_dim, n_actions, reference):
        # initialize the pytorch module
        super(SearchTask, self).__init__()

        # remember reference
        self.reference = reference

        # set up cost sensitive one-against-all
        self._lts_csoaa_predict = nn.Linear(state_dim, n_actions)

    def act(self, state):
        if not self.training:
            # we're in test mode, just act greedily
            return self._act_greedy(state)
        
        # otherwise, we're training, so we need to ask the lts_method
        # how we should act
        return self._lts_method.act(self, state)

    def _act_greedy(self, state):
        # predict costs using the csoaa model
        pred_costs = self._lts_csoaa_predict(state)
        # return the argmin cost
        return pred_costs.argmin()
    
    def forward(self, input, truth=None, lts_method=None):
        # if we're running in test mode, that's easy
        if truth is None or lts_method is None:
            self.training = False
            return self._run(input)

        # otherwise, we're training, which means that lts_method needs
        # to be in charge
        self.training = True

        # construct the reference policy
        self.ref_policy = self.reference(truth) if training else None

        # to act() we need the lts_method to tell us what to do
        self._lts_method = lts_method

        # start training
        self.last_loss = lts_method.train(self, input)
        
