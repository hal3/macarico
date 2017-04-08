import lts

class MaximumLikelihood(lts.LTS):
    def __init__(self):
        super(MaximumLikelihood, self).__init__()

    def train(self, task, input):
        # remember the task and run it
        self.task = task
        self.task._run(input)

        # return the total loss from the reference policy
        return self.task.ref_policy.final_loss()

    def act(self, state, a_ref=None):
        # in maximum likelihood, past actions are always taken to be
        # "truth"; in this context, that means that they are
        # ref-optimal actions, given to us by a_ref.
        #
        # however, if we are _training_, we need to accumulate loss
        # for making incorrect predictions
        if self.task.training:
            if a_ref is None:
                self._warn('act() called in training mode with a_ref=None; ignoring')
            else:
                # increment the objective to include to loss for this
                # prediction
                self.objective += self.task.lts_objective(state, a_ref)
                
        return a_ref
        
    
