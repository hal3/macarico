class MaximumLikelihood(macarico.LTS):
    def __init__(self):
        super(MaximumLikelihood, self).__init__()

    def train(self, task, input):
        self.task = task
        
        # reset the reference policy before running
        task.ref_policy.reset()
        
        # remember the task and run it
        task._run(input)

        # return the total loss
        return task.ref_policy.final_loss()

    def act(self, state, a_ref=None):
        # in maximum likelihood, past actions are always taken to be
        # "truth"; in this context, that means that they are
        # ref-optimal actions, given to us by a_ref.
        #
        # all we have to do is execute that action.
        return self._execute_action(a_ref)
    
    
