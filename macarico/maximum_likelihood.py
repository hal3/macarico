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
        return task.ref_policy.loss()

    def act(self, state):
        # in maximum likelihood, past actions are always taken to be
        # "truth"; in this context, that means that they are
        # ref-optimal actions
        a_ref = task.ref_policy.next()
        
        # convert the list to an int if necessary (just pick the first
        # reference action)
        if isinstance(a_ref, list):
            a_ref = a_ref[0] if len(a_ref) > 0 else 0

        # tell the reference that that's the action we're taking
        task.ref_policy.step(a_ref)

        return a_ref
    
    
