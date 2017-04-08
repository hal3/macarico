from lts import LTS

class MaximumLikelihood(LTS):
    def __init__(self):
        super(MaximumLikelihood, self).__init__()

    def train(self, task, input):
        # remember the task and run it
        self.task = task
        task._run(input)

        # return the total loss from the reference policy
        return task.ref_policy.final_loss()

    def act(self, state, a_ref=None):
        # in maximum likelihood, past actions are always taken to be
        # "truth"; in this context, that means that they are
        # ref-optimal actions, given to us by a_ref.
        return a_ref
    
    
