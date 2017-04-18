from __future__ import division

from lts import LTS

class Reinforce(LTS):
    def __init__(self):
        super(Reinforce, self).__init__()

    def train(self, task, input):
        # remember the task and run it
        self.task = task
        task._run(input)

        # get the total loss
        loss = task.ref_policy.final_loss()

        # update gradients
        TODO

        # done
        return loss

    def act(self, state, a_ref=None):
        # sample an action according to softmin on costs
        return self.act_sample(state)
