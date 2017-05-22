import random
import numpy as np
# helpful functions

def break_ties_by_policy(reference, policy, state, force_advance_policy=True):
    costs = np.zeros(state.n_actions)
    try:
        reference.set_min_costs_to_go(state, costs)
    except NotImplementedError:
        ref = reference(state)
        if force_advance_policy:
            policy(state)
        return ref
    # otherwise we successfully got costs
    old_actions = state.actions
    min_cost = min((costs[a] for a in old_actions))
    state.actions = [a for a in old_actions if costs[a] <= min_cost]
    a = policy(state)  # advances policy
    assert a is not None, 'got action None in %s, costs=%s, old_actions=%s' % (state.actions, costs, old_actions)
    state.actions = old_actions
    return a
