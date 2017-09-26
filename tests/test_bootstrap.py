import dynet as dy
from macarico.policies.bootstrap import build_policy_bag
from macarico.policies.bootstrap import BootstrapPolicy

if __name__ == '__main__':
    num_actions = 10
    bag_size = 16
    policy_bag = []
    state = None
    dy_model = dy.ParameterCollection()
    features_bag = [None for i in range(bag_size)]
    loss_fn = 'squared'
    print(build_policy_bag(dy_model, features_bag, num_actions, loss_fn))
    bootstrap_policy = BootstrapPolicy(dy_model, features_bag, num_actions, loss_fn)
#    print(bootstrap_probabilities(num_actions, bag_size, policy_bag, state))
