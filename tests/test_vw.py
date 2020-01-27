import argparse
import sys
from functools import partial

from tensorboardX import SummaryWriter

import macarico.data.synthetic as synth
import macarico.tasks.cartpole as cartpole
import macarico.tasks.dependency_parser as dep
import macarico.tasks.gridworld as gridworld
import macarico.tasks.hexgame as hexgame
import macarico.tasks.mdp as mdp
import macarico.tasks.mountain_car as car
import macarico.tasks.pendulum as pendulum
import macarico.tasks.seq2json as s2j
import macarico.tasks.seq2seq as s2s
import macarico.tasks.sequence_labeler as sl
from macarico.actors.bow import BOWActor
from macarico.actors.timed_bow import TimedBowActor
from macarico.actors.rnn import RNNActor
from macarico.annealing import ExponentialAnnealing, stochastic, NoAnnealing
from macarico.data.types import Dependencies
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt, FrontBackAttention, \
    SoftmaxAttention, AverageAttention
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.monte_carlo import MonteCarlo
from macarico.lts.reinforce import Reinforce, LinearValueFn, A2C
from macarico.lts.reslope import Reslope
from macarico.lts.bellman import Bellman
from macarico.lts.vd_reslope import VdReslope
from macarico.lts.vw_prep import VwPrep
from macarico.lts.vw_prep_policy_gradient import VwPrepPolicyGradient
from macarico.policies.linear import *
from macarico.policies.bootstrap import BootstrapPolicy
from macarico.policies.regressor import Regressor


def debug_on_assertion(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type != AssertionError:
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, ipdb
        traceback.print_exception(type, value, tb)
        print()
        ipdb.pm()


sys.excepthook = debug_on_assertion


def build_CB_learner(features, n_actions, alr=0.5, vdlr=0.5, clr=0.5, exp_type='eps', exp_param=0.3, is_timed_bow=False,
                  act_window=0, obs_window=0, learner_type='PREP'):
    if is_timed_bow:
        actor = TimedBowActor(features, n_actions, horizon, act_history_length=act_window,
                              obs_history_length=obs_window)
    else:
        actor = BOWActor(attention, n_actions, act_history_length=act_window, obs_history_length=obs_window)
    # Build the policy
    policy = lambda: VWPolicy(actor, n_actions, lr=alr, exp_type=exp_type, exp_param=exp_param)
    vd_regressor = pyvw.vw('-l ' + str(vdlr), quiet=True)
    ref_critic = pyvw.vw('-l ' + str(clr), quiet=True)
    learner = VwPrep(policy, actor, vd_regressor, ref_critic, learner_type)
    parameters = []
    return policy, learner, parameters


def test_sp(environment_name, n_epochs=1, n_examples=4, fixed=False, gpu_id=None, alr=0.2, vdlr=0.5,
            clr=0.5, eps=0.2, learner_type='PREP'):
    print(environment_name)
    is_timed_bow = False
    action_history = 0
    obs_history = 0
    if environment_name == 'sl':
        n_types = 50 if fixed else 10
        n_labels = 9 if fixed else 3
        horizon = 4
        features = BOWFeatures(n_types)
        attention = [AttendAt(features)]
        data = synth.make_sequence_mod_data(n_examples, horizon, n_types, n_labels)
        mk_env = sl.SequenceLabeler
        loss_fn = sl.HammingLoss
        action_history = 2
        obs_history = 3
        ref = sl.HammingLossReference()
        require_attention = None
        n_actions = mk_env(data[0]).n_actions
    elif environment_name == 'dep':
        data = [Dependencies(tokens=[0, 1, 2, 3, 4], heads=[1, 5, 4, 4, 1], token_vocab=5) for _ in range(n_examples)]
        mk_env = dep.DependencyParser
        loss_fn = dep.AttachmentLoss
        # TODO Add feature computation and attention
        ref = dep.AttachmentLossReference()
        require_attention = dep.DependencyAttention
        n_actions = mk_env(data[0]).n_actions
    elif environment_name == 's2s':
        data = synth.make_sequence_mod_data(n_examples, horizon, n_types, n_labels, include_eos=True)
        mk_env = s2s.Seq2Seq
        loss_fn = s2s.EditDistance
        ref = s2s.NgramFollower()
        # TODO Add feature computation and attention
        # Softmax Attention
        require_attention = AttendAt
        n_actions = mk_env(data[0]).n_actions
    elif environment_name == 's2j':
        data = synth.make_json_mod_data(n_examples, horizon, n_types, n_labels)
        mk_env = lambda ex: s2j.Seq2JSON(ex, n_labels, n_labels)
        loss_fn = s2j.TreeEditDistance
        ref = s2j.JSONTreeFollower()
        # TODO Add feature computation and attention
        require_attention = FrontBackAttention
        n_actions = mk_env(data[0]).n_actions
    else:
        # RL
        # TODO maybe convert this to a function
        tasks = {
            'pocman': (pocman.MicroPOCMAN, pocman.LocalPOCFeatures, pocman.POCLoss, pocman.POCReference),
            'cartpole': (cartpole.CartPoleEnv, cartpole.CartPoleFeatures, cartpole.CartPoleLoss, None),
            'blackjack': (blackjack.Blackjack, blackjack.BlackjackFeatures, blackjack.BlackjackLoss, None),
            'hex': (hexgame.Hex, hexgame.HexFeatures, hexgame.HexLoss, None),
            'gridworld': (gridworld.make_default_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
            'pendulum': (pendulum.Pendulum, pendulum.PendulumFeatures, pendulum.PendulumLoss, None),
            'car': (car.MountainCar, car.MountainCarFeatures, car.MountainCarLoss, None),
            'mdp': (
                lambda: synth.make_ross_mdp()[0], lambda: mdp.MDPFeatures(3), mdp.MDPLoss,
                lambda: synth.make_ross_mdp()[1]),
        }
        rl_mk_env, mk_fts, loss_fn, ref = tasks[environment_name]
        env = rl_mk_env()
        if 'gridworld' in environment_name:
            features = mk_fts(4, 4)
        else:
            features = mk_fts()
        if 'gridworld' in environment_name or 'cartpole' in environment_name:
            is_timed_bow = True
        n_actions = env.n_actions
        data = [rl_mk_env() for _ in range(2 ** 15)]
        attention = [AttendAt(features, lambda _: 0)]

        def train_loop_mk_env(example):
            return rl_mk_env()

        mk_env = train_loop_mk_env

    while True:
        policy, learner, parameters = build_CB_learner(attention, n_actions, alr, vdlr, clr, exp_type, exp_param,
                                                       is_timed_bow, act_window=action_history, obs_window=obs_history,
                                                       learner_type=learner_type)
        if fixed or not (environment_name in ['s2s', 's2j'] and (
                isinstance(learner, AggreVaTe) or isinstance(learner, Coaching))):
            break

    print(learner)

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        policy = policy.cuda()
        # TODO do we need to .cuda() everything else? maybe this should go in trainloop so it's isolated?
        # TODO call .cpu() on everything when we need it on the cpu
        # TODO replace:
        #   torch.zeros(...) -> self._new(...).zero_()
        #   torch.LongTensor(...) -> self._new(...).long()
        #   onehot -> onehot(new)
    optimizer = torch.optim.Adam(parameters, lr=0.0001)
    train_data = data[:-16]
    dev_data = data[-16:]
    util.TrainLoop(mk_env, policy, learner, optimizer,
                   print_freq=1.5,
                   losses=[loss_fn, loss_fn, loss_fn],
                   progress_bar=False,
                   minibatch_size=np.random.choice([1]), ).train(train_data, dev_data=dev_data, n_epochs=n_epochs)

def run_test(env, alr, vdlr, clr, clip, exp, exp_param, learner_type):
    # TODO can we run on GPU?
    gpu_id = None
    seed = 90210
    print('seed', seed)
    util.reseed(seed, gpu_id=gpu_id)
    test_sp(environment_name=env, n_epochs=1, n_examples=2*2*2*2*2**12, fixed=True, gpu_id=gpu_id,
            alr=alr, vdlr=vdlr, clr=clr, eps=exp_param, learner_type=learner_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['vd_reslope', 'reslope', 'vw-prep', 'reinforce',
                                                       'vw-prep-policy-gradient'],
                        default='vw-prep')
    parser.add_argument('--env', type=str, choices=[
        'gridworld', 'gridworld_stoch', 'gridworld_ep', 'cartpole', 'hex', 'blackjack', 'sl', 'dep'],
                        help='Environment to run on', default='gridworld')
    parser.add_argument('--alr', type=float, help='Actor learning rate', default=0.0005)
    parser.add_argument('--vdlr', type=float, help='Value difference learning rate', default=0.005)
    parser.add_argument('--clr', type=float, help='Critic learning rate', default=0.005)
    parser.add_argument('--clip', type=float, help='Gradient clipping argument', default=10)
    parser.add_argument('--exp', type=str, help='Exploration method', default='bootstrap',
                        choices=['eps-greedy', 'boltzmann', 'bootstrap'])
    parser.add_argument('--exp_param', type=float, help='Parameter for exp. method', default=0.4)
    args = parser.parse_args()
    # TODO support different methods
    run_test(env=args.env, alr=args.alr, vdlr=args.vdlr, clr=args.clr, clip=args.clip, exp=args.exp,
             exp_param=args.exp_param, learner_type=args.method)
