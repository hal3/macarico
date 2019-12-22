import argparse
import sys, os
import pathlib
from functools import partial

from tensorboardX import SummaryWriter

import macarico.data.synthetic as synth
import macarico.tasks.blackjack as blackjack
import macarico.tasks.cartpole as cartpole
import macarico.tasks.gridworld as gridworld
import macarico.tasks.hexgame as hexgame
import macarico.tasks.mdp as mdp
import macarico.tasks.mountain_car as car
import macarico.tasks.pendulum as pendulum
import macarico.tasks.pocman as pocman
from macarico.actors.bow import BOWActor
from macarico.actors.timed_bow import TimedBowActor
from macarico.annealing import ExponentialAnnealing, stochastic, NoAnnealing
from macarico.data.types import Dependencies
from macarico.features.sequence import EmbeddingFeatures, BOWFeatures, RNN, DilatedCNN, AttendAt
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.monte_carlo import MonteCarlo
from macarico.lts.MC_critic import MonteCarloC
from macarico.lts.reslope import Reslope
from macarico.lts.bellman import Bellman
from macarico.lts.vd_reslope import VdReslope
from macarico.lts.vw_prep import VwPrep
from macarico.policies.linear import *
from macarico.policies.bootstrap import BootstrapPolicy
from macarico.policies.regressor import Regressor


def test_vd_rl(environment_name, exp, exp_par, n_epochs=10000, plr=0.001, vdlr=0.001, clr=0.001, grad_clip=1,
               run_id=0, save_log=False):
    print('rl', environment_name)
    tasks = {
        'pocman': (pocman.MicroPOCMAN, pocman.LocalPOCFeatures, pocman.POCLoss, pocman.POCReference),
        'cartpole': (cartpole.CartPoleEnv, cartpole.CartPoleFeatures, cartpole.CartPoleLoss, None),
        'blackjack': (blackjack.Blackjack, blackjack.BlackjackFeatures, blackjack.BlackjackLoss, None),
        'hex': (hexgame.Hex, hexgame.HexFeatures, hexgame.HexLoss, None),
        'gridworld': (gridworld.make_default_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
        'gridworld_stoch': (
        gridworld.make_stochastic_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
        'gridworld_ep': (gridworld.make_episodic_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
        'pendulum': (pendulum.Pendulum, pendulum.PendulumFeatures, pendulum.PendulumLoss, None),
        'car': (car.MountainCar, car.MountainCarFeatures, car.MountainCarLoss, None),
        'mdp': (
        lambda: synth.make_ross_mdp()[0], lambda: mdp.MDPFeatures(3), mdp.MDPLoss, lambda: synth.make_ross_mdp()[1]),
    }
    logs = []
    logs.append('Epoch\tloss_val\tAvg_loss')

    mk_env, mk_fts, loss_fn, ref = tasks[environment_name]
    env = mk_env()
    # Compute features
    if 'gridworld' in environment_name:
        features = mk_fts(4, 4)
    else:
        features = mk_fts()
    # Compute some attention
    attention = [AttendAt(features, position=lambda _: 0)]
    # Build an actor
    actor = TimedBowActor(attention, env.n_actions, env.horizon(), act_history_length=0, obs_history_length=0)
    # actor = BOWActor(attention, env.n_actions, act_history_length=0, obs_history_length=0)
    # Build the policy
    # policy_fn = lambda: CSOAAPolicy(actor, env.n_actions)
    policy_fn = lambda: VWPolicy(actor, env.n_actions)
    policy = policy_fn()
    exploration = BanditLOLS.EXPLORE_UNIFORM
    explore = exp_par
    expb = 0.0
    learner = VwPrep(reference=None, policy=policy, exploration=exploration, explore=explore, save_log=save_log,
                     actor=actor, attach_time=False, expb=expb)
    print(learner)

    losses, objs = [], []
    for epoch in range(1, 1 + n_epochs):
        env = mk_env()
        learner.new_example()
        env.run_episode(learner)
        loss_val = loss_fn()(env.example)
        obj = learner.get_objective(loss_val, env)
        if not isinstance(obj, float):
            obj.backward()
            obj = obj.data[0]
        losses.append(loss_val)
        objs.append(obj)
        log_str = f'{epoch}' + f'\t{loss_val}' + f'\t{np.mean(losses[-100:])}'
        logs.append(log_str)
        if epoch % 100 == 0 or epoch == n_epochs:
            print(epoch, np.mean(losses[-100:]), np.mean(objs[-100:]))
    logdir = ''
    with open(logdir + '/stats.txt', 'w') as fout:
        fout.writelines('%s\n' % line for line in logs)


def test_vd_reslope(env, plr, vdlr, clr, clip, exp, exp_param):
    # run on CPU
    gpu_id = None
    # if len(sys.argv) == 1:
    util.reseed(90210, gpu_id=gpu_id)
    seeds = np.random.randint(0, 1e9, 10)
    # elif sys.argv[1] == 'fixed':
    #     seed = 90210
    # else:
    #     seed = int(sys.argv[1])
    # print('seed', seed)
    for i in range(10):
        util.reseed(seeds[i], gpu_id=gpu_id)
        test_vd_rl(environment_name=env, n_epochs=10000, plr=plr, vdlr=vdlr, clr=clr, grad_clip=clip, exp=exp,
                   exp_par=exp_param, run_id=i+1, save_log=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['vd_reslope', 'reslope', 'mcarlo', 'mcarlo-c', 'bellman'],
                        default='vd_reslope')
    parser.add_argument('--env', type=str, choices=['gridworld', 'gridworld_stoch', 'gridworld_ep',
                                                    'cartpole', 'hex', 'blackjack'],
                        help='Environment to run on', default='gridworld')
    parser.add_argument('--alr', type=float, help='Actor learning rate', default=0.0005)
    parser.add_argument('--vdlr', type=float, help='Value difference learning rate', default=0.005)
    parser.add_argument('--clr', type=float, help='Critic learning rate', default=0.005)
    parser.add_argument('--clip', type=float, help='Gradient clipping argument', default=10)
    parser.add_argument('--exp', type=str, help='Exploration method', default='eps-greedy',
                        choices=['eps-greedy', 'boltzmann', 'bootstrap'])
    parser.add_argument('--exp_param', type=float, help='Parameter for exp. method', default=4)
    args = parser.parse_args()
    # if args.tune == 0:
    #     xxx
    # else:
    #     # Set 10 seeds
    if args.method == 'vd_reslope':
        test_vd_reslope(args.env, args.alr, args.vdlr, args.clr, args.clip, args.exp, args.exp_param)
    # elif args.method == 'mcarlo-c':
    #     test_mcarlo_critic(args.env, args.alr, args.clr, args.clip, args.exp, args.exp_param)
    # elif args.method == 'reslope':
    #     test_reslope(args.env, args.alr, args.clip, args.exp, args.exp_param)
    # elif args.method == 'mcarlo':
    #     test_mcarlo(args.env, args.alr, args.clip, args.exp, args.exp_param)
    else:
        print('Invalid input')