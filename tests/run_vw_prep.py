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
    temp = 0.0
    if exp == 'bootstrap':
        policy = BootstrapPolicy(policy_fn=policy_fn, bag_size=4, n_actions=env.n_actions, greedy_predict=False,
                                 greedy_update=True)
        exploration = BanditLOLS.EXPLORE_BOOTSTRAP
        explore = 1.0
        expb = exp_par
    elif exp == 'boltzmann':
        policy = policy_fn()
        exploration = BanditLOLS.EXPLORE_BOLTZMANN
        explore = 1.0
        temp = exp_par
        expb = 0.0
    elif exp == 'eps-greedy':
        policy = policy_fn()
        exploration = BanditLOLS.EXPLORE_UNIFORM
        explore = exp_par
        expb = 0.0
    else:
        raise ValueError('Invalid exploration method')
    # Set up the initial value critic
    # ref_critic = Regressor(actor.dim, n_hid_layers=0, loss_fn='huber')
    ref_critic = Regressor(actor.dim, n_hid_layers=0)
    # Set up value difference regressor
    # vd_regressor = Regressor(2 * actor.dim + 1, n_hid_layers=1, loss_fn='huber')
    vd_regressor = Regressor(2 * actor.dim + 1, n_hid_layers=1)
    # Logging directory
    logdir = os.getcwd() + '/VDR_rl/' + environment_name + '_huber/' + exp + f'/{int(run_id)}'
    pathlib.Path(logdir).mkdir(parents=True)
    if save_log == True:
        writer = SummaryWriter(logdir)
    else:
        writer = None
    residual_loss_clip_fn = partial(np.clip, a_min=-200, a_max=200)
    learner = VdReslope(reference=None, policy=policy, ref_critic=ref_critic, vd_regressor=vd_regressor,
                        exploration=exploration, explore=explore, temperature=temp,
                        learning_method=BanditLOLS.LEARN_MTR,
                        save_log=save_log, writer=writer, actor=actor, attach_time=False,
                        residual_loss_clip_fn=residual_loss_clip_fn, expb=expb)
    print(learner)

    # Set up optimizers with different learning rates for the three networks
    # Policy parameters
    parameters = list(policy.parameters())
    optimizer = torch.optim.Adam(parameters, lr=plr)
    # Initial value critic
    critic_params = list(ref_critic.parameters())
    critic_optimizer = torch.optim.SGD(critic_params, lr=clr)
    # Value difference regressor
    vd_params = list(vd_regressor.parameters())
    vd_optimizer = torch.optim.Adam(vd_params, lr=vdlr)
    losses, objs = [], []
    reg_loss, ret_reg_loss, critic_loss, pred_loss = [], [], [], []
    for epoch in range(1, 1 + n_epochs):
        optimizer.zero_grad()
        critic_optimizer.zero_grad()
        vd_optimizer.zero_grad()
        env = mk_env()
        learner.new_example()
        env.run_episode(learner)
        loss_val = loss_fn()(env.example)
        obj, curr_reg_loss = learner.get_objective(loss_val, env)
        if not isinstance(obj, float):
            obj.backward()
            obj = obj.data[0]
        torch.nn.utils.clip_grad_norm(parameters, grad_clip)
        optimizer.step()
        torch.nn.utils.clip_grad_norm(vd_params, grad_clip)
        vd_optimizer.step()
        torch.nn.utils.clip_grad_norm(critic_params, grad_clip)
        critic_optimizer.step()
        losses.append(loss_val)
        objs.append(obj)
        reg_loss.append(curr_reg_loss[0])
        ret_reg_loss.append(curr_reg_loss[1])
        critic_loss.append(curr_reg_loss[2])
        pred_loss.append(curr_reg_loss[3])
        # log_str = f'{epoch}' + f'\t{loss_val}' + f'\t{np.mean(losses[-100:])}'
        log_str = f'{epoch}' + f'\t{loss_val}' + f'\t{np.mean(losses[-100:])}' + f'\t{np.var(losses[-100:])}' \
                  + f'\t{np.mean(reg_loss[-100:])}' + f'\t{np.mean(ret_reg_loss[-100:])}' \
                  + f'\t{np.mean(pred_loss[-100:])}' + f'\t{np.mean(critic_loss[-100:])}'
        logs.append(log_str)
        if epoch % 100 == 0 or epoch == n_epochs:
            print(epoch, np.mean(losses[-200:]), np.mean(objs[-200:]))
            if save_log == True:
                writer.add_scalar('Avg_loss', np.mean(losses[-200:]), epoch // 100)
    with open(logdir + '/stats.txt', 'w') as fout:
        fout.writelines('%s\n' % line for line in logs)

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
    parser.add_argument('--exp', type=str, help='Exploration method', default='bootstrap',
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