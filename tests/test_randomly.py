import argparse
import sys
from functools import partial

from tensorboardX import SummaryWriter

import macarico.data.synthetic as synth
import macarico.tasks.blackjack as blackjack
import macarico.tasks.cartpole as cartpole
import macarico.tasks.dependency_parser as dep
import macarico.tasks.gridworld as gridworld
import macarico.tasks.hexgame as hexgame
import macarico.tasks.mdp as mdp
import macarico.tasks.mountain_car as car
import macarico.tasks.pendulum as pendulum
import macarico.tasks.pocman as pocman
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
from macarico.lts.aggrevate import AggreVaTe
from macarico.lts.behavioral_cloning import BehavioralCloning
from macarico.lts.dagger import DAgger, Coaching
from macarico.lts.lols import LOLS, BanditLOLS
from macarico.lts.monte_carlo import MonteCarlo
from macarico.lts.reinforce import Reinforce, LinearValueFn, A2C
from macarico.lts.reslope import Reslope
from macarico.lts.vd_reslope import VdReslope
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


def build_learner(n_types, n_actions, horizon, ref, loss_fn, require_attention):
    dim = 50
    features = RNN(EmbeddingFeatures(n_types, d_emb=dim), d_rnn=dim, cell_type='LSTM')
    features = BOWFeatures(n_types)
    attention = (require_attention or AttendAt)(features)
    actor = BOWActor([attention], n_actions)
    #policy = WMCPolicy(actor, n_actions)
    policy = CSOAAPolicy(actor, n_actions)
    learner = BehavioralCloning(policy, ref)
    #learner = AggreVaTe(policy, ref)
    #learner = LOLS(policy, ref, loss_fn())
    #learner = Reinforce(policy)
    #value_fn = LinearValueFn(actor)
    #learner = A2C(policy, value_fn)
    #LOLS, BanditLOLS, Reinforce, A2C
    return policy, learner, list(policy.parameters()) #+ list(value_fn.parameters())


def build_reslope_learner(n_types, n_actions, horizon, ref, loss_fn, require_attention):
    # compute base features
    features = BOWFeatures(n_types)
    # compute some attention
    attention = [AttendAt(features, position=lambda _: 0)]
    # build an actor
#    actor = TimedBowActor(attention, n_actions, horizon, act_history_length=0, obs_history_length=0)
    actor = BOWActor(attention, n_actions, act_history_length=0, obs_history_length=0)
    # build the policy
    policy_fn = lambda: CSOAAPolicy(actor, n_actions, 'squared')
    exploration = 'bootstrap'
    if exploration == 'bootstrap':
        policy = BootstrapPolicy(policy_fn=policy_fn, bag_size=3, n_actions=n_actions)
        exploration = BanditLOLS.EXPLORE_BOOTSTRAP
    else:
        policy = policy_fn()
        exploration = BanditLOLS.EXPLORE_BOLTZMANN

    parameters = list(policy.parameters())
    # build the reslope learner
    p_ref = stochastic(ExponentialAnnealing(0.9))

    class NoRef(object):
        def step(self):
            pass

        def __call__(self):
            return False

    temp = 2*0.0001
    learner_type = 'monte-carlo'
    if learner_type == 'reslope':
        learner = Reslope(exploration=exploration, reference=ref, policy=policy, p_ref=NoRef(), explore=1.0,
                          temperature=temp, update_method=BanditLOLS.LEARN_MTR)
    elif learner_type == 'vd-reslope':
        ref_critic = Regressor(actor.dim)
        vd_regressor = Regressor(2*actor.dim+2, n_hid_layers=1)
        parameters += list(ref_critic.parameters())
        parameters += list(vd_regressor.parameters())
        temp = 0.1
        save_log = False
        logdir = 'VDR_sl' #+ f'/temp-{temp}' + f'_plr-{plr}' + f'_vdlr-{vdlr}' + f'_clr-{clr}' + f'_gc-{grad_clip}'
        writer = SummaryWriter(logdir)
        residual_loss_clip_fn = partial(np.clip, a_min=-2, a_max=2)
        learner = VdReslope(exploration=exploration, reference=None, policy=policy, ref_critic=ref_critic,
                            vd_regressor=vd_regressor, p_ref=stochastic(NoAnnealing(0)), temperature=temp,
                            learning_method=BanditLOLS.LEARN_MTR, save_log=save_log, writer=writer, actor=actor,
                            residual_loss_clip_fn=residual_loss_clip_fn)
    else:
        learner = MonteCarlo(policy, reference=None, p_rollin_ref=NoAnnealing(0), p_rollout_ref=NoAnnealing(0),
                             update_method=BanditLOLS.LEARN_MTR, exploration=exploration, p_explore=NoAnnealing(1.0),
                             mixture=LOLS.MIX_PER_ROLL, temperature=temp, is_episodic=True)

    print('learner: ', learner)
    return policy, learner, parameters


def build_random_learner(n_types, n_actions, horizon, ref, loss_fn, require_attention):
    # compute base features
    features = np.random.choice([lambda: EmbeddingFeatures(n_types),
                                 lambda: BOWFeatures(n_types)])()
    # optionally run RNN or CNN
    features = np.random.choice([lambda: features,
                                 lambda: RNN(features,
                                             cell_type=np.random.choice(['RNN', 'GRU', 'LSTM']),
                                             bidirectional=np.random.random() < 0.5),
                                 lambda: DilatedCNN(features)])()
    # maybe some nn magic
    if np.random.random() < 0.5:
        features = macarico.Torch(features,
                                  50, # final dimension, too hard to tell from list of layers :(
                                  [nn.Linear(features.dim, 50),
                                   nn.Tanh(),
                                   nn.Linear(50, 50),
                                   nn.Tanh()])
    # compute some attention
    if require_attention is not None:
        attention = [require_attention(features)]
    else:
        attention = [np.random.choice([lambda: AttendAt(features, 'n'), # or `lambda s: s.n`
                                       lambda: AverageAttention(features),
                                       lambda: FrontBackAttention(features),
                                       lambda: SoftmaxAttention(features)])()] # note: softmax doesn't work with BOWActor
        if np.random.random() < 0.2:
            attention.append(AttendAt(features, lambda s: s.N-s.n))

    # build an actor
    if any((isinstance(x, SoftmaxAttention) for x in attention)):
        actor = RNNActor(attention, n_actions)
    else:
        actor = np.random.choice([lambda: RNNActor(attention,
                                                   n_actions,
                                                   d_actemb=np.random.choice([None,5]),
                                                   cell_type=np.random.choice(['RNN', 'GRU', 'LSTM'])),
                               lambda: BOWActor(attention, n_actions, act_history_length=3, obs_history_length=2)])()

    # do something fun: add a torch module in the middle
    if np.random.random() < 0.5:
        actor = macarico.Torch(actor,
                               27, # final dimension, too hard to tell from list of layers :(
                               [nn.Linear(actor.dim, 27),
                                nn.Tanh()])
    # build the policy
    policy = np.random.choice([lambda: CSOAAPolicy(actor, n_actions, 'huber'),
                               lambda: CSOAAPolicy(actor, n_actions, 'squared'),
                               lambda: WMCPolicy(actor, n_actions, 'huber'),
                               lambda: WMCPolicy(actor, n_actions, 'hinge'),
                               lambda: WMCPolicy(actor, n_actions, 'multinomial'), ])()
    parameters = policy.parameters()

    # build the learner
    if np.random.random() < 0.1:
        value_fn = LinearValueFn(actor)
        # A2C
        learner = A2C(policy, value_fn)
        parameters = list(parameters) + list(value_fn.parameters())
    else:
        learner = np.random.choice([BehavioralCloning(policy, ref),
                                    DAgger(policy, ref), #, ExponentialAnnealing(0.99))
                                    Coaching(policy, ref, policy_coeff=0.1),
                                    AggreVaTe(policy, ref),
                                    Reinforce(policy),
                                    BanditLOLS(policy, ref),
                                    LOLS(policy, ref, loss_fn)])
    return policy, learner, parameters


def test_rl(environment_name, n_epochs=10000):
    print('rl', environment_name)
    tasks = {
        'pocman': (pocman.MicroPOCMAN, pocman.LocalPOCFeatures, pocman.POCLoss, pocman.POCReference),
        'cartpole': (cartpole.CartPoleEnv, cartpole.CartPoleFeatures, cartpole.CartPoleLoss, None),
        'blackjack': (blackjack.Blackjack, blackjack.BlackjackFeatures, blackjack.BlackjackLoss, None),
        'hex': (hexgame.Hex, hexgame.HexFeatures, hexgame.HexLoss, None),
        'gridworld': (gridworld.make_default_gridworld, gridworld.LocalGridFeatures, gridworld.GridLoss, None),
        'pendulum': (pendulum.Pendulum, pendulum.PendulumFeatures, pendulum.PendulumLoss, None),
        'car': (car.MountainCar, car.MountainCarFeatures, car.MountainCarLoss, None),
        'mdp': (lambda: synth.make_ross_mdp()[0], lambda: mdp.MDPFeatures(3), mdp.MDPLoss, lambda: synth.make_ross_mdp()[1]),
    }
              
    mk_env, mk_fts, loss_fn, ref = tasks[environment_name]
    env = mk_env()
    features = mk_fts()
    
    attention = AttendAt(features, position=lambda _: 0)
    actor = np.random.choice([BOWActor([attention], env.n_actions), RNNActor([attention], env.n_actions, cell_type = 'LSTM', d_actemb = None)])
    policy = CSOAAPolicy(actor, env.n_actions)
    # learner = Reinforce(policy)
    learner = Reslope(reference=None,policy=policy,p_ref=stochastic(NoAnnealing(0)), deviation='single')
    # learner = Reslope(reference=None,policy=policy,p_ref=stochastic(NoAnnealing(0)))
    # learner = BanditLOLS(policy=policy)
    print(learner)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    losses, objs = [], []
    for epoch in range(1, 1+n_epochs):
        optimizer.zero_grad()
        env = mk_env()
        learner.new_example()
        env.run_episode(learner)
        loss_val = loss_fn()(env.example)
        obj = learner.get_objective(loss_val)
        if not isinstance(obj, float):
            obj.backward()
            obj = obj.data[0]
        optimizer.step()
        losses.append(loss_val)
        objs.append(obj)
        if epoch%100 == 0 or epoch==n_epochs:
            print(epoch, np.mean(losses[-500:]), np.mean(objs[-500:]))


def test_vd_rl(environment_name, exp, exp_par, n_epochs=10000, plr=0.001, vdlr=0.001, clr=0.001, grad_clip = 1,
               save_log=False):
    print('rl', environment_name)
    tasks = {
        'pocman': (pocman.MicroPOCMAN, pocman.LocalPOCFeatures, pocman.POCLoss, pocman.POCReference),
        'cartpole': (cartpole.CartPoleEnv, cartpole.CartPoleFeatures, cartpole.CartPoleLoss, None),
        'blackjack': (blackjack.Blackjack, blackjack.BlackjackFeatures, blackjack.BlackjackLoss, None),
        'hex': (hexgame.Hex, hexgame.HexFeatures, hexgame.HexLoss, None),
        'gridworld': (gridworld.make_default_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
        'gridworld_stoch': (gridworld.make_stochastic_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
        'gridworld_ep': (gridworld.make_episodic_gridworld, gridworld.GlobalGridFeatures, gridworld.GridLoss, None),
        'pendulum': (pendulum.Pendulum, pendulum.PendulumFeatures, pendulum.PendulumLoss, None),
        'car': (car.MountainCar, car.MountainCarFeatures, car.MountainCarLoss, None),
        'mdp': (lambda: synth.make_ross_mdp()[0], lambda: mdp.MDPFeatures(3), mdp.MDPLoss, lambda: synth.make_ross_mdp()[1]),
    }
    logs = []
    logs.append('Epoch\tloss_val\tAvg_loss')
              
    mk_env, mk_fts, loss_fn, ref = tasks[environment_name]
    env = mk_env()
    #Compute features
    if 'gridworld' in environment_name:
        features = mk_fts(4,4)
    else:
        features = mk_fts()
    # Compute some attention
    attention = [AttendAt(features, position=lambda _: 0)]
    # Build an actor
    actor = TimedBowActor(attention, env.n_actions, env.horizon(), act_history_length=0, obs_history_length=0)
    # actor = BOWActor(attention, env.n_actions, act_history_length=0, obs_history_length=0)
    # Build the policy
    policy_fn = lambda: CSOAAPolicy(actor, env.n_actions)
    temp = 0.0
    if exp == 'bootstrap':
        policy = BootstrapPolicy(policy_fn=policy_fn, bag_size=4, n_actions=env.n_actions)
        exploration = BanditLOLS.EXPLORE_BOOTSTRAP
        explore = exp_par
    elif exp == 'boltzmann':
        policy = policy_fn()
        exploration = BanditLOLS.EXPLORE_BOLTZMANN
        explore = 1.0
        temp = exp_par
    elif exp == 'eps-greedy':
        policy = policy_fn()
        exploration = BanditLOLS.EXPLORE_UNIFORM
        explore = exp_par
    # Set up the initial value critic
    ref_critic = Regressor(actor.dim)
    # Set up value difference regressor
    vd_regressor = Regressor(2*actor.dim+1, n_hid_layers=1)
    # Logging directory
    logdir = 'VDR_rl/'+environment_name
    writer = SummaryWriter(logdir)
    residual_loss_clip_fn = partial(np.clip, a_min=-2, a_max=3)
    learner = VdReslope(reference=None, policy=policy, ref_critic=ref_critic, vd_regressor=vd_regressor,
                        exploration=exploration, explore=explore, temperature=temp, learning_method=BanditLOLS.LEARN_MTR,
                        save_log=save_log, writer=writer, actor=actor, attach_time=False,
                        residual_loss_clip_fn=residual_loss_clip_fn)
    # learner = BanditLOLS(reference=None, policy=policy, p_rollout_ref=NoAnnealing(0),
    #                      exploration=exploration, temperature=temp)
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
    for epoch in range(1, 1+n_epochs):
        optimizer.zero_grad()
        critic_optimizer.zero_grad()
        vd_optimizer.zero_grad()
        env = mk_env()
        learner.new_example()
        env.run_episode(learner)
        loss_val = loss_fn()(env.example)
        obj = learner.get_objective(loss_val, env)
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
        log_str = f'{epoch}' + f'\t{loss_val}' + f'\t{np.mean(losses[-100:])}'
        logs.append(log_str)
        if epoch%100 == 0 or epoch==n_epochs:
            print(epoch, np.mean(losses[-200:]), np.mean(objs[-200:]))
            if save_log == True:
              writer.add_scalar('Avg_loss', np.mean(losses[-200:]), epoch//100)
    with open(logdir + '/stats.txt', 'w') as fout:
        fout.writelines('%s\n' % line for line in logs)


def test_reslope_sp(environment_name, n_epochs=1, n_examples=4, fixed=False, gpu_id=None):
    return test_sp(environment_name, n_epochs, n_examples, fixed, gpu_id, builder=build_reslope_learner)


def test_sp(environment_name, n_epochs=1, n_examples=4, fixed=False, gpu_id=None, builder=None):
    print('sp', environment_name)
    n_types = 50 if fixed else 10
    length = [4, 5, 6] if fixed else 4
    n_labels = 9 if fixed else 3

    mk_env = None
    if environment_name == 'sl':
        n_types = 2
        n_labels = 2
        length = 1
        data = synth.make_sequence_mod_data(n_examples, length, n_types, n_labels)
        mk_env = sl.SequenceLabeler
        loss_fn = sl.HammingLoss
        ref = sl.HammingLossReference()
        require_attention = None
    elif environment_name == 'dep':
        data = [Dependencies(tokens=[0, 1, 2, 3, 4], heads=[1, 5, 4, 4, 1], token_vocab=5) for _ in range(n_examples)]
        mk_env = dep.DependencyParser
        loss_fn = dep.AttachmentLoss
        ref = dep.AttachmentLossReference()
        require_attention = dep.DependencyAttention
    elif environment_name == 's2s':
        data = synth.make_sequence_mod_data(n_examples, length, n_types, n_labels, include_eos=True)
        mk_env = s2s.Seq2Seq
        loss_fn = s2s.EditDistance
        ref = s2s.NgramFollower()
        require_attention = AttendAt# SoftmaxAttention
    elif environment_name == 's2j':
        data = synth.make_json_mod_data(n_examples, length, n_types, n_labels)
        mk_env = lambda ex: s2j.Seq2JSON(ex, n_labels, n_labels)
        loss_fn = s2j.TreeEditDistance
        ref = s2j.JSONTreeFollower()
        require_attention = FrontBackAttention

    if builder is None:
        builder = build_learner if fixed else build_random_learner

    n_actions = mk_env(data[0]).n_actions
    while True:
        policy, learner, parameters = builder(n_types, n_actions, length, ref, loss_fn, require_attention)
        if fixed or not (environment_name in ['s2s','s2j'] and (isinstance(learner, AggreVaTe) or isinstance(learner, Coaching))):
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
    util.TrainLoop(mk_env, policy, learner, optimizer,
#                   print_freq=2,
                   losses=[loss_fn, loss_fn, loss_fn],
                   progress_bar=False,
                   minibatch_size=np.random.choice([1]),).train(data[len(data)//2:], dev_data = data[:len(data)//2],
                                                                  n_epochs=n_epochs)


def test_reslope():
    # run on CPU
    gpu_id = None
    seed = 90210
    print('seed', seed)
    util.reseed(seed, gpu_id=gpu_id)
    test_reslope_sp(environment_name='sl', n_epochs=1, n_examples=2*2*2*2*2**12, fixed=True, gpu_id=gpu_id)


def test_vd_reslope(env, plr, vdlr, clr, clip, exp, exp_param):
    # run on CPU
    gpu_id = None
    # if len(sys.argv) == 1:
    seed = np.random.randint(0, 1e9)
    # elif sys.argv[1] == 'fixed':
    #     seed = 90210
    # else:
    #     seed = int(sys.argv[1])
    # print('seed', seed)
    util.reseed(seed, gpu_id=gpu_id)
    test_vd_rl(environment_name=env, n_epochs=10000, plr=plr, vdlr=vdlr, clr=clr, grad_clip=clip,exp=exp,
               exp_par=exp_param, save_log=True)


def test_all_random():
    gpu_id = None # run on CPU
    fixed = False
    if len(sys.argv) == 1:
        seed = np.random.randint(0, 1e9)
    elif sys.argv[1] == 'fixed':
        seed = 90210
        fixed = True
    else:
        seed = int(sys.argv[1])
    print('seed', seed)
    util.reseed(seed, gpu_id=gpu_id)
    if fixed or np.random.random() < 1.0:
        #    if fixed or np.random.random() < 0.8:
        test_sp(environment_name='s2j' if fixed \
            #                else np.random.choice(['sl', 'dep', 's2s', 's2j']),
        else np.random.choice(['sl', 'dep', 's2s']),
                n_epochs=1,
                n_examples=2**12 if fixed else 4,
                fixed=fixed,
                gpu_id=gpu_id)
    else:
        test_rl(np.random.choice('pocman cartpole blackjack hex gridworld pendulum car mdp'.split()),
                n_epochs=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['vd_reslope', 'reslope'], default='vd_reslope')
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
    if args.method == 'vd_reslope':
        test_vd_reslope(args.env, args.alr, args.vdlr, args.clr, args.clip, args.exp, args.exp_param)
    elif args.method == 'reslope':
        test_reslope()
    else:
        print('Invalid input')
