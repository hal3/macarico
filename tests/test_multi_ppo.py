from macarico.tasks.mountain_car import MountainCar
from macarico.tasks.mountain_car import MountainCarLoss
from macarico.tasks.mountain_car import MountainCarFeatures
from macarico.tasks.cartpole import CartPoleEnv
from macarico.tasks.cartpole import CartPoleFeatures
from macarico.tasks.cartpole import CartPoleLoss
from macarico.features.actor import TransitionBOW
from macarico.features.sequence import AttendAt
from macarico.policies.linear import LinearPolicy
from macarico.annealing import EWMA
from macarico.lts.ppo import PPO
from macarico.lts.reinforce import Reinforce
from argparse import ArgumentParser
import dynet as dy


def parse_arguments():
    ap = ArgumentParser()
    ap.add_argument('--eps', '-e', default='0.8', type=float,
                    help='epsilon for PPO')
    ap.add_argument('--task', '-t', default='mountaincar', type=str,
                    help='Taks: either cartpole or mountaincar')
    ap.add_argument('--learner', '-l', default='ppo', type=str,
                    help='Learner: either PPO or reinforce')
    return ap.parse_args()


def run_ppo(ex, actor, loss_fn, eps, learner_type):
    print(learner_type)
    print('Eps: ', eps)
    dy_model = dy.ParameterCollection()
    policy = LinearPolicy(dy_model, actor(dy_model), ex.n_actions, n_layers=1)
    baseline = EWMA(0.8)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.01)
    losses = []
    n_episodes = 10000
    # Total number of iterations
    I = 100
    # Number of episodes per iteration is N
    N = 10
    for i in range(I):
        for n in range(N):
            dy.renew_cg()
            if learner_type == 'ppo':
                learner = PPO(policy, baseline, eps)
            elif learner_type == 'reinforce':
                learner = Reinforce(policy, baseline)
            env = ex.mk_env()
            env.run_episode(learner)
            loss = loss_fn(ex, env)
            losses.append(loss)
        if n % 5 == 0:
            print('n: ', n, 'loss:',
                sum(losses[-500:]) / len(losses[-500:]))
        learner.update(loss)
        optimizer.update()


def test():
    print('')
    print('Proximal Policy Optimization')
    print('')
    args = parse_arguments()
    if args.task == 'mountaincar':
        print('Mountain Car')
        ex = MountainCar()
        run_ppo(
            ex,
            lambda dy_model:
            TransitionBOW(dy_model,
                          [MountainCarFeatures()],
                          [AttendAt(lambda _: 0, 'mountain_car')],
                          ex.n_actions),
            MountainCarLoss(),
            args.eps,
            args.learner,
        )
    elif args.task == 'cartpole':
        print('Cart Pole')
        ex = CartPoleEnv()
        run_ppo(
            ex,
            lambda dy_model:
            TransitionBOW(dy_model,
                          [CartPoleFeatures()],
                          [AttendAt(lambda _: 0, 'cartpole')],
                          ex.n_actions),
            CartPoleLoss(),
            args.eps,
            args.learner,
        )
    else:
        print('Unsupported Task!')
        exit(-1)

if __name__ == '__main__':
    test()
