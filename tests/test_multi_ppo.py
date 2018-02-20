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
    ap.add_argument('--dynet-seed', type=int, required=False)
    return ap.parse_args()


def run_ppo(ex, actor, loss_fn, eps, learner_type):
    print(learner_type)
    print('Eps: ', eps)
    dy_model = dy.ParameterCollection()
    policy = LinearPolicy(dy_model, actor(dy_model), ex.n_actions, n_layers=1)
    baseline = EWMA(0.8)
    optimizer = dy.AdamTrainer(dy_model, alpha=0.01)
    # Total number of iterations
    I = 10000
    # Number of episodes per iteration is N
    N = 5
    # Number of epochs K
    K = 200
    # Mini-batch size M, M <= N * T
    T = ex.T
    M = T
    running_loss = []
    for i in range(I):
        learners = []
        losses = []
        # TODO is this the correct place to implement the renew_cg() func?
        dy.renew_cg()
        for n in range(N):
            learner = PPO(policy, baseline, eps)
            env = ex.mk_env()
            env.run_episode(learner)
            loss = loss_fn(ex, env)
            learners.append(learner)
            losses.append(loss)
            running_loss.append(loss)
        for k in range(K):
            for learner, loss in zip(learners, losses):
                learner.update_ppo(loss)
                optimizer.update()
        print('episode: ', i, 'loss:',
              sum(running_loss[-500:]) / len(running_loss[-500:]))

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
