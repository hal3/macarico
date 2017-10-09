from macarico.tasks.cartpole import CartPoleEnv
from macarico.tasks.cartpole import CartPoleFeatures
from macarico.tasks.cartpole import CartPoleLoss
from test_pocman import run_environment
from macarico.features.actor import TransitionBOW
from macarico.features.sequence import AttendAt


def test():
    print('')
    print('Cart Pole')
    print('')
    ex = CartPoleEnv()
    run_environment(
        ex,
        lambda dy_model:
        TransitionBOW(dy_model,
                      [CartPoleFeatures()],
                      [AttendAt(lambda _: 0, 'cartpole')],
                      ex.n_actions),
        CartPoleLoss(),
    )


if __name__ == '__main__':
    test()
