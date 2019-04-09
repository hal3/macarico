import macarico.util
from macarico.lts.reinforce import Reinforce, AdvantageActorCritic, LinearValueFn
from macarico.annealing import EWMA
from macarico.tasks.cartpole import CartPoleEnv, CartPoleFeatures, CartPoleLoss
from test_pocman import run_environment
from macarico.features.actor import TransitionBOW
from macarico.features.sequence import AttendAt

macarico.util.reseed(90210)


baseline = None
def reinforce(policy):
    global baseline
    if baseline is None:
        baseline = EWMA(0.8)
    return Reinforce(policy, baseline)

def a2c(policy):
    global baseline
    if baseline is None:
        baseline = LinearValueFn(6)
    return AdvantageActorCritic(policy,
                                baseline,
                                disconnect_values=True,
                                vfa_multiplier=1.0,
                                temperature=1.0)
    

def test():
    print('')
    print('Cart Pole')
    print('')
    ex = CartPoleEnv()
    run_environment(
        ex,
        lambda:
        TransitionBOW([CartPoleFeatures()],
                      [AttendAt(lambda _: 0, 'cartpole')],
                      ex.n_actions),
        CartPoleLoss(),
        rl_alg=reinforce,
        n_epochs=100,
        lr=0.1,
    )


if __name__ == '__main__':
    test()
