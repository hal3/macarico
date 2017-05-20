r"""The macarico (truly, there should be a cedilla on the "c") package
contains a library for structured prediction and reinforcement
learning on top of pytorch (which must be installed properly, as must
numpy).

Macarico splits the universe into four pieces: (1) tasks (like
sequence labeling or dependency parsing in structured prediction, or
gridworld or mountain car in reinforcement learning); (2) features
(like bag of words, or biLSTM or convnets); (3) learners (like
reinforce or DAgger or LOLS); and (4) policies (like a simple linear
policy).

The goal of the design is to make it easy to add new tasks (if you're
doing applied stuff) and get features and learning "for free", as well
as add new RL or structured prediction algorithms and get features and
tasks "for free".

The easiest places to start exploring is the tests directory.
"""

from base import *
