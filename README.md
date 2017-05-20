# maçarico

An implementation of the imperative learning to search framework [1]
in pytorch, compatible with automatic differentiation, for deep
learning-based structured prediction and reinforcement learning.

[1] http://hal3.name/docs/daume16compiler.pdf

The basic structure is:

    macarico/
        base.py          defines the abstract classes used for maçarico,
                         such as Env, Policy, Features, Learner, Attention

        annealing.py     tools for annealing, useful for eg DAgger

        util.py          basic utility functions

        tasks/           example tasks, such as: sequence_labeler,
                         dependency_parser, sequence2sequence, etc. all of
                         these define an Env that can be run

        features/        contains example static features and dynamic features
            sequence.py  defines two types of static features: RNNFeatures
                         (obtained by running an RNN over the input) and
                         BOWFeatures (simple bag of words). also defines
                         useful attention models over sequences.

            actor.py     defines two types of dynamic features: TransitionRNN
                         (which is an actor that has an RNN-like hidden state),
                         and TransitionBOW (which has no hidden state and
                         instead just conditions on the previous actions)

        policies/        currently only implements a linear policy

        lts/             various learning to search algorithms, such as:
                         maximum_likelihood, dagger, reinforce,
                         aggrevate and LOLS

    tests/
        run_tests.sh     run all (or some) of the tests, compare the outputs
                         to previous versions to make sure you didn't botch
                         anything. (*please* run this before pushing changes.)

        test_util.py     some utilities for running tests, such as train/eval
                         loops, printint outputs, etc.

        nlp_data.py      generate or load data for various natural language
                         processing tasks. (requires external data.)

        test_X.py        various tests for different parts of maçarico. if you
                         develop something new, please create a test!

        output/          outputs from prvious runs

# To create a new task

Take a look at existing tasks.

Create a new mytask.py file in macarico/tasks that defines:

1. an `Example` class, which contains labeled examples for your
task. This class must define a `mk_env` function that returns an
environment (`Env`) particular to this task.

2. an `Env` class, which defines how your environment works. It must
provide `run_episode` and `loss` at the minimum. For some learning
algorithms, it must provide `rewind` (mostly for efficiency) and/or
`reference`.

3. if none of the existing attention models make sense for your task
(this is the case for e.g. `DependencyParser`), define your own
`Attention` mechanism.

4. make a test case in tests/test_mytask.py that tests it.


# To create new features

There are two types of features: static features (things that can be
precomputed on the input before the environment starts running) and
dynamic features (things that depend on the status of the
environment).

For static features (like `RNNFeatures`), create a class the derives
from `macarico.Features` (and probably also from `nn.Module` if it has
any of its own parameters). At a minimum, this must define its
dimensionality and give a name to itself (called the `field`). This
`field` can then be referenced either by other features or by
`Attention` modules. This should defined a `_forward` method that
computes the static features, and which will be cached automatically
for you. It should return a tensor of dimension (M,dim), where M is
arbitrary (but which must be compatible with `Attention`) and where
dim is the pre-declared dimensionality.

For dynamic features (like `TransitionRNN`), create a class as before.
However, instead of defining the static `_forward` function, you must
define your own dynamic `forward` function.  This can peek at
`state.t` and `state.T` to get the current and maximum time step. It
should return features /just/ for the current timestep, `state.t`.


# To create new attention mechanisms

An `Attention` mechanism tells a dynamic model where to look to access
its features. There are type types: hard attention and soft attention.

A hard attention mechanism defines its field (which features it is
attending to) and its arity (how many feature vectors does it attend
to at any given time).  Then, at runtime, given a `state`, it must
return the indices into the corresonding fields based on the state,
where the number of indices is exactly equal to its arity.

A soft attention mechanism still defines its field but declares its
arity to be `None`. This means that instead of returning an /index/
into its input, it must return a /distribution/ over its input as a
torch Variable tensor.


# To create new learners

The most basic type of `Learner` basically behaves like a `Policy`,
but additionally provides an `update` function that, for instance,
does backprop.

Perhaps the simplest example is `MaximumLikelihood`, which just
behaves according to a reference policy, but accumulates an objective
function that's the sum of individual predictions. At `update` time,
it runs backprop on this objective.

One *very important thing* in Learners, is that even if they do not
use the return value of their underlying policy, they *must* call the
underlying policy every time they run. Why? Because the underlying
policy may accumulate state (as in the case of `TransitionRNN`) and if
it is "skipped" the policy will become very confused because it will
have missed some input.

A slightly different example is `Reinforce`, which implements the
reinforce RL algorithm. This Learner does not explicitly accumulate an
objective that it then backprops on; instead it uses the fact that
stochastic choices can be backpropped through automatically using
torch's `.reinforce` function.


# Understanding how everything fits together

Because we have designed maçarico to be as modular as possible, there
are some places where the different pieces need to "talk" to each
other.

Let's take `test_sequence_labeler.py` as an example. In `test_demo`,
we have code that looks like:

    data = [Example(x, y, n_labels) for x, y in ...]

This constructs `sequence_labeler.Example` data structures and calls
them data.  If you look at the `Example` data structure, you find it
has two main components: `tokens` and `labels`, corresponding to `x`
and `y` respectively, above.

Next, we build some *static* features:

    features = RNNFeatures(n_types,
			   input_field  = 'tokens',
		           output_field = 'tokens_feats')

This constructs a biLSTM over the inputs. Where does it look? It looks
in `tokens` because that's the specified input field. And it stores
the features generated by the biLSTM in `tokens_feats`. You can
therefore think of `features` as something that maps from `tokens` to
tokens_feats`. (Note: those two arguments are the default and could
have been left off for convenience, but here we're trying to make
everything explicit.)

Next, we need an actor. The actor is the thing that takes a state of
the world and produces a feature representation. (This feature
representation will later be consumed by the `Policy` to predict an
action.) However, the actor needs to *attend* somewhere when making
predictions. In this case, when the environment (the sequence labeler)
is predicting the label of the `n`th word, the actor should look at
that word! This can be done with the `AttendAt` attention mechanism.

    attention = AttendAt(field='tokens_feats',
		         get_position=lambda state: state.n)

This constructs an attention mechanism that essentially returns
`tokens_feats[state.n]` when the environment state is on word
`n`. Note that this hinges on the fact that we *know* that the
environment stores "current position" in `state.n`. (Again, these
arguments are the default and could be left off.)

Next, we can construct the actor. The actor itself an RNN (not
bidirectional this time), which uses the biLSTM features we build
above, together with the simple attention mechanism.

    actor = TransitionRNN([features],
			  [attention],
		          n_labels)

Finally, we can construct the policy. In this case, it's just a linear
function that maps from the `actor`'s feature represention to one of
`n_labels` actions:

    policy = LinearPolicy(actor, n_labels)

Tracing back through this. The policy maps from a state feature
representation to an action. This mapping is done by
`LinearPolicy`. But where does the state feature representation come
from? It comes from the `actor`, which, when labeling word `n` asks
its attention model(s) what features to use. In this case, the
attention tells it to look at `tokens_feats[n]`, where
`tokens_feats[n]` is the output of the biLSTM.

We will now train this model with DAgger. In order to do this, we need
to anneal the degree to which rollin is done according to the
reference policy versus the learned policy:

    p_rollin_ref = stochastic(ExponentialAnnealing(0.99))

Next, we construct an optimizer. This is exactly like you would do in
pytorch, extracting parameters from the `policy`:

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)

And now we can train:

    for epoch in xrange(5):
    	# train on each example, one at a time
    	for ex in data:
	    optimizer.zero_grad()
	    learner = DAgger(ref, policy, p_rollin_ref)
	    env = ex.mk_env()
	    env.run_episode(learner)
	    learner.update(env.loss())
	    optimizer.step()
	    p_rollin_ref.step()

	# now make some predictions
	for ex in data:
	    env = ex.mk_env()
	    out = env.run_episode(policy)
	    print 'prediction = %s' % out

