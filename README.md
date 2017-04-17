# macarico

An implementation of the imperative learning to search framework [1]
in pytorch, compatible with automatic differentiation, for deep
learning-based structured prediction and reinforcement learning.

[1] http://hal3.name/docs/daume16compiler.pdf

The basic structure is:

    macarico/
        lts.py                    defines the base interface for
			          building a learning to search
			          learning algorithm (not task)
			          ex: dagger.py, maximum_likelihood.py
			      
        search_task.py            defines the base interface for
			          constructing a new task
			          ex: tasks/sequence_labeler.py

        annealing.py	          some useful strategies for annealing

	maximum_likelihood.py     example lts algorithms of varying
	dagger.py		  complexity, both as examples
	reinforce.py	  	  and cuz they're useful

    tasks/
        sequence_labeler.py	  an example biRNN sequence labeling
				  example for testing

    tests/
	test_sequence_labeler.py  test cases for sequence_labeler.py



