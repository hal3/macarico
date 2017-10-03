#!/bin/bash

mkdir bbl_sup

for opt in adam rmsprop ; do
    for lr in 0.001 0.005 0.01 0.05 0.1 ; do
	for alg in dagger aggrevate ; do
	    for p_rin in 0.0 0.999 0.99999 1.0 ; do
		for task in pos-tweet dep-tweet ctb-nw ; do
		    fname="$opt"_"$lr"_"$alg"_"$p_rin"_"$task"
		    
		    slush \
			PYTHONPATH=/fs/clip-ml/hal/projects/macarico \
		        /fs/clip-ml/hal/pyd/bin/python big_banditlols.py \
			$task \
			$alg::p_rin=$p_rin \
			$opt \
			$lr \
			reps=5 \
			embed=yes \
			save=bbl_sup/$fname.model \
			\> bbl_sup/$fname.err 2\>\&1
		done
	    done
	done
    done
done
