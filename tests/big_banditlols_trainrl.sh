#!/bin/bash

mkdir bbl_trainrl

task=$1

# depending on task, set up algorithm arguments
if [[ "$task" == "pos-wsj" ]] ; then
    embed=300
    d_rnn=50
    n_layers=1
    p_layers=1
    load=bbl_sup_size/MODEL
elif [[ "$task" == "dep-wsj" ]] ; then
    embed=300
    d_rnn=50
    n_layers=1
    p_layers=1
    load=bbl_sup_size/MODEL
elif [[ "$task" == "ctb-sc" ]] ; then
    embed=300
    d_rnn=50
    n_layers=1
    p_layers=1
    load=bbl_sup_size/MODEL
else
    echo "unknown task $task"
    exit
fi
    

for opt in adam ; do
    for lr in 0.0001 0.0005 0.001 0.005 0.01 ; do
	for alg in aac::mult=5.0 aac::mult=2.0 aac::mult=1.0 aac::mult=0.5 aac::mult=0.2 reinforce::baseline=0.0 reinforce::baseline=0.8 reinforce::baseline=0.0::maxd=1 reinforce::baseline=0.8::maxd=1 dagger::p_rin=0.0 dagger::p_rin=0.999 dagger::p_rin=0.99999 ; do
	    for decay in 0.0 0.01 0.1 ; do

		supervised=""
		if [[ "$(echo $alg | cut -c1-6)" == "dagger" ]] ; then
		    supervised="supervised"
		fi
	    
		fname="$opt"_"$lr"_"$alg"_"$task"
		echo \
		    PYTHONPATH=/fs/clip-ml/hal/projects/macarico \
		    /fs/clip-ml/hal/pyd/bin/python big_banditlols.py \
		    $task \
		    $alg \
		    $opt \
		    $lr \
		    reps=5 \
		    embed=yes \
                    f=rnn::$d_rnn::$n_layers \
                    p_layers=$p_layers \
		    load=$load \
		    save=bbl_trainrl/$fname.model \
		    $supervised \
                    --dynet-weight-decay=$decay \
		    \> bbl_trainrl/$fname.err 2\>\&1
	    done
	done
    done
done
