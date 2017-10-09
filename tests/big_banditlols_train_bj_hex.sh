#!/bin/bash


algs=""
# reinforce
for baseline in 0.0 0.8 ; do
    algs="$algs reinforce::baseline=$baseline"
    algs="$algs reinforce::baseline=$baseline::maxd=1"
done

# a2c
for mult in 0.2 0.5 1.0 2.0 5.0 ; do
    algs="$algs aac::mult=$mult"
done

# blols
for update in dr mtr ; do
    for multidev in '' '::multidev' ; do
	for upc in '' '::upc' ; do
	    for oft in '' '::oft' ; do
		# uniform exploration
		algs="$algs blols::$update$multidev$upc$oft::uniform"
		
		# boltzmann exploration; tune temperature
		for temp in 1.0 ; do
		    algs="$algs blols::$update$multidev$upc$oft::boltzmann::temp=$temp"
		done

		# bootstrap exploration; vary number of policies and greediness
		for bag_size in 5 ; do
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size"
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size::greedy_update"
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size::greedy_predict"
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size::greedy_predict::greedy_update"
		done
	    done
	done
    done
done

for task in blackjack hex ; do
    mkdir bbl_train$task
done

for opt in adam ; do
    for lr in 0.0005 0.001 0.005 0.01 ; do
	for p_layers in 1 2 ; do
	    for p_dim in 20 50 ; do
		if [[ "$p_layers" == "1" && "$p_dim" != 20 ]] ; then continue ; fi
		for alg in $(echo $algs | tr ' ' '\n' | sort -R) ; do
		    for task in blackjack hex ; do
			fname="$opt"_"$lr"_"$alg"_"$p_layers"_"$p_dim"
			echo \
			    PYTHONPATH=/fs/clip-ml/hal/projects/macarico \
			    /fs/clip-ml/hal/pyd/bin/python big_banditlols.py \
			    $task \
			    $alg \
			    $opt \
			    $lr \
			    reps=3 \
			    p_layers=$p_layers \
			    p_dim=$p_dim \
			    \> bbl_train$task/$fname.err 2\>\&1
		    done
		done
	    done
	done
    done
done
