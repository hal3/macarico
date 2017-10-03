#!/bin/bash

mkdir bbl_traingrid

algs=""
# reinforce
for baseline in 0.0 0.2 0.5 0.8 0.9 ; do
    algs="$algs reinforce::baseline=$baseline"
    algs="$algs reinforce::baseline=$baseline::maxd=1"
done

# a2c
for mult in 0.2 0.5 1.0 2.0 5.0 ; do
    algs="$algs aac::mult=$mult"
done

# blols
for update in ips dr mtr ; do
    for multidev in '' '::multidev' ; do
	for upc in '' '::upc' ; do
	    for oft in '' '::oft' ; do
		# uniform exploration
		algs="$algs blols::$update$multidev$upc$oft::uniform"
		
		# boltzmann exploration; tune temperature
		for temp in 0.2 1.0 5.0 ; do
		    algs="$algs blols::$update$multidev$upc$oft::boltzmann::temp=$temp"
		done

		# bootstrap exploration; vary number of policies and greediness
		for bag_size in 3 5 10 ; do
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size"
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size::greedy_update"
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size::greedy_predict"
		    algs="$algs blols::$update$multidev$upc$oft::bootstrap::bag_size=$bag_size::greedy_predict::greedy_update"
		done
	    done
	done
    done
done

for opt in adam ; do
    for lr in 0.0001 0.0005 0.001 0.005 0.01 ; do
	for p_layers in 2 ; do
	    for p_dim in 20 ; do
		for alg in $(echo $algs | tr ' ' '\n' | sort -R) ; do
		    fname="$opt"_"$lr"_"$alg"_"$p_layers"_"$p_dim"
		    echo \
			PYTHONPATH=/fs/clip-ml/hal/projects/macarico \
			/fs/clip-ml/hal/pyd/bin/python big_banditlols.py \
			grid \
			$alg \
			$opt \
			$lr \
			reps=3 \
			p_layers=$p_layers \
			p_dim=$p_dim \
			\> bbl_traingrid/$fname.err 2\>\&1
		done
	    done
	done
    done
done
