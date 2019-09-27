#!/usr/bin/env bash

env='gridworld_ep'
#for env in 'gridworld2' 'gridworld_det' 'gridworld_stoch' ; do
#for alr in 0.01 0.005 ; do
#    for vdlr in 0.05 ; do
#        for clr in 0.05 ; do
#	        for temp in 0.01 ; do
#	            for grad_clip in 1 ; do
#			        python test_randomly.py --env $env --alr $alr --vdlr $vdlr --clr $clr --temp $temp --clip $grad_clip
#			    done
#		    done
#		done
#	done
#done
#done

#method='vd_reslope'
#for env in 'gridworld2' 'gridworld_det' 'gridworld_stoch' ; do
#for alr in 0.01 0.005 0.001 0.0005 ; do
#    for vdlr in 0.05 0.01 0.005 0.001 ; do
#        for clr in 0.05 0.01 0.005 0.001 ; do
#	        for temp in 0.01 0.1 0.2 0.5 ; do
#	            for grad_clip in 1 10 20 ; do
#			        python test_randomly.py --alr $alr --vdlr $vdlr --clr $clr --temp $temp --clip $grad_clip 1>&- 2>&- &
#			    done
#		    done
#		done
#	done
#done
#done

#for env in 'gridworld2' 'gridworld_det' 'gridworld_stoch' ; do
for alr in 0.001 0.0005 ; do
    for vdlr in 0.05 0.01 0.005 ; do
        for clr in 0.05 0.01 0.005 ; do
	        for temp in 0.1 0.2 ; do
	            for grad_clip in 1 10 ; do
	            for i in 1 ; do
			        python test_randomly.py --env $env --alr $alr --vdlr $vdlr --clr $clr --temp $temp --clip $grad_clip --ws 1>&- 2>&- &
			    done
			    done
		    done
		done
	done
done
#done