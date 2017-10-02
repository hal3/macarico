#!/bin/bash

export PYTHONPATH=..:$PYTHONPATH

# if python fails, fail on tee
set -o pipefail

if [[ "$#" -gt "0" ]] ; then
    commands=$@
else
    commands=`ls test_*.py | sort`
fi

for test_prog in $commands ; do
    echo "###############################################"
    echo "## Running $test_prog"
    echo "###############################################"
    echo ""
    
    ( python -u $test_prog 2>&1 ) | tee .current_output
    if [[ "$?" -gt "0" ]] ; then
	echo ""
	echo "Failure on $test_prog."
	exit 1
    fi
    
    echo ""
    echo ""
    if [[ -e "output/$test_prog.output" ]] ; then
	diff .current_output output/$test_prog.output >& .current_output.diff
	if [[ "$?" -gt "0" ]] ; then
	    echo "###############################################"
	    echo "## Diff on $test_prog"
	    echo "###############################################"
	    echo ""
	    cat .current_output.diff
	    echo ""
	    echo "Failure on $test_prog."
	    exit 1
	fi
    else
	echo "no previous output exists in output/$test_prog.output; copying"
	cp .current_output output/$test_prog.output
    fi
    echo "###############################################"
    echo "## Succeeded with $test_prog"
    echo "###############################################"
    echo ""
    echo ""
done
