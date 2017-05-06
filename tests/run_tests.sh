#!/bin/bash

for test_prog in test_*.py ; do
    echo "###############################################"
    echo "## Running $test_prog"
    echo "###############################################"
    echo ""
    ( python -u $test_prog 2>&1 ) | tee .current_output
    echo ""
    echo ""
    if [[ -e "output/$test_prog.output" ]] ; then
	diff .current_output output/$test_prog.output
	if [[ "$?" -gt 0 ]] ; then
	    echo "Failure."
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
