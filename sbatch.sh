#!/bin/bash

# Usage: ./sbatch.sh SCRIPTNAME

SCRIPTNAME="$1"

if [ -z "$SCRIPTNAME" ]; then
    echo "Usage: $0 SCRIPTNAME"
    exit 1
fi

ssh gpucluster2.doc.ic.ac.uk "sbatch /vol/bitbucket/al1624/FIP/albus-bayesian-moe-router/scripts/bash/$SCRIPTNAME.sh"