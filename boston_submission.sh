#!/bin/bash

##      Script is submitted to this Queue:
#PBS -q wsuq

##      node:  asx1-42
##	CPU: 6282SE
#PBS -l select=1:ncpus=32:mem=32GB:cpu_model=E5-4627v4

##      Commands to be executed:
cd $TMPDIR
cp -R /wsu/home/aj/aj76/aj7682/boston/. $TMPDIR

module load python-3.4.3
python3.4 boston_XGB.py

cp -R $TMPDIR/. /wsu/home/aj/aj76/aj7682/boston/
