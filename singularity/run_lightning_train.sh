#!/bin/bash
cd /work/fgiuliari/Puzzle-Diffusion/singularity

NAME='Diff'
ARGS='-gpus 4 -batch_size 32 -steps 600'


qsub -v pyfile=lightning_main.py,args="$ARGS" -N "$NAME"  pbs_args.sh

