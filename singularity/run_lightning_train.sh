#!/bin/bash
cd /work/fgiuliari/PuzzleDiffusion-GNN/singularity

NAME='Diff-GNN'
ARGS='-gpus 4 -batch_size 16 -steps 600'


qsub -v pyfile=puzzle_diff/train_diff.py,args="$ARGS" -N "$NAME"  pbs_args.sh

