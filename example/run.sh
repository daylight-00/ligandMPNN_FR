#!/bin/bash
#SBATCH -J fast_cycle
#SBATCH -c 64
#SBATCH -p cpu-farm
#SBATCH --array=0-7
source $HOME/miniforge3/bin/activate rfdaa
export LMPNN_DIR=$HOME/aipd/ligandMPNN
export LMPNN_FR_DIR=$HOME/aipd/ligandMPNN_FR
export PYTHONPATH=$LMPNN_DIR:$LMPNN_FR_DIR:$PYTHONPATH

python lmpnn_fr.py pdb_list.parquet $SLURM_ARRAY_TASK_ID 8
