#!/bin/bash
#SBATCH -J ast_finetuned		                # name of job
#SBATCH -p dgxs 								# name of partition or queue
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100                       # request node with V100 GPU
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH -o ast_finetuned.out				    # name of output file for this submission script
#SBATCH -e ast_finetuned.err				    # name of error file for this submission script
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dilgrenc@oregonstate.edu

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load python/3.12
module load cuda/12.2

# load venv
source /nfs/hpc/share/dilgrenc/topnmusic/.venv/bin/activate

# run my job (e.g. matlab, python)
for lr in 5e-5
do
	for bs in 16
	do
   		.venv/bin/python3 ./src/topnmusic/ast_finetuned_audioset_finetuned_gtzan.py $lr $bs
    done
done

# interactive run
# srun -p dgxs --gres=gpu:1 --constraint=v100 --mem=10G --pty bash