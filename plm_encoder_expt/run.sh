#!/bin/bash
#SBATCH -A irel
#SBATCH --gres=gpu:2
#SBATCH -c 20
#SBATCH --time=4-00:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output disagreement_new.out

#Activate conda environment role_spec
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base

#If using checkpoint, pass appropriate arguments in code below, otherwise set checkpoint path to 'None' in below line
#checkpoint_path=none

#For sanity checking whole pipeline with small data, pass argument 'yes'. For full run, pass 'no'
python3 train.py \
--train_path '../data/train_new.csv' \
--val_path '../data/val_new.csv' \
--tokenizer_name_or_path 'l3cube-pune/hing-bert' \
--train_batch_size 4 \
--val_batch_size 4 \
--model_name_or_path 'l3cube-pune/hing-bert' \
--learning_rate 3e-5 \
--gpus 2 \
--max_epochs 50 \
--strategy 'ddp' \
--log_dir '/scratch/shivprasad.sagare/experiments' \
--project_name 'hinglish' \
--run_name 'plm_only_output_Disagreement_indicbert'

scp -r /scratch/shivprasad.sagare/experiments/hinglish/* ~/stuff/hinglish/plm_encoder_expt/experiments/