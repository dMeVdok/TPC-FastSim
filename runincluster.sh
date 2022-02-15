#!/bin/bash
#SBATCH -n 1 -c 6 -G 1

python run_model_v4.py --checkpoint_name pretraining_feb15 --config models/configs/moments_ilayer_pretraining.yaml >& pretraining_feb15.txt &

echo "Waiting for individual processes to finish"
wait

echo "Done"