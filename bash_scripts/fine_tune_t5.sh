#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/mnt/project/kroeger/predictprotein1_solubility
python3 Model_training/train_model.py --solubility_data Data/PSI_Biology_solubility_trainset.csv --model fine_tune_lora --epochs 50 --batch_size 30 --acc_grad 2