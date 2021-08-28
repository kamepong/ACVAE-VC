#!/bin/sh

# Copyright 2021 Hirokazu Kameoka
# 
# Usage:
# ./run_test_arctic_4spk.sh [-g gpu] [-e exp_name] [-c checkpoint]
# Options:
#     -g: GPU device#  
#     -e: Experiment name (e.g., "conv_exp1")
#     -c: Model checkpoint to load (0 indicates the newest model)

db_dir="/misc/raid58/kameoka.hirokazu/db/arctic/wav/test"
dataset_name="arctic_4spk"
gpu=0
checkpoint=0

while getopts "g:e:c:" opt; do
       case $opt in
              g ) gpu=$OPTARG;;
              e ) exp_name=$OPTARG;;
			  c ) checkpoint=$OPTARG;;
       esac
done

dconf_path="./dump/${dataset_name}/data_config.json"
stat_path="./dump/${dataset_name}/stat.pkl"
out_dir="./out/${dataset_name}"
model_dir="./model/${dataset_name}"
vocoder_dir="pwg/egs/arctic_4spk_flen64ms_fshift8ms/voc1"

python convert.py -g ${gpu} \
	--input ${db_dir} \
	--dataconf ${dconf_path} \
	--stat ${stat_path} \
	--out ${out_dir} \
	--model_rootdir ${model_dir} \
	--experiment_name ${exp_name} \
	--voc_dir ${vocoder_dir} \
	--checkpoint ${checkpoint}