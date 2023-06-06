#!/bin/bash

MODEL_PATH='/home/michal/Desktop/GET3D/pretrained_model/shapenet_car.pt'
TRAIN_PATH='/home/michal/Desktop/GET3D/train_3d.py'
INFERENCE_PATH='/home/michal/Desktop/GET3D/save_inference_results/shapenet_car/inference'

mkdir ./all_preds
for i in {1..80}
do
python train_3d.py --outdir=save_inference_results/shapenet_car  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --seed $i --resume_pretrain $MODEL_PATH

python data_loader.py
mv $INFERENCE_PATH/mesh_pred all_preds/$i
mv $INFERENCE_PATH/fakes_000000_00.png all_preds/$i/fakes_$i.png
done
