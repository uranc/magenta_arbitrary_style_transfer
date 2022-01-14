#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate adain2
export TF_XLA_FLAGS=--tf_xla_cpu_global_jit
export CUDA_VISIBLE_DEVICES=-1

python /mnt/hpx/projects/MWNaturalPredict/DL/magenta/magenta/models/arbitrary_image_stylization/arbitrary_image_stylization_with_weights.py --checkpoint=models/model.ckpt --output_dir=outputs --style_images_paths=$1 --content_images_paths=$2 --image_size=2048 --content_square_crop=False --style_image_size=256 --style_square_crop=False --logtostderr
exit 0


