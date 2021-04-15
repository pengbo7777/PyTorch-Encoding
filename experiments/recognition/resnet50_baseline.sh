# baseline
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset minc --model deepten_resnet50_minc --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 256

python train.py --dataset minc --model deepten_resnet50_minc --lr-scheduler cos --epochs 120 --checkname resnet50_check --lr 0.025 --batch-size 64 --gpu 0,1,2,3

# rectify
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt --lr 0.1 --batch-size 256 --rectify

# warmup
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_warm --lr 0.1 --batch-size 256 --warmup-epochs 5 --rectify 

# no-bn-wd
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_nobnwd_warm --lr 0.1 --batch-size 256 --no-bn-wd --warmup-epochs 5 --rectify 

# LS
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_ls --lr 0.1 --batch-size 256 --label-smoothing 0.1 --rectify

# Mixup + LS
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 200 --checkname resnet50_rt_ls_mixup --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2 --rectify

# last-gamma
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 120 --checkname resnet50_rt_gamma --lr 0.1 --batch-size 256 --last-gamma  --rectify

# BoTs
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 200 --checkname resnet50_rt_bots --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2 --last-gamma --no-bn-wd --warmup-epochs 5 --rectify

# resnet50d
python train_dist.py --dataset imagenet --model resnet50d --lr-scheduler cos --epochs 200 --checkname resnet50d_rt_bots --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2 --last-gamma --no-bn-wd --warmup-epochs 5 --rectify

# dropblock
python train_dist.py --dataset imagenet --model resnet50 --lr-scheduler cos --epochs 200 --checkname  --label-smoothing 0.1 --mixup 0.2 --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2  --dropblock-prob 0.1 --rectify

# resnest50
python train_dist.py --dataset imagenet --model resnest50 --lr-scheduler cos --epochs 270 --checkname resnest50_rt_bots --lr 0.1 --batch-size 256 --label-smoothing 0.1 --mixup 0.2  --last-gamma --no-bn-wd --warmup-epochs 5 --dropblock-prob 0.1 --rectify



CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py --dataset minc --model deepten_resnet50_minc --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.025 --batch-size 256 >deepten03091950.out 2>&1 &


CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python train.py --dataset minc --model deepten_resnet50_minc --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 256 >03111747.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python train.py --dataset minc --model deepten_resnet50_minc --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 64 > 03151747.out 2>&1 &

 CUDA_VISIBLE_DEVICES=2,3  python train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 64 --crop 352 > 03161933_deepten_crop352_valid1.out &


CUDA_VISIBLE_DEVICES=0,1 nohup python /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 128 --crop 224 > 0326_Musten_crop224_valid1.out &
CUDA_VISIBLE_DEVICES=0,1 python /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 128
CUDA_VISIBLE_DEVICES=0,1 nohup python /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 64 > 0326_Musten_crop224_valid1.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 128 >> 03291043_Musten_crop224_valid1.out &


CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 128 >> 03291043_Musten_crop224_valid1.out &


CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname seten_7sum --lr 0.001 --batch-size 64 > /workspace/experiments/results/0407patch7_sum_crop224_valid1.out &
CUDA_VISIBLE_DEVICES=2,3 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 8 > /workspace/experiments/results/03311139patch7_sum_crop224_valid1.out &
CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model att_patchnet --lr-scheduler poly --checkname resnet50_check --lr 0.001 --batch-size 64 > /workspace/experiments/results/04091655patchslide_sum_crop224_valid1.out &
CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model att_patchnet --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.002 --batch-size 64 > /workspace/experiments/results/03301705patch7_sum_crop224_valid1.out &


CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model deepten_resnet50_minc --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 64 > /workspace/experiments/results/test.out &

CUDA_VISIBLE_DEVICES=2,3 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model att_patchnet --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 64 > /workspace/experiments/results/04011811att_patch_crop224_valid1.out &


CUDA_VISIBLE_DEVICES=0,1 nohup python /workspace/experiments/recognition/train_triple.py --dataset minc --model deepten_triplet --lr-scheduler poly --checkname resnet50_check --lr 0.025 --batch-size 64  > /workspace/encoding/data/0325minc_triplet_crop224_valid1.out 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 python train_triple.py --dataset minc --model deepten_triplet --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.025 --batch-size 128


CUDA_VISIBLE_DEVICES=0,1 python /workspace/experiments/recognition/train.py --dataset minc --model seten --lr-scheduler poly --epochs 5000 --checkname resnet50_check --lr 0.001 --batch-size 8


CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model att_patchnet --lr-scheduler poly --checkname 0412deepencoder_64_128_crop224_valid1 --lr 0.001 --batch-size 64 > /workspace/experiments/results/0412deepencoder_64_128_crop224_valid1.out &


/home/pengbo/results/runs/minc/deepten_resnet50_minc/resnet50_check/
CUDA_VISIBLE_DEVICES=0,1 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model att_patchnet --lr-scheduler poly --checkname 0412deepencoder_64_128_crop224_valid1 --lr 0.001 --batch-size 64 --epochs 600 > /workspace/experiments/results/0413deepencoder_64_128_crop224_valid1.out &


CUDA_VISIBLE_DEVICES=2 nohup python -u /workspace/experiments/recognition/train.py --dataset minc --model att_patchnet --lr-scheduler step --checkname 0414deepencoder_64_128_crop224_valid1 --lr 0.001 --batch-size 32 --epochs 600 > /workspace/experiments/results/0414deepencoder_1h1d_64_128_crop224_valid1.out &
