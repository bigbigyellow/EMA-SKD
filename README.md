# EMA-SKD
Implementation of EMA-SKD

how to use:

```shell
CUDA_VISIBLE_DEVICES=0,1 python main.py --classifier_type ResNet50 --experiments_dir main_experiment_imagenet_50 --workers 8 --experiment_type imagenet_Res50 --weight 0.5 --weight2 0.5 --beta 0.5 --seed 1 --EHSKD --batch_size 128 --lr_decay_schedule 30 60 90 --start_epoch 0 --end_epoch 100 --data_path /home/liu/ssd/datasets/imagenet --data_type imagenet --rank 0 --multiprocessing_distributed --dist_url tcp://127.0.0.1:10000 --saveckp_freq 100
```
