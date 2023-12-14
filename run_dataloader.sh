export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc-per-node=2 dataloader.py
