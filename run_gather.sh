export CUDA_VISIBLE_DEVICES=0,1,2

torchrun --nnodes=1 --nproc-per-node=3 gather.py
