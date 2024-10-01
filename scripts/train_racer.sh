export GPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# batch size  3->12GB   6->22GB, 12 -> 44GB

torchrun --nnodes 1 --nproc_per_node $GPUS_PER_NODE --node_rank 0 --master_addr localhost --master_port 29504 \
    racer/train.py \
    --exp_cfg_opts 'exp_id train_racer_debug bs 12 num_workers 6 epochs 18' \
    --exp_cfg_path racer/configs/rich.yaml \
    --log-dir racer/runs \
    --replay-dir-aug racer/replay_buffers/racer_replay_public \
    --mvt_cfg_opts 'proprio_dim 3 use_dropout True'