CUDA_VISIBLE_DEVICES=0,1 python racer/gradio_demo/run.py \
    --lm-address http://141.212.110.118:8000/encode/ \
    --vlm-address http://141.212.110.118:21002 --unseen_task

# change the IP address to your own address