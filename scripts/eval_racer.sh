
MODEL=racer_ckpt
CKPTID=17
echo ${MODEL}
start=$(date +%s)
CUDA_VISIBLE_DEVICES=1 python racer/evaluation/rollout.py \
    --model-folder racer/runs/${MODEL} \
    --eval-datafolder racer/data/rlbench/test \
    --tasks all \
    --start-episode 0 \
    --eval-episodes 25 \
    --episode-length 30 \
    --log-name test-llava-0928 \
    --model-name model_17.pth \
    --eval-log-dir racer/runs/${MODEL}/eval \
    --lm-address http://141.212.110.118:8000/encode/ \
    --vlm-address http://141.212.110.118:21002 \
    --use-vlm # comment this line out if eval with task-goal only, i.e., no instructions

end=$(date +%s)
runtime=$((end-start))
runtime_minutes=$(echo "$runtime / 60" | bc -l)
printf "Execution time: %.2f minutes\n" $runtime_minutes
