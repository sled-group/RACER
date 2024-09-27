
MODEL=racer_ckpt
CKPTID=17
echo ${MODEL}
start=$(date +%s)
CUDA_VISIBLE_DEVICES=0 python racer/evaluation/rollout.py \
    --model-folder racer/runs/${MODEL} \
    --eval-datafolder racer/data/rlbench/test \
    --tasks all \
    --start-episode 0 \
    --eval-episodes 25 \
    --episode-length 30 \
    --log-name test \
    --model-name model_17.pth \
    --eval-log-dir racer/runs/${MODEL}/eval \
    --vlm-address http://141.212.106.177:21002  --use-vlm

end=$(date +%s)
runtime=$((end-start))
runtime_minutes=$(echo "$runtime / 60" | bc -l)
printf "Execution time: %.2f minutes\n" $runtime_minutes
