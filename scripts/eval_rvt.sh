
MODEL=rvt_ckpt
CKPTID=14
echo ${MODEL}
start=$(date +%s)
CUDA_VISIBLE_DEVICES=0 DISPLAY=:9 python racer/evaluation/rollout.py \
    --model-folder racer/runs/${MODEL} \
    --eval-datafolder racer/data/rlbench/test \
    --tasks all \
    --start-episode 0 \
    --eval-episodes 25 \
    --episode-length 30 \
    --log-name test \
    --model-name model_14.pth \
    --eval-log-dir racer/runs/${MODEL}/eval \
    --use-full-langlen

end=$(date +%s)
runtime=$((end-start))
runtime_minutes=$(echo "$runtime / 60" | bc -l)
printf "Execution time: %.2f minutes\n" $runtime_minutes
