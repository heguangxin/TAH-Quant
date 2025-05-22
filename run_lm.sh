
# Important! Pre-trained weights need to be partioned before fine-tuning.
python3.9 convert_gpt2_checkpoint.py --model-name gpt2-xl --save-dir checkpoints


ARGS="--model-name gpt2-xl \
--tokenizer-name gpt2-xl \
--load-pretrained-model true \
--task-name wikitext --n-epochs 10 --warmup-epochs 0 \
--num-layers 6 --num-heads 25 --embedding-dim 1600 \
--num-iters 10000000 --lr 5e-6 --seq-length 1024 --batch-size 32 --micro-batch-size 1 \
--forward-compress-method tah \
--tile-size 64 \
--high-precision-bits 4 \
--low-precision-bits 3 \
--high-precision-allocation-ratio 0.8 \
--forward-bits 4 \
--backward-compress-method fixpoint \
--backward-tile-size 32 \
--backward-bits 6 \
--dist-url tcp://127.0.0.1:9033 \
--world-size 8 --pipeline-group-size 8 \
--pp-mode gpipe --profiling no-profiling --do-evaluation false"

(trap 'kill 0' SIGINT; \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python3.9 dist_lm_runner.py $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)

> /dev/null 2>&1 &
