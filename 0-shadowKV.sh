CUDA_VISIBLE_DEVICES=4

python test/eval_acc_debug.py \
    --model_name /remote-home/pengyichen/ShadowKV/Llama-3-8B-Instruct-Gradient-1048k \
    --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_single_3,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" \
    --num_samples -1 \
    --batch_size 1 \
    --method shadowkv \
    --sparse_budget 2048 \
    --rank 160 \
    --chunk_size 8 \