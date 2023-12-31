CUDA_VISIBLE_DEVICES=3 python SFT.py \
    --base_model "/data/zsp/nips_zsp/Qwen-14B/qwen-14b" \
    --output_dir "ckpt_save/exp4" \
    --dataset "dataset/prompt3/data_train_10000.json" \
    --epoch 3 \
    --batch 1

CUDA_VISIBLE_DEVICES=3 python test.py \
    --base_model "/data/zsp/nips_zsp/Qwen-14B/qwen-14b" \
    --submit_dir "submit/exp4_1w_3epoch" \
    --adapter_name_or_path "ckpt_save/exp4/final_checkpoint" \
    --dev_dataset "/data/xsd/data/dev/dev_3.json"