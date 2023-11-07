CUDA_VISIBLE_DEVICES=2 python SFT.py \
    --base_model "Qwen/Qwen-14B" \
    --output_dir "ckpt_save/exp5" \
    --dataset "dataset/prompt5/data_train_20000.json" \
    --epoch 3 \
    --batch 1

CUDA_VISIBLE_DEVICES=2 python test.py \
    --base_model "Qwen/Qwen-14B" \
    --submit_dir "submit/exp5_2w_3epoch" \
    --adapter_name_or_path "ckpt_save/exp5/final_checkpoint" \
    --dev_dataset "/data/xsd/data/dev/dev_5.json"