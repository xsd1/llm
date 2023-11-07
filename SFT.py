from datasets import load_dataset
import torch,einops
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PromptEncoderConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import argparse
import yaml
import os
#from transformers.generation import GenerationConfig

def main(config):
    # dataset = load_dataset("json",data_files="/data/xsd/data/train/train.json",split="train")
    dataset = load_dataset("json", data_files=config['dataset'], split="train")

    output_dir = config['output_dir']

    base_model_name = config['base_model']


    #########量化加载##########
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,#在4bit上，进行量化
        bnb_4bit_use_double_quant=True,# 嵌套量化，每个参数可以多节省0.4位
        bnb_4bit_quant_type="nf4",#NF4（normalized float）或纯FP4量化 博客说推荐NF4
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,#本地模型名称
        fp16=True,
        quantization_config=bnb_config,#上面本地模型的配置
        device_map=device_map,#使用GPU的编号
        trust_remote_code=True,
        use_auth_token=True
    )

    '''base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
    #    load_in_4bit=True,
        device_map='auto',
    )'''

    #base_model.generation_config = GenerationConfig.from_pretrained(base_model_name, trust_remote_code=True)

    base_model.config.use_cache = False
    #base_model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    ID_PAD = 151643
    ID_EOS = 151643  # endoftext
    tokenizer.pad_token_id = ID_PAD
    tokenizer.eos_token_id = ID_EOS
    tokenizer.padding_side = "right"  # NO use attention-mask
    #响应模板，即从哪开始设定label {text:'###Instruction:\n......\n\n###Input:\n......\n\n### Response:\n......'}
    response_template_with_context = "\n### Response:"  # We added context here: "\n". This is enough for this tokenizer
    response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)
    #lora
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=4,
        target_modules=["c_attn","c_proj",],
        bias="none",
        task_type="CAUSAL_LM",
    )

    '''peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        target_modules=[
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj", 
            "lm_head",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )'''
    #adalora
    '''peft_config = AdaLoraConfig(
        peft_type="ADALORA", 
        task_type="CAUSAL_LM", 
        r=64, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        )'''

    #prefix-tuning
    '''peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens = 10,
        prefix_projection=True,
        encoder_hidden_size = 1024
    )'''
    #p-tuning
    '''peft_config = PromptEncoderConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens = 20,
        encoder_hidden_size=128,
    )'''

    training_args = TrainingArguments(
        report_to="none",
        output_dir=output_dir,#训练后输出目录
        per_device_train_batch_size=config['batch'],#每个GPU的批处理数据量
        gradient_accumulation_steps=16,#在执行反向传播/更新过程之前，要累积其梯度的更新步骤数
        learning_rate=5e-5,#超参、初始学习率。太大模型不稳定，太小则模型不能收敛
        logging_steps=10,#两个日志记录之间的更新步骤数
        #max_steps=8000,#要执行的训练步骤总数
        num_train_epochs=config['epoch'],
    )
    max_seq_length = 2048
    #TrainingArguments 的参数详解：https://blog.csdn.net/qq_33293040/article/details/117376382



    trainer = SFTTrainer(
        model=base_model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        args=training_args,
    )

    trainer.train()

    output_dir2 = output_dir + "/final_checkpoint"
    trainer.model.save_pretrained(output_dir2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFT Qwen')

    parser.add_argument('--base_model', type=str, help='path to base model')
    parser.add_argument('--output_dir', type=str, help='path to output')
    parser.add_argument('--dataset', type=str, help='path to dataset')
    parser.add_argument('--epoch', type=int, help='num_train_epochs')
    parser.add_argument('--batch', type=int, default=1, help='train_batch_size')

    args = parser.parse_args()

    # 构建参数字典
    config = args.__dict__
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 将参数保存为YAML文件
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    main(config)