import logging
import sys

import torch,time
#from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
#from transformers.generation import GenerationConfig



logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

#login(token="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")S

base_model_name ="/data/zsp/nips_zsp/Qwen-14B/qwen-14b"
adapter_name_or_path ="/data/xsd/code/emo/llm/results/exp1/final_checkpoint"

# model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     return_dict=True,
#     load_in_4bit=True,
#     device_map={"":0},
#     #fp16=True,
#     #bnb_4bit_compute_dtype=torch.float16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True
# )
tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side = "left", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    return_dict=True,
    fp16=True,
    load_in_8bit=True,
    quantization_config= BitsAndBytesConfig(load_in_8bit=True),
    device_map={"":0},
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

#model.generation_config = GenerationConfig.from_pretrained(base_model_name, trust_remote_code=True)

model = PeftModel.from_pretrained(model, adapter_name_or_path)

#model.generation_config = GenerationConfig.from_pretrained(base_model_name, trust_remote_code=True)

model.eval()

profix = "You are a responsible assistant. You are required to provide information accurately, impartially, objectively, and without bias."
prompt = profix+'''<user>:What kind of fruit do you like to eat? Can you recommend it to me? \n<assistant>:'''
encoded = tokenizer(prompt, return_tensors="pt")
encoded.pop('token_type_ids', None)

print(encoded["input_ids"][0].size(0))

t0 = time.perf_counter()
input_ids = torch.LongTensor(test_dataset["input_ids"][start: end]).to(device)
attention_mask = torch.LongTensor(test_dataset["attention_mask"][start: end]).to(device)
encoded = {k: v.to("cuda:0") for k, v in encoded.items()}
with torch.no_grad():
    outputs = model.generate(
        **encoded,
        max_new_tokens=50,
        top_k =50,
        #do_sample=False,
        temperature=0.01,
        return_dict_in_generate=True,
        output_scores=True,
    )

t = time.perf_counter() - t0

print(f"Time for inference: {t:.02f} sec total")

print(f"Memory used: {torch.cuda.max_memory_reserved(0) / 1e9:.02f} GB")

output = tokenizer.decode(outputs.sequences[0][encoded["input_ids"][0].size(0):], skip_special_tokens=True)
print(output)

