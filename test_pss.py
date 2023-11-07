import logging
import sys
from datasets import load_dataset
import torch,time
from tqdm import tqdm
#from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
#from transformers.generation import GenerationConfig


device = torch.device('cuda')
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
tokenizer = AutoTokenizer.from_pretrained('/data/zsp/nips_zsp/Qwen14b_12_4', padding_side = "left", trust_remote_code=True)

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

# tokenizer.pad_token = tokenizer.eos_token
# ID_PAD = 151643
# ID_EOS = 151643  # endoftext
# tokenizer.pad_token_id = ID_PAD
# tokenizer.eos_token_id = ID_EOS
# tokenizer.padding_side = "right"  # NO use attention-mask

min_target_length = 1
ignore_pad_token_for_loss = True
padding  = "max_length"
prefix = ""
max_source_length = 768
max_target_length = 1024

def preprocess_function_test(examples):
    text = examples.data['text']
    inputs = []
    targets = []
    for i in range(len(text)):
        # input,target = text[i].split("\n### Response:")
        input, target = text[i].split("\n### xxx###")
        input += '\n### Response:'
        inputs.append(input)
        targets.append(target)
    model_inputs = tokenizer(inputs)

    # Setup the tokenizer for targets
    # with tokenizer.as_target_tokenizer():
    #     labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
    #
    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# test_dataset = load_dataset("json",data_files="./data_test_1000.json",split='train')
test_dataset = load_dataset("json",data_files="/data/xsd/data/dev/dev.json",split='train')
test_dataset = test_dataset.map(
              preprocess_function_test,
              batched=True,
              num_proc=None,
              load_from_cache_file=None,
              desc="Running tokenizer on validation dataset",
          )

test_batch_size = 1
test_batches = len(test_dataset) // test_batch_size + 1
predictions = []
with torch.no_grad():
    for i in tqdm(range(test_batches)):
        start = i * test_batch_size
        end = min((i + 1) * test_batch_size, len(test_dataset))
        output_preds = model.generate(
            input_ids = torch.LongTensor(test_dataset["input_ids"][start: end]).to(model.device),
            num_beams=1, do_sample=True, top_k=0,top_p=0.8,
            max_new_tokens = 50,
            repetition_penalty=1.1
            # attention_mask = torch.LongTensor(test_dataset["attention_mask"][start: end]).to(device),
            # max_length=50, num_beams=1, do_sample=True, top_k=10, temperature=0.7
        )
        output_preds = tokenizer.decode(
                output_preds.cpu()[0], skip_special_tokens=True
            )
        print(output_preds)
        predictions += output_preds

predictions = [pred.strip() for pred in predictions]

test_df = pd.read_csv(test_file)
references = test_df['response'].tolist()

emotion_dict = {'amusement' : 0,
 'anger' : 1,
 'awe' : 2,
 'contentment' : 3,
 'disgust' : 4,
 'excitement' : 5,
 'fear' : 6,
 'sadness' : 7,
 'neutral' : 8}

references_emo = []
references_expl = []
for i, ref in enumerate(references):
    emo = ref.split()[0]
    expl = " ".join(ref.split()[2:])
    references_emo.append(int(emotion_dict[emo.strip()]))
    references_expl.append(expl)

predictions_emo = []
predictions_expl = []
for i, ref in enumerate(predictions):
    if(predictions[i]):
        emo = ref.split()[0]
        expl = " ".join(ref.split()[2:])
        if(emo.strip() not in emotion_dict):
            predictions_emo.append(int((references_emo[i] + 1) % len(emotion_dict)))
        else:
            predictions_emo.append(int(emotion_dict[emo.strip()]))
        predictions_expl.append(expl)
    else:
        predictions_emo.append(int((references_emo[i] + 1) % len(emotion_dict)))
        predictions_expl.append('')

precision, recall, f1, _ = precision_recall_fscore_support(references_emo, predictions_emo, average='weighted')
acc = accuracy_score(references_emo, predictions_emo) * 100
f1 = f1 * 100

print("Accuracy {} and F1 {} ".format(acc, f1))

bleu = evaluate.load("bleu")
bleu_results = bleu.compute(predictions=predictions_expl, references=references_expl, tokenizer=word_tokenize)
print("BLEU scores: {} ".format(bleu_results))

meteor = evaluate.load("meteor")
meteor_results = meteor.compute(predictions=predictions_expl, references=references_expl)
print("Meteor scores: {} ".format(meteor_results))

rouge = evaluate.load("rouge")
rouge_results = rouge.compute(predictions=predictions_expl, references=references_expl)
print("ROUGE scores: {} ".format(rouge_results))

bertscore = evaluate.load("bertscore")
bertscore_results = bertscore.compute(predictions=predictions_expl, references=references_expl, lang="en")
bertscore_results = sum(bertscore_results['recall']) / len(predictions)
print("BERTScore: {} ".format(bertscore_results))

bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
bart_scorer.load(path='bart_score.pth')
bartscore_results = bart_scorer.score(predictions_expl, references_expl, batch_size=4)
bartscore_results = sum(bartscore_results) / len(bartscore_results)
print("BARTScore: {} ".format(bartscore_results))

save_emo_expla = os.path.join(output_dir, 'weights', savename, task, str(numepochs),
    str(max_source_length) + '_' + str(max_target_length), 'emo_expla.txt')
with open(save_emo_expla, 'w', encoding='utf-8') as f:
    for sen in predictions:
        f.write("{}\n".format(sen))

all_metrics = {}
all_metrics['accuracy'] = acc
all_metrics['f1-weighted'] = f1
all_metrics['bleu-1'] = bleu_results['precisions'][0]
all_metrics['bleu-2'] = bleu_results['precisions'][1]
all_metrics['bleu-3'] = bleu_results['precisions'][2]
all_metrics['bleu-4'] = bleu_results['precisions'][3]
all_metrics['avg-bleu'] = bleu_results['bleu']
all_metrics['rouge'] = rouge_results['rougeL']
all_metrics['meteor'] = meteor_results['meteor']
all_metrics['bert-score'] = bertscore_results
all_metrics['bart-score'] = bartscore_results

save_res_file = os.path.join(output_dir, 'weights', savename, task, str(numepochs),
    str(max_source_length) + '_' + str(max_target_length), 'metrics.json')

with open(save_res_file, 'w') as f:
    json.dump(all_metrics, f)

