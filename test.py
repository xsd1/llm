import logging
import re
import argparse,os,yaml
from datasets import load_dataset
import torch,time
from tqdm import tqdm
import json
import pandas as pd
import pickle as pkl
#from huggingface_hub import login
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel
#from transformers.generation import GenerationConfig

def main(config):
    device = torch.device('cuda')
    logger = logging.getLogger(__name__)
    # Configure the logging module
    logging.basicConfig(level=logging.INFO)

    #login(token="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")S

    base_model_name = config['base_model']
    adapter_name_or_path = config['adapter_name_or_path']

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

    tokenizer.pad_token = tokenizer.eos_token
    ID_PAD = 151643
    ID_EOS = 151643  # endoftext
    tokenizer.pad_token_id = ID_PAD
    tokenizer.eos_token_id = ID_EOS
    tokenizer.padding_side = "right"  # NO use attention-mask

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
    test_dataset = load_dataset("json",data_files=config['dev_dataset'],split='train')
    test_dataset = test_dataset.map(
                preprocess_function_test,
                batched=True,
                num_proc=None,
                load_from_cache_file=None,
                desc="Running tokenizer on validation dataset",
            )

    test_batch_size = 1
    test_batches = len(test_dataset) // test_batch_size
    predictions = []

    with open('data/dev/dev_set.pickle', 'rb') as f:
        dev_set = pkl.load(f)

    with torch.no_grad():
        for i in tqdm(range(test_batches)):
            start = i * test_batch_size
            end = min((i + 1) * test_batch_size, len(test_dataset))
            output_preds = model.generate(
                # input_ids = torch.LongTensor(test_dataset["input_ids"][start: end]).to(device),
                # attention_mask = torch.LongTensor(test_dataset["attention_mask"][start: end]).to(device),
                # max_length=max_target_length,
                # max_new_tokens=64,
                # num_beams=1
                input_ids = torch.LongTensor(test_dataset["input_ids"][start: end]).to(model.device),
                num_beams=1, do_sample=True, top_k=10, temperature=0.7,
                max_new_tokens = 50,
            )
            output_preds = tokenizer.decode(
                    output_preds.cpu()[0], skip_special_tokens=True
                )
            # output_preds = output_preds[0].strip()
            # predictions += output_preds
            # print(output_preds)
            with open(f"{config['submit_dir']}/pred.json", 'a') as f:
                if i != len(test_dataset):
                    json.dump({'pred': output_preds, 'dialog_id': dev_set[i]['dialog_id']}, f)
                    f.write(',')
            with open(f"{config['submit_dir']}/submit.json", 'a') as f:
                matches = re.findall(r"Response: Emotion: (\w+) Explanation: (.+?)(?:###|$)", output_preds)
                
                if matches:
                    match = matches[0]  
                    emo = match[0]
                    expl = match[1]
                else:
                    emo = 'something else'
                    expl = "i don't find the image interesting."
                emo = 'something else' if emo.strip() == 'neutral' else emo.strip()
                expl = expl.strip()
                if i != len(test_dataset):
                    json.dump({'dialog_id': dev_set[i]['dialog_id'], 'predicted_emotion': emo, 'generated_explanation': expl}, f)
                    f.write(',')
    exit(0)

    predictions = [pred.strip() for pred in predictions]

    # test_df = pd.read_csv(test_file)
    # references = test_df['response'].tolist()

    emotion_dict = {'amusement' : 0,
    'anger' : 1,
    'awe' : 2,
    'contentment' : 3,
    'disgust' : 4,
    'excitement' : 5,
    'fear' : 6,
    'sadness' : 7,
    'neutral' : 8}

    # references_emo = []
    # references_expl = []
    # for i, ref in enumerate(references):
    #     emo = ref.split()[0]
    #     expl = " ".join(ref.split()[2:])
    #     references_emo.append(int(emotion_dict[emo.strip()]))
    #     references_expl.append(expl)


    result = [{'pred': pred, 'dialog_id': dev['dialog_id']} for pred, dev in zip(predictions, dev_set)]
    with open('submit/result.json', 'w') as f:
        json.dump(result, f)
    result = []
    predictions_emo = []
    predictions_expl = []
    predictions = predictions[:len(test_dataset)]
    for i, ref in enumerate(predictions):
        if(predictions[i]):
            if 'Output' not in ref:
                emo = 'something else'
                expl = "i don't find the image interesting."
            else:
                ref = ref.split('Output', 1)[1]
                matches = re.findall(r"### The questioner feels (\w+) because (.+)", ref)
                if matches:
                    match = matches[0]  
                    emo = match[0]
                    expl = match[1]
                else:
                    emo = 'something else'
                    expl = "i don't find the image interesting."
            emo = 'something else' if emo.strip() == 'neutral' else emo.strip()
            expl = expl.strip()
            result.append({'dialog_id': dev_set[i]['dialog_id'], 'predicted_emotion': emo, 'generated_explanation': expl})
        #     if(emo.strip() not in emotion_dict):
        #         predictions_emo.append(int((references_emo[i] + 1) % len(emotion_dict)))
        #     else:
        #         predictions_emo.append(int(emotion_dict[emo.strip()]))
        #     predictions_expl.append(expl)
        # else:
        #     predictions_emo.append(int((references_emo[i] + 1) % len(emotion_dict)))
        #     predictions_expl.append('')

    exit(0)
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SFT Qwen')

    parser.add_argument('--base_model', type=str, help='path to base model')
    parser.add_argument('--submit_dir', type=str, help='path to result')
    parser.add_argument('--dev_dataset', type=str, help='path to dev_dataset')
    parser.add_argument('--adapter_name_or_path', type=str, help='path to adapter')

    args = parser.parse_args()

    # 构建参数字典
    config = args.__dict__
    
    os.makedirs(args.submit_dir, exist_ok=True)

    # 将参数保存为YAML文件
    with open(os.path.join(args.submit_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    main(config)