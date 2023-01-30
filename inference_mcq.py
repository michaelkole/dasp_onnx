from datasets import load_dataset
from transformers import AutoModelWithHeads, AutoTokenizer
from huggingface_hub import hf_hub_download
import onnxruntime
import torch
from tqdm import tqdm
from multiprocess import Process
import numpy as np
import pandas as pd
import os

# Load needed skills by skilltype (span-extraction, multiple-choice, categorical, abstractive)
def load_skills(skill_type, path="square_skills/impl_skills.csv"):
    all_skills = pd.read_csv(path)
    skills = all_skills[all_skills["Type"] == skill_type]
    return skills

#Choose Skill
skill =  "multiple-choice"
skills_df = load_skills(skill)


def load_onnx_model(model_onnx, model_onnx_quant, as_list=False):
    onnx_model = onnxruntime.InferenceSession(model_onnx, providers=["CPUExecutionProvider"])
    onnx_model_quant = onnxruntime.InferenceSession(model_onnx_quant, providers=["CPUExecutionProvider"])
    
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    onnx_model_opt = onnxruntime.InferenceSession(model_onnx, so)
    onnx_model_quant_opt = onnxruntime.InferenceSession(model_onnx_quant, so)
    
    if as_list:
        return [onnx_model, onnx_model_opt, onnx_model_quant, onnx_model_quant_opt]
    return onnx_model, onnx_model_opt, onnx_model_quant, onnx_model_quant_opt

def repo_builder(reader, adapter):
    repo_id = f"UKP-SQuARE/{reader}-pf-{adapter}-onnx"
    filename_onnx = "model.onnx"
    filename_onnx_quant = "model_quant.onnx"

    model_onnx = hf_hub_download(repo_id=repo_id, filename=filename_onnx)
    model_onnx_quant = hf_hub_download(repo_id=repo_id, filename=filename_onnx_quant)

    return model_onnx, model_onnx_quant

def save_df(df_new, path_to_logger_file = "logs/logger_all.csv"):
    if os.path.exists(path_to_logger_file):
        df_fin = pd.concat([pd.read_csv(path_to_logger_file), df_new])
        df_fin.to_csv(path_to_logger_file,index=False)
    else: 
        df_new.to_csv(path_to_logger_file,index=False)

def mc_base_inference(model, tokenizer, question, context, choices):
    outputs = []
    raw_input = [[context, question + " " + choice] for choice in choices]
    inputs = tokenizer(raw_input, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    answer_logits = outputs.logits
    answer_idx = torch.argmax(answer_logits)
    answer = choices[answer_idx]

    return answer, answer_logits

def mc_onnx_inference(onnx_model, tokenizer, context, question, choices):

    raw_input = [[context, question + " " + choice] for choice in choices]
    inputs = tokenizer(raw_input, padding=True, truncation=True, return_tensors="np")

    inputs['input_ids'] =  np.expand_dims(inputs['input_ids'], axis=0)
    inputs['attention_mask'] =  np.expand_dims(inputs['attention_mask'], axis=0)
    if "token_type_ids" in inputs: #roberta does not use this
        inputs['token_type_ids'] = np.expand_dims(inputs['token_type_ids'], axis=0)
    
    outputs = onnx_model.run(input_feed=dict(inputs), output_names=None)

    answer_logits = outputs[0]
    answer_idx = np.argmax(answer_logits)
    answer = choices[answer_idx]
    
    return answer, outputs[0]

def run_inf(
        preped_data_set, modelname, run_func, input_model, tokenizer,

    ):    
    df = pd.DataFrame(columns=[
        "skill", "reader", "adapter", "modelname",
        "timestamp", 
        "answer", "logits_answer",
        "data_id", "dataset", "question", "context", "choices", "answer_dataset"
    ])

    for data_id in tqdm(range(len(preped_data_set))):
        
        example_id = preped_data_set[data_id][0]
        question = preped_data_set[data_id][1]
        context = preped_data_set[data_id][2]
        choices = preped_data_set[data_id][3]
        answer_dataset = preped_data_set[data_id][4]

        answer, answer_logits = run_func(input_model, tokenizer, question, context, choices)   
        data_set_name = adapter

        df.loc[len(df)] = [
            skill, reader, adapter, modelname,
            pd.Timestamp.now(),
            answer, answer_logits,
            example_id, data_set_name, question, context[:90], choices, answer_dataset
        ]
    
    save_df(df, f"temp/{skill}/{adapter}_{reader}_{modelname}.csv")
    

def load_and_prep_dataset(data_set_name, example_amount=0):
    if example_amount == 0:
        print(f"Loading all example data of {data_set_name} dataset")
        split_size = f"validation" #loading complete dataset
    else:
        print(f"Loading just {example_amount} example of {data_set_name} dataset")
        split_size = f"validation[:{example_amount}]" #loading only a part of the dataset
        
    preped_data_set = []
    print("Now laoding dataset.")

    if data_set_name in ["cosmos_qa", "quail", "quartz"]:
        data = load_dataset(data_set_name, split=split_size)
    elif data_set_name == "race":
        data = load_dataset(data_set_name, "middle", split=split_size)
    elif data_set_name in ["multi_rc", "commonsense_qa", "social_i_qa"]: #social_i_qa not implemented
        print("Error. Not implemented data_set. Don't know how to build preped_data_set.")
        return False
    else: 
        print("Error. Not implemented data_set. Cant load dataset.")
        return False
    
    print(f"Loaded dataset: {data_set_name}. Now preping dataset")
    
    # build preped data 
    i = 0 #helper varibale for social_i_qa dataset
    for example in data:
        if data_set_name == "cosmos_qa":
            example_id = example["id"]
            question = example["question"]
            context = example["context"]
            choices = [example["answer0"], example["answer1"], example["answer2"], example["answer3"]]
            correct_answer = choices[example["label"]]
        elif data_set_name == "quail":
            example_id = example["id"]
            question = example["question"]
            context = example["context"]
            choices = example["answers"]
            correct_answer = example["answers"][example["correct_answer_id"]]
        elif data_set_name == "quartz":
            example_id = example["id"]
            question = example["question"]
            context = example["para"]
            choices = example["choices"]["text"]
            correct_answer = example["choices"]["text"][ord(example["answerKey"])-65] # convert ASCII char to Int.
        elif data_set_name =="race":
            example_id = example["example_id"]
            question = example["question"]
            context = example["article"]
            choices = example["options"]
            correct_answer = example["options"][ord(example["answer"])-65] # convert ASCII char to Int.  
        elif data_set_name == "social_i_qa":
            example_id = i
            i+=1
            question = example["question"]
            context = example["context"]
            choices = [example["answerA"], example["answerB"], example["answerC"]]
            correct_answer = choices[int(example["label"])-1]
        else:
            print("Error. Not implemented data_set. Don't know how to build preped_data_set.")
            Exception
        
        preped_data_set.append((example_id, question, context, choices, correct_answer))
    return preped_data_set

example_amount = 0

skipping_adapters = ["race", "social_i_qa"] 
for adapter in skills_df["Reader Adapter"].unique():

    if adapter in skipping_adapters:
        print(f"Skipping {adapter}")
        continue

    adapter_df = skills_df[skills_df["Reader Adapter"] == adapter]
    # load dataset
    data_set_name = adapter
    preped_data_set = load_and_prep_dataset(data_set_name, example_amount=example_amount)
    
    if not preped_data_set:
        continue
    print(f"Loaded and preped dataset: {data_set_name} with {len(preped_data_set)} example questions")

    # load models
    for reader in adapter_df["Reader Model"].unique():
        print(f"Loading: {reader} {adapter}")
        
        #  load base model
        tokenizer = AutoTokenizer.from_pretrained(reader)
        base_model = AutoModelWithHeads.from_pretrained(reader)
        adapter_name = base_model.load_adapter(f"AdapterHub/{reader}-pf-{adapter}", source="hf")
        base_model.active_adapters = adapter_name
        
        #load and eval quant model 
        quantized_base_model = torch.quantization.quantize_dynamic(base_model, {torch.nn.Linear}, dtype=torch.qint8)

        #load onnx models
        model_onnx, model_onnx_quant = repo_builder(reader, adapter)
        onnx_model, onnx_model_opt, onnx_model_quant, onnx_model_quant_opt = load_onnx_model(model_onnx, model_onnx_quant)

        #Run infernece in parallel.

        base_p = Process(target=run_inf, args=(preped_data_set, "base", mc_base_inference, base_model, tokenizer))
        quant_base_p = Process(target=run_inf, args=(preped_data_set, "quant_base", mc_base_inference, quantized_base_model, tokenizer))
        onnx_p = Process(target=run_inf, args=(preped_data_set, "onnx", mc_onnx_inference, onnx_model, tokenizer))
        onnx_opt_p = Process(target=run_inf, args=(preped_data_set, "onnx_opt", mc_onnx_inference, onnx_model_opt, tokenizer))
        quant_onnx_p = Process(target=run_inf, args=(preped_data_set, "quant_onnx", mc_onnx_inference, onnx_model_quant, tokenizer))
        quant_onnx_opt_p = Process(target=run_inf, args=(preped_data_set, "quant_onnx_opt", mc_onnx_inference, onnx_model_quant_opt, tokenizer))
    
        base_p.start()
        quant_base_p.start()
        onnx_p.start()
        onnx_opt_p.start()
        quant_onnx_p.start()
        quant_onnx_opt_p.start()

        base_p.join()
        quant_base_p.join()
        onnx_p.join()
        onnx_opt_p.join()
        quant_onnx_p.join()
        quant_onnx_opt_p.join()