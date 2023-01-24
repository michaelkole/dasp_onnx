from datasets import load_dataset
from transformers import AutoModelWithHeads, AutoTokenizer
import onnxruntime


import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from multiprocess import Process

from huggingface_hub import hf_hub_download


#helper funcs
# Load needed skills by skilltype (span-extraction, multiple-choice, categorical, abstractive)
def load_skills(skill_type, path="square_skills/impl_skills.csv"):
    all_skills = pd.read_csv(path)
    skills = all_skills[all_skills["Type"] == skill_type]
    return skills

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

# Inf functions
def base_model_inference(model, tokenizer, question, context):
    inputs = tokenizer(question, context, padding=True, truncation=True, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits, axis=1).numpy()[0]
    end_idx = (torch.argmax(outputs.end_logits, axis=1) + 1).numpy()[0]
    return tokenizer.decode(inputs['input_ids'][0, start_idx:end_idx])


def onnx_inference(onnx_model, tokenizer, question, context):
    inputs = tokenizer(question, context, padding=True, truncation=True, return_tensors="np")
    preped_inputs = {key: np.array(inputs[key], dtype=np.int64) for key in inputs}
    outputs = onnx_model.run(input_feed=dict(preped_inputs), output_names=None)

    start_scores = outputs[0]
    end_scores = outputs[1]
    ans_start = np.argmax(start_scores)
    ans_end = np.argmax(end_scores)+1

    return tokenizer.decode(inputs['input_ids'][0, ans_start:ans_end])


skill =  "span-extraction"
skills_df = load_skills(skill)

def run_inf(
        data, modelname, run_func, model, tokenizer, adapter
    ):
    
    df = pd.DataFrame(columns=[
            "skill", "reader", "adapter", "modelname",
            "timestamp", 
            "answer", 
            "data_id", "dataset", 
            "question", "context", "answer_dataset"
        ])
    
    if adapter == "drop":
        context_name = "passage"
        id_name = "query_id"
        answers_name = "answers_spans"
    else:
        context_name = "context"
        id_name = "id"
        answers_name = "answers"

    examples = list(zip(data["question"], data[context_name]))
    
    for data_id in tqdm(range(len(examples))):
        example = data[data_id]        
        
        example_id = example[id_name]
        question = example["question"]
        context = example[context_name]
        answer_dataset = example[answers_name]
        
        prediction = run_func(model, tokenizer, question, context)
        
        data_set_name = adapter
        df.loc[len(df)] = [
            skill, reader, adapter, modelname,
            pd.Timestamp.now(),
            prediction, 
            example_id, data_set_name, 
            question, context[:90], answer_dataset
        ]
    print(f"Finished {adapter}_{reader}_{modelname}")
    save_df(df, f"temp/{skill}/{adapter}_{reader}_{modelname}.csv")


example_amount = 10

skipping_adapters = ["newsqa", "hotpotqa"] 
for adapter in skills_df["Reader Adapter"].unique():
    if adapter in skipping_adapters:
        print(f"Skipping {adapter}")
        continue
    
    adapter_df = skills_df[skills_df["Reader Adapter"] == adapter]

    #load adapter specific dataset
    data_set_name = adapter
    if example_amount == 0:
        data = load_dataset(data_set_name, split=f"validation")
    else: 
        data = load_dataset(data_set_name, split=f"validation[:{example_amount}]")
    
    print(f"Loaded and preped dataset: {data_set_name} with {len(data)} example questions")

    # load models
    for reader in adapter_df["Reader Model"].unique():
        print(f"Loading: {reader} {adapter} models")
        
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
        print(f"Loaded: {reader} {adapter} models")
        
        print("Starting inf.")
        base_p = Process(target=run_inf, args=(data, "base", base_model_inference, base_model, tokenizer, adapter))
        quant_base_p = Process(target=run_inf, args=(data, "quant_base", base_model_inference, quantized_base_model, tokenizer, adapter))
        onnx_p = Process(target=run_inf, args=(data, "onnx", onnx_inference, onnx_model, tokenizer, adapter))
        onnx_opt_p = Process(target=run_inf, args=(data, "onnx_opt", onnx_inference, onnx_model_opt, tokenizer, adapter))
        quant_onnx_p = Process(target=run_inf, args=(data, "quant_onnx", onnx_inference, onnx_model_quant, tokenizer, adapter))
        quant_onnx_opt_p = Process(target=run_inf, args=(data, "quant_onnx_opt", onnx_inference, onnx_model_quant_opt, tokenizer, adapter))
    
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