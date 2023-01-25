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
skill =  "categorical"
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

def categorical_base_inference(model, tokenizer, question, context):
    
    raw_input = [[context, question]]
    inputs = tokenizer(raw_input, padding=True, truncation=True, return_tensors="pt")
    
    outputs = model(**inputs)
    answer_idx = torch.argmax(outputs.logits)

    return bool(answer_idx), outputs.logits[0].detach().numpy()

def categorical_onnx_inference(onnx_model, tokenizer, question, context):

    inputs = tokenizer(question, context, padding=True, truncation=True, return_tensors="np")
    inputs = {key: np.array(inputs[key], dtype=np.int64) for key in inputs}

    outputs = onnx_model.run(input_feed=dict(inputs), output_names=None)

    return bool(np.argmax(outputs[0][0])), outputs[0][0]

def run_inf(
        data, modelname, run_func, input_model, tokenizer,

    ):    
    df = pd.DataFrame(columns=[
        "skill", "reader", "adapter", "modelname",
        "timestamp", 
        "answer", "logits_answer",
        "data_id", "dataset", "question", "context", "answer_dataset"
    ])

    for data_id in tqdm(range(len(data))):
        
        question = data[data_id]["question"]
        context = data[data_id]["passage"]
        answer_dataset = data[data_id]["answer"]
        
        answer, answer_logits = run_func(input_model, tokenizer, question, context)   
        data_set_name = adapter

        df.loc[len(df)] = [
            skill, reader, adapter, modelname,
            pd.Timestamp.now(),
            answer, answer_logits,
            data_id, data_set_name, question, context[:90], answer_dataset
        ]
    
    save_df(df, f"temp/{skill}/{adapter}_{reader}_{modelname}.csv")
    
example_amount = 0

skipping_adapters = [] 
for adapter in skills_df["Reader Adapter"].unique():

    if adapter in skipping_adapters:
        print(f"Skipping {adapter}")
        continue
    adapter_df = skills_df[skills_df["Reader Adapter"] == adapter]
    # load dataset
    data_set_name = adapter
    if example_amount == 0:
        data = load_dataset(data_set_name, split="validation")
    else: 
        data = load_dataset(data_set_name, split=f"validation[:{example_amount}]")
    
    print(f"Loaded and preped dataset: {data_set_name} with {len(data)} example questions")

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

        base_p = Process(target=run_inf, args=(data, "base", categorical_base_inference, base_model, tokenizer))
        quant_base_p = Process(target=run_inf, args=(data, "quant_base", categorical_base_inference, quantized_base_model, tokenizer))
        onnx_p = Process(target=run_inf, args=(data, "onnx", categorical_onnx_inference, onnx_model, tokenizer))
        onnx_opt_p = Process(target=run_inf, args=(data, "onnx_opt", categorical_onnx_inference, onnx_model_opt, tokenizer))
        quant_onnx_p = Process(target=run_inf, args=(data, "quant_onnx", categorical_onnx_inference, onnx_model_quant, tokenizer))
        quant_onnx_opt_p = Process(target=run_inf, args=(data, "quant_onnx_opt", categorical_onnx_inference, onnx_model_quant_opt, tokenizer))
    
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