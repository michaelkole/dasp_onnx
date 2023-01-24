from datasets import load_dataset, load_metric
import evaluate
from transformers import AutoModelWithHeads, AutoTokenizer
from transformers.models.bert import BertOnnxConfig
from transformers.onnx import OnnxConfig, validate_model_outputs, export

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime import InferenceSession
import onnxruntime


import time
from typing import Tuple, Union
import torch
import numpy as np
import pandas as pd
import os
from typing import Mapping, OrderedDict
from tqdm import tqdm
from multiprocessing import Process

from huggingface_hub import hf_hub_download


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

# Inference extractive_qa models
# base model

def base_predict(
            model, input, tokenizer, preprocessing_kwargs, model_kwargs, batch_size=1, disable_gpu=True, output_features=False
    ) -> Union[dict, Tuple[dict, dict]]:
        """
        Inference on the input.
        Args:
         request: the request with the input and optional kwargs
         output_features: return the features of the input.
            Necessary if, e.g., attention mask is needed for post-processing.
        Returns:
             The model outputs and optionally the input features
        """

        all_predictions = []
        preprocessing_kwargs["padding"] = preprocessing_kwargs.get(
            "padding", True
        )
        preprocessing_kwargs["truncation"] = preprocessing_kwargs.get(
            "truncation", True
        )
        model.to(
            "cuda"
            if torch.cuda.is_available() and not disable_gpu
            else "cpu"
        )

        features = tokenizer(
            input, return_tensors="pt", **preprocessing_kwargs
        )

        for start_idx in range(0, len(input), batch_size):
            with torch.no_grad():
                input_features = {
                    k: features[k][start_idx: start_idx + batch_size]
                    for k in features.keys()
                }
                predictions = model(**input_features, **model_kwargs)
                all_predictions.append(predictions)

        keys = all_predictions[0].keys()
        final_prediction = {}
        for key in keys:
            # HuggingFace outputs for "attentions" and more is
            # returned as tuple of tensors
            # Tuple of tuples only exists for "past_key_values"
            # which is only relevant for generation.
            # Generation should NOT use this function
            if isinstance(all_predictions[0][key], tuple):
                tuple_of_lists = list(
                    zip(
                        *[
                            [
                                torch.stack(p).cpu()
                                if isinstance(p, tuple)
                                else p.cpu()
                                for p in tpl[key]
                            ]
                            for tpl in all_predictions
                        ]
                    )
                )
                final_prediction[key] = tuple(torch.cat(l) for l in tuple_of_lists)
            else:
                final_prediction[key] = torch.cat(
                    [p[key].cpu() for p in all_predictions]
                )
        if output_features:
            return final_prediction, features

        return final_prediction

def base_qa(model, tokenizer, input, preprocessing_kwargs, task_kwargs, model_kwargs):
    """
    Span-based question answering for a given question and context.
    We expect the input to use the (question, context) format for the text pairs.
    Args:
        request: the prediction request
    """    
    preprocessing_kwargs["truncation"] = "only_second"
    features = tokenizer(
        input, return_tensors="pt", **preprocessing_kwargs
    )
    predictions, features = base_predict(model, input, tokenizer, preprocessing_kwargs, model_kwargs, output_features=True)

    task_outputs = {
        "answers": [],
        "attributions": [],
        "adversarial": {
            "indices": [],
        },  # for hotflip, input_reduction and topk
    }

    for idx, (start, end, (_, context)) in enumerate(
            zip(predictions["start_logits"], predictions["end_logits"], input)
    ):
        # Ensure padded tokens & question tokens cannot
        # belong to the set of candidate answers.
        question_tokens = np.abs(np.array([s != 1 for s in features.sequence_ids(idx)]) - 1)
        # Unmask CLS token for "no answer"
        question_tokens[0] = 1
        undesired_tokens = question_tokens & features["attention_mask"][idx].numpy()

        # Generate mask
        undesired_tokens_mask = undesired_tokens == 0.0

        # Make sure non-context indexes in the tensor cannot
        # contribute to the softmax
        start = np.where(undesired_tokens_mask, -10000.0, start)
        end = np.where(undesired_tokens_mask, -10000.0, end)

        start = np.exp(start - np.log(np.sum(np.exp(start), axis=-1, keepdims=True)))
        end = np.exp(end - np.log(np.sum(np.exp(end), axis=-1, keepdims=True)))

        # Get score for "no answer" then mask for decoding step (CLS token
        no_answer_score = (start[0] * end[0]).item()
        start[0] = end[0] = 0.0

        starts, ends, scores = decode(
            start,
            end,
            task_kwargs.get("topk", 1),
            task_kwargs.get("max_answer_len", 128),
            undesired_tokens,
        )

        enc = features[idx]
        original_ans_start = enc.token_to_word(starts[0])
        original_ans_end = enc.token_to_word(ends[0])
        answers = [
            {
                "score": score.item(),
                "start": enc.word_to_chars(enc.token_to_word(s), sequence_index=1)[0],
                "end": enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[1],
                "answer": context[
                            enc.word_to_chars(enc.token_to_word(s), sequence_index=1)[0]: enc.word_to_chars(
                                enc.token_to_word(e), sequence_index=1
                            )[1]
                            ],
            }
            for s, e, score in zip(starts, ends, scores)
        ]
        if task_kwargs.get("show_null_answers", True):
            answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: task_kwargs.get("topk", 1)]
        task_outputs["answers"].append(answers)

    return predictions, task_outputs, original_ans_start, original_ans_end

def decode(
            start_: np.ndarray,
            end_: np.ndarray,
            topk: int,
            max_answer_len: int,
            undesired_tokens_: np.ndarray,
    ) -> Tuple:
    """
    Take the output of any :obj:`ModelForQuestionAnswering` and
        will generate probabilities for each span to be the
        actual answer.
    In addition, it filters out some unwanted/impossible cases
    like answer len being greater than max_answer_len or
    answer end position being before the starting position.
    The method supports output the k-best answer through
    the topk argument.
    Args:
        start_ (:obj:`np.ndarray`): Individual start
            probabilities for each token.
        end (:obj:`np.ndarray`): Individual end_ probabilities
            for each token.
        topk (:obj:`int`): Indicates how many possible answer
            span(s) to extract from the model output.
        max_answer_len (:obj:`int`): Maximum size of the answer
            to extract from the model"s output.
        undesired_tokens_ (:obj:`np.ndarray`): Mask determining
            tokens that can be part of the answer
    """
    # Ensure we have batch axis
    if start_.ndim == 1:
        start_ = start_[None]

    if end_.ndim == 1:
        end_ = end_[None]

    # Compute the score of each tuple(start_, end_) to be the real answer
    outer = np.matmul(np.expand_dims(start_, -1), np.expand_dims(end_, 1))

    # Remove candidate with end_ < start_ and end_ - start_ > max_answer_len
    candidates = np.tril(np.triu(outer), max_answer_len - 1)

    #  Inspired by Chen & al. (https://github.com/facebookresearch/DrQA)
    scores_flat = candidates.flatten()
    if topk == 1:
        idx_sort = [np.argmax(scores_flat)]
    elif len(scores_flat) < topk:
        idx_sort = np.argsort(-scores_flat)
    else:
        idx = np.argpartition(-scores_flat, topk)[0:topk]
        idx_sort = idx[np.argsort(-scores_flat[idx])]

    starts_, ends_ = np.unravel_index(idx_sort, candidates.shape)[1:]
    desired_spans = np.isin(starts_, undesired_tokens_.nonzero()) & np.isin(
        ends_, undesired_tokens_.nonzero()
    )
    starts_ = starts_[desired_spans]
    ends_ = ends_[desired_spans]
    scores_ = candidates[0, starts_, ends_]

    return starts_, ends_, scores_

# Code from SQuARE ONNX QA Pipeline (note: some features like explainability and attack mode have been removed)
def question_answering(model_qa, tokenizer, input, preprocessing_kwargs, task_kwargs, model_kwargs):
    """
    Span-based question answering for a given question and context.
    We expect the input to use the (question, context) format for the text pairs.
    Args:
        request: the prediction request
    """    
    preprocessing_kwargs["truncation"] = "only_second"

    features = tokenizer(
        input, return_tensors="np", **preprocessing_kwargs
    )
    onnx_inputs = {key: np.array(features[key], dtype=np.int64) for key in features}
    
    predictions_onnx = model_qa.run(input_feed=onnx_inputs, output_names=None)
    predictions = {
        "start_logits": predictions_onnx[0],
        "end_logits": predictions_onnx[1]
    }

    task_outputs = {
        "answers": [],
        "attributions": [],
        "adversarial": {
            "indices": [],
        },  # for hotflip, input_reduction and topk
    }

    for idx, (start, end, (_, context)) in enumerate(
            zip(predictions["start_logits"], predictions["end_logits"], input)
    ):
        # Ensure padded tokens & question tokens cannot
        # belong to the set of candidate answers.
        question_tokens = np.abs(np.array([s != 1 for s in features.sequence_ids(idx)]) - 1)
        # Unmask CLS token for "no answer"
        question_tokens[0] = 1
        undesired_tokens = question_tokens & features["attention_mask"][idx]

        # Generate mask
        undesired_tokens_mask = undesired_tokens == 0.0

        # Make sure non-context indexes in the tensor cannot
        # contribute to the softmax
        start = np.where(undesired_tokens_mask, -10000.0, start)
        end = np.where(undesired_tokens_mask, -10000.0, end)

        start = np.exp(start - np.log(np.sum(np.exp(start), axis=-1, keepdims=True)))
        end = np.exp(end - np.log(np.sum(np.exp(end), axis=-1, keepdims=True)))

        # Get score for "no answer" then mask for decoding step (CLS token
        no_answer_score = (start[0] * end[0]).item()
        start[0] = end[0] = 0.0

        starts, ends, scores = decode(
            start,
            end,
            task_kwargs.get("topk", 1),
            task_kwargs.get("max_answer_len", 128),
            undesired_tokens,
        )

        enc = features[idx]
        original_ans_start = enc.token_to_word(starts[0])
        original_ans_end = enc.token_to_word(ends[0])
        answers = [
            {
                "score": score.item(),
                "start": enc.word_to_chars(enc.token_to_word(s), sequence_index=1)[0],
                "end": enc.word_to_chars(enc.token_to_word(e), sequence_index=1)[1],
                "answer": context[
                            enc.word_to_chars(enc.token_to_word(s), sequence_index=1)[0]: enc.word_to_chars(
                                enc.token_to_word(e), sequence_index=1
                            )[1]
                            ],
            }
            for s, e, score in zip(starts, ends, scores)
        ]
        if task_kwargs.get("show_null_answers", True):
            answers.append({"score": no_answer_score, "start": 0, "end": 0, "answer": ""})
        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[: task_kwargs.get("topk", 1)]
        task_outputs["answers"].append(answers)

    return predictions, task_outputs, original_ans_start, original_ans_end



skill =  "span-extraction"
skills_df = load_skills(skill)

preprocessing_kwargs = {"padding": True, "truncation": True}
task_kwargs = {"show_null_answers": False, "topk": 1, "max_answer_len": 128}
model_kwargs = {"": {}}


def run_inf(
        data, modelname, run_func, tokenizer, adapter
    ):    

    df = pd.DataFrame(columns=[
            "skill", "reader", "adapter", "modelname",
            "timestamp", 
            "answer", 
            "data_id", "dataset", "question", "context", "answer_dataset"
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
        question = data["question"]
        context = example[context_name]
        answer_dataset = example[answers_name]
        
        p, task_outputs, _, _ = run_func(modelname, tokenizer, [example], preprocessing_kwargs, task_kwargs, model_kwargs)
        print(p)
        prediction = task_outputs["answers"][0][0]["answer"]

        
        data_set_name = adapter
        df.loc[len(df)] = [
            skill, reader, adapter, modelname,
            pd.Timestamp.now(),
            prediction, 
            example_id, data_set_name, question, context[:90], answer_dataset
        ]
    print(f"Finished {adapter}_{reader}_{modelname}")
    save_df(df, f"temp/{adapter}_{reader}_{modelname}.csv")

example_amount = 1

skipping_adapters = ["newsqa", "hotpot_qa"] 
# TODO quail for roberta
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
        
        print("Starting inf.")
        base_p = Process(target=run_inf, args=(data, "base", base_qa, base_model, tokenizer, adapter))
        quant_base_p = Process(target=run_inf, args=(data, "quant_base", base_qa, quantized_base_model, tokenizer, adapter))
        onnx_p = Process(target=run_inf, args=(data, "onnx", question_answering, onnx_model, tokenizer, adapter))
        onn_opt_p = Process(target=run_inf, args=(data, "onnx_opt", question_answering, onnx_model_opt, tokenizer, adapter))
        quant_onnx_p = Process(target=run_inf, args=(data, "quant_onnx", question_answering, onnx_model_quant, tokenizer, adapter))
        quant_onnx_opt_p = Process(target=run_inf, args=(data, "quant_onnx_opt", question_answering, onnx_model_quant_opt, tokenizer, adapter))
    
        base_p.start()
        quant_base_p.start()
        onnx_p.start()
        onn_opt_p.start()
        quant_onnx_p.start()
        quant_onnx_opt_p.start()

        base_p.join()
        quant_base_p.join()
        onnx_p.join()
        onn_opt_p.join()
        quant_onnx_p.join()
        quant_onnx_opt_p.join()