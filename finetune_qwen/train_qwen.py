import json
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
import pdb
from datasets import Dataset
import os
import pandas as pd
import shutil
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import re
import numpy as np
import math
from datasets import load_from_disk

class Trie(object):
    def __init__(self, sequences):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: List[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: List[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: List[int], trie_dict: Dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: List[int],
        trie_dict: Dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def prefix_allowed_tokens_fn(candidate_trie, prompt_lens, tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id

    def prefix_allowed_tokens(batch_id, sentence):
        gen_only = sentence[prompt_lens:].tolist()
        trie_out = candidate_trie.get(gen_only)

        if not trie_out:
            return [eos_id]

        return trie_out

    return prefix_allowed_tokens


def qwen_datasets(seq_data_path, codebook_path, idmap_path, mode='train'):
    # 最长历史序列长度
    max_his = 100


    with open(seq_data_path, mode='r', encoding='utf-8') as f:
        user_seq = json.load(f)
    
    with open(codebook_path, mode='r', encoding='utf-8') as f:
        codebook_map = json.load(f)

    with open(idmap_path, mode='r', encoding='utf-8') as f:
        id_map = {}
        for line in f:
            idmap = json.loads(line)
            id_map[idmap['asin']] = idmap['id']
    
    
    user_seq_mapped = {
                        user: [codebook_map[str(id_map[t])] for t in seq if t in id_map]
                        for user, seq in user_seq.items()
                    }
    
    seqs = [i for i in user_seq_mapped.values()]
    
    data_samples = []
    if mode == 'train':
        for seq in seqs:
            items = seq[:-2]

            for i in range(len(items)):
                if i == 0:
                    continue
                one_sample = dict()
                one_sample["target"] = items[i]
                history = items[:i]
                if max_his > 0:
                    history = history[-max_his:]
                one_sample["history"] = history
                data_samples.append(one_sample)

    elif mode == 'valid':
         for seq in seqs:
            items = seq

            one_sample = dict()
            one_sample["target"] = items[-2]
            history = items[:-2]
            if max_his > 0:
                history = history[-max_his:]
                one_sample["history"] = history
            data_samples.append(one_sample)
    
    elif mode == 'test':
         for seq in seqs:
            items = seq

            one_sample = dict()
            one_sample["target"] = items[-1]
            history = items[:-1]
            if max_his > 0:
                history = history[-max_his:]
                one_sample["history"] = history
            data_samples.append(one_sample)

    return data_samples, codebook_map

PROMPT_seq = """You are a recommender responsible for suggesting the next item a user might purchase based on their historical buying behavior. Please process the input according to the following rules:
            Input Format:1. The user will provide a sequence of previously purchased items, each represented by four codes.
                         2. The codes for each item are separated by commas and enclosed in curly braces {}, e.g., {code1, code2, code3, code4}.
                         3. Multiple items are also separated by commas, e.g., {code1, code2, code3, code4},{code5, code6, code7, code8},{code9, code10, code11, code12}.
            Output Format:1. You need to output a recommended item consisting of four codes. The four output codes must be separated by commas and enclosed in curly braces {}, e.g., {recommended_code1, recommended_code2, recommended_code3, recommended_code4}. 
                          2. Do not add quotes, extra punctuation, or any explanatory text."""
MAX_LENGTH = 32768

def dataset_jsonl_transfer(data_samples, save_data, mode='Train'):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    
    seq_datasets = []

    for line in data_samples:

        seq = ','.join(['{' + ','.join(map(str, i)) + '}' for i in line['history']])
        target = '{' + ','.join(map(str, line['target'])) + '}'

        seq_dataset = {
            "instruction":PROMPT_seq,
            "input": f"What would the user purchase after {seq}",
            "output": f"{target}",
        }
        seq_datasets.append(seq_dataset)

    with open(save_data, mode='w', encoding='UTF-8') as f1:
        for seq in seq_datasets:
            f1.write(json.dumps(seq, ensure_ascii=False) + "\n")

def left_pad_sequences(sequences, padding_value):
    max_len = max(seq.size(0) for seq in sequences)
    padded = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padded.append(
            torch.cat([torch.full((pad_len,), padding_value, dtype=seq.dtype, device=seq.device),
                       seq])
        )
    return torch.stack(padded, dim=0)

def process_func(example):
    instruction_ids = tokenizer(
        f"<|im_start|>system\n{PROMPT_seq}<|im_end|>\n"
        f"<|im_start|>user\n{example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction_ids["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction_ids["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction_ids["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        # 关键：保留提示长度
        "prompt_len": len(instruction_ids["input_ids"]),
    }

class ConstrainedCLMTrainer(Trainer):
    def __init__(self, *args, candidate_trie=None, topk=10, **kwargs):
        # 不要把 tokenizer 传给父类；改走 processing_class（在外部实例化时传）
        super().__init__(*args, **kwargs)
        self.candidate_trie = candidate_trie
        self.topk = topk
        self.fixed_gen_len = 32
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        if prediction_loss_only or not self.args.predict_with_generate:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        prompt_lens = inputs.pop("prompt_lens").tolist()
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        prompt_ids_list  = [input_ids[i, :L] for i, L in enumerate(prompt_lens)]
        prompt_attn_list = [attention_mask[i, :L] for i, L in enumerate(prompt_lens)]

        # 关键：改为使用 processing_class（等价于 tokenizer）
        tok = self.processing_class
        pad_id = tok.pad_token_id or tok.eos_token_id
        eos_id = tok.eos_token_id

        prompts      = left_pad_sequences(prompt_ids_list, padding_value=pad_id)
        prompts_attn = left_pad_sequences(prompt_attn_list, padding_value=0)
        B = prompts.size(0)
        input_len = prompts.size(1)

        def prefix_allowed_tokens(batch_id, sent):
            gen_only = sent[input_len:].tolist()
            out = self.candidate_trie.get(gen_only)
            return out if out else [eos_id]

        gen_kwargs = dict(
            max_new_tokens=self.fixed_gen_len,
            num_beams=max(self.topk, getattr(self.args, "generation_num_beams", 1) or 1),
            num_return_sequences=self.topk,
            do_sample=False,
            early_stopping=True,
            prefix_allowed_tokens_fn=prefix_allowed_tokens,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            length_penalty=1.0,
            return_dict_in_generate=False,
        )
        generated = model.generate(input_ids=prompts, attention_mask=prompts_attn, **gen_kwargs)

        rows = []
        for b in range(B):
            for k in range(self.topk):
                row = generated[b*self.topk + k]
                gen_only = row[input_len:]
                rows.append(gen_only)
        gen_only_padded = left_pad_sequences(rows, padding_value=pad_id)

        target_len = self.fixed_gen_len
        t = gen_only_padded
        if t.size(1) < target_len:
            pad = t.new_full((t.size(0), target_len - t.size(1)), pad_id)
            t = torch.cat([t, pad], dim=1)
        elif t.size(1) > target_len:
            t = t[:, :target_len]
        preds_t = t.view(B, self.topk, target_len)

        outputs = model(**inputs)
        loss = outputs.loss if "labels" in inputs else None
        lbls_t  = inputs["labels"].detach().cpu() if "labels" in inputs else None
        return (loss.detach().cpu() if loss is not None else None, preds_t.detach().cpu(), lbls_t)

    
brace_pat = re.compile(r"\{([^}]*)\}")

def parse_codes(text: str):
    m = brace_pat.search(text)
    if not m:
        return tuple()
    parts = [p.strip() for p in m.group(1).split(",")]
    return tuple(parts)

def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def make_compute_metrics(tokenizer, ks=(1,3,5,10)):
    ks = tuple(sorted(set(ks)))
    def compute_metrics(eval_pred):
        preds, labels = eval_pred   # preds: (B, topk, T)；labels: (B, T)

        # 解码预测 Top-K
        B, topk, T = preds.shape
        pred_texts = [[tokenizer.decode(preds[b, k], skip_special_tokens=True)
                       for k in range(topk)] for b in range(B)]
        # 解码金标
        label_texts = []
        for b in range(labels.shape[0]):
            row = np.array(labels[b])
            ref_ids = row[row != -100]
            label_texts.append(tokenizer.decode(ref_ids, skip_special_tokens=True))

        # 逐样本评估
        metrics = {f"Recall@{k}": 0.0 for k in ks}
        metrics.update({f"NDCG@{k}": 0.0 for k in ks})

        for b in range(B):
            # 预测序列 → 去重后的 Top-K 代码元组列表
            cand_codes = [parse_codes(t) for t in pred_texts[b]]
            cand_codes = [c for c in cand_codes if len(c) > 0]
            cand_codes = unique_preserve_order(cand_codes)

            # 金标集合（可扩展到多个正例）
            gold = parse_codes(label_texts[b])
            gold_set = {gold} if len(gold) > 0 else set()

            for K in ks:
                topK = cand_codes[:K]
                topK_set = set(topK)

                # Recall@K（多正例时：|topK∩gold| / |gold|；单正例时就是命中/未命中）
                inter = topK_set & gold_set
                denom = max(1, len(gold_set))
                recK = len(inter) / denom
                metrics[f"Recall@{K}"] += recK

                # NDCG@K
                # 相关性列表（按排序次序，命中=1，否则=0）
                rels = [1 if x in gold_set else 0 for x in topK]
                dcg = sum(rel / math.log2(i+2) for i, rel in enumerate(rels))
                # IDCG：理想命中 min(|gold|,K) 个
                idcg = sum(1.0 / math.log2(i+2) for i in range(min(len(gold_set), K)))
                ndcg = (dcg / idcg) if idcg > 0 else 0.0
                metrics[f"NDCG@{K}"] += ndcg

        # 平均
        for k in list(metrics.keys()):
            metrics[k] /= max(1, B)
        return metrics
    return compute_metrics

if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    local_model_path = "./cache/AmazonReviews2014/models/Qwen3-0.6B/QWen/Qwen3-0.6B/"
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto")
    # model.enable_input_require_grads()
    gc = model.generation_config

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    seq_data_path = '../../gen_rec/RPG-rqvae-rpg/cache/AmazonReviews2014/Beauty/processed/all_item_seqs.json'
    codebook_path = '../../gen_rec/RPG-rqvae-rpg/cache/AmazonReviews2014/Beauty/codebook/codebook.json'
    idmap_path = '../../gen_rec/RPG-rqvae-rpg/cache/AmazonReviews2014/Beauty/processed_category/id2category.json'

    mode='train'
    train_seq_data = f'./cache/AmazonReviews2014/Beauty/processed_category/{mode}_data'
    data_samples, codebook_map = qwen_datasets(seq_data_path, codebook_path, idmap_path, mode=mode)
    dataset_jsonl_transfer(data_samples, train_seq_data, mode=mode)

    mode='valid'
    valid_seq_data = f'./cache/AmazonReviews2014/Beauty/processed_category/{mode}_data'
    data_samples, codebook_map = qwen_datasets(seq_data_path, codebook_path, idmap_path, mode=mode)
    dataset_jsonl_transfer(data_samples, valid_seq_data, mode=mode)

    mode='test'
    test_seq_data = f'./cache/AmazonReviews2014/Beauty/processed_category/{mode}_data'
    data_samples, codebook_map = qwen_datasets(seq_data_path, codebook_path, idmap_path, mode=mode)
    dataset_jsonl_transfer(data_samples, test_seq_data, mode=mode)

    SAVE_DIR = "./cache/AmazonReviews2014/Beauty/hf_datasets"
    os.makedirs(SAVE_DIR, exist_ok=True)
    if not os.path.exists(f"{SAVE_DIR}/train"):
        train_df = pd.read_json(train_seq_data, lines=True)
        train_ds = Dataset.from_pandas(train_df)
        train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)
        train_dataset.save_to_disk(f"{SAVE_DIR}/train")
    else:
        train_dataset = load_from_disk(f"{SAVE_DIR}/train")
    
    if not os.path.exists(f"{SAVE_DIR}/valid"):
        valid_df = pd.read_json(valid_seq_data, lines=True)
        valid_ds = Dataset.from_pandas(valid_df)
        valid_dataset = valid_ds.map(process_func, remove_columns=valid_ds.column_names)
        valid_dataset.save_to_disk(f"{SAVE_DIR}/valid")
    else:
        valid_dataset = load_from_disk(f"{SAVE_DIR}/valid")

    if not os.path.exists(f"{SAVE_DIR}/test"):
        test_df = pd.read_json(test_seq_data, lines=True)
        test_ds = Dataset.from_pandas(test_df)
        test_dataset = test_ds.map(process_func, remove_columns=test_ds.column_names)
        test_dataset.save_to_disk(f"{SAVE_DIR}/test")
    else:
        test_dataset = load_from_disk(f"{SAVE_DIR}/test")

    # 2) Trie 构造
    candidate_trie = Trie([
        tokenizer.encode('{' + ','.join(map(str, v)) + '}', add_special_tokens=False)
        for _, v in codebook_map.items()
    ])

    # 3) 评测 Top-K 清单
    K_LIST = (1, 3, 5, 10)
    TOPK = max(K_LIST)

    # 4) TrainingArguments：确保 predict_with_generate=True（触发我们自定义的 generate）
    args = Seq2SeqTrainingArguments(
            output_dir="./autodl2-tmp/output/Qwen3-0.6B",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=64,
            eval_strategy="steps",     
            eval_steps=100,
            logging_steps=10,
            num_train_epochs=1,
            save_steps=400,
            learning_rate=1e-5,
            save_on_each_node=True,
            # gradient_checkpointing=True,
            predict_with_generate=True,      # ← 现在归属在 Seq2SeqTrainingArguments 中
            generation_num_beams=TOPK,       # ← 同上
            remove_unused_columns=False,     # ← 防止 HF 移除您自定义的字段
            dataloader_num_workers=4,
            # 关键：不做任何 checkpoint
            save_strategy="no",
            load_best_model_at_end=False,    # 保险起见，避免尝试“保存与回载最佳模型”
        )

    base_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    def collate_with_prompt_len(features):
        prompt_lens = [f.pop("prompt_len") for f in features]
        batch = base_collator(features)
        batch["prompt_lens"] = torch.tensor(prompt_lens, dtype=torch.long)
        return batch

    # 6) 计算指标
    compute_metrics = make_compute_metrics(tokenizer, ks=K_LIST)

    # 7) Trainer
    trainer = ConstrainedCLMTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        candidate_trie=candidate_trie,
        topk=TOPK,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collate_with_prompt_len,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_res = trainer.evaluate(eval_dataset=test_dataset)
    print(test_res) 
