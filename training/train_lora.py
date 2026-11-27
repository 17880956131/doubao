#!/usr/bin/env python3
"""在 GPU（电脑/Colab/Kaggle/云）上做 QLoRA 蒸馏；手机只负责推理。
用法：python train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --data sft.jsonl --out ./zhiming-lora
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments)
from trl import SFTTrainer


def format_row(row):
    return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)


def main():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data", default="sft.jsonl")
    parser.add_argument("--out", default="./zhiming-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.map(lambda row: {"text": format_row(row)})

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="epoch",
            bf16=True,
            gradient_checkpointing=True,
            report_to="none",
        ),
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(json.dumps({"ok": True, "out": args.out, "samples": len(dataset)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
