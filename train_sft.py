import json
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
import os

def train():

    model_path = "google/gemma-2b"
    data_path1 = '/home/yran1/NLP/proposal/train_data'
    output_dir = 'gemma_output'

    max_seq_length = 512

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # load_in_4bit=True,
        device_map="auto"
    )

    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for input, output in zip(inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = f"Query: {input} " \
                   f"Response: {output}"
            text = text + EOS_TOKEN
            texts.append(text)
        return {"text": texts, }

    dataset = load_dataset("json", data_files=f"{data_path1}/*.json", split="train", field="train")

    dataset = dataset.map(formatting_prompts_func, batched=True)
    dataset = dataset.remove_columns(["input", "output"])

    model.resize_token_embeddings(len(tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=3,
            learning_rate=1e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_hf",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="./outputs",
        ),
    )

    trainer.train()

    trainer.save_model(output_dir)

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer, dataset


if __name__ == '__main__':
    # Train
    model, tokenizer, dataset = train()
    # Test