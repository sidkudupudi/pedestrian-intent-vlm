import os
import torch
from datasets import load_dataset
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
# The recorded run trained on the 524-clip training split (131 optimizer steps/epoch x 4 grad-accumulation = 524),
# keeping the 131 clips in splits/v2_test.jsonl held out. qwen_v2_training_data.jsonl holds all 655 clips.
DATASET_FILE = "splits/v2_train.jsonl"
OUTPUT_DIR = "./qwen-pedestrian-intent-v2"

print("🚀 Initializing V2 Hyper-Contextual Training Pipeline...")

# 1. Load Dataset
print(f"Loading training data from {DATASET_FILE}...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print(f"Loaded {len(dataset)} contextual video examples.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading Base Model and Processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 4. Prepare for PEFT / LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

class Qwen2VLDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        messages_batch = [example["messages"] for example in examples]
        
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) 
            for msg in messages_batch
        ]
        
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        batch = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

collator = Qwen2VLDataCollator(processor)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4, 
    learning_rate=2e-4,
    num_train_epochs=5,
    optim="paged_adamw_8bit",
    bf16=True, 
    logging_steps=5,
    save_strategy="epoch",
    gradient_checkpointing=True, 
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collator,
)

print("\n Starting Fine-Tuning! (This will take a while)...")
trainer.train()

print(f"\n✅ Training Complete! Saving V2 LoRA adapters to {OUTPUT_DIR}...")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done.")