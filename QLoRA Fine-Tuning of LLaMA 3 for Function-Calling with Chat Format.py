# -*- coding: utf-8 -*-

from huggingface_hub import notebook_login

notebook_login()

!pip install transformers datasets evaluate peft trl wandb

!wandb login

import wandb
wb_token = "your_wb_token"
run = wandb.init(
    project="Fine-tune Llama 3.1 8b for Function Calling 2",
    job_type="training",
    anonymous="allow"
)

"""Dataset"""

from datasets import load_dataset
hermes_dataset = load_dataset("NousResearch/hermes-function-calling-v1", split='all')
# hermes_dataset = hermes_dataset.train_test_split(test_size=0.2)
print(hermes_dataset)
hermes_dataset = hermes_dataset.flatten()
# hermes_dataset['train'][0]

"""Tokenization and testing

"""

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
# Assign the eos_token as the pad_token
tokenizer.pad_token = tokenizer.eos_token


"""preprocess"""


def format_chat_template(row):
    row_json = [{"role": "system", "content": row['conversations'][0]['value']},
                {"role": "user", "content": row['conversations'][1]['value']},
                {"role": "assistant", "content": row['conversations'][2]['value']}]
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

dataset = hermes_dataset.map(
    format_chat_template,
    num_proc=4
)

dataset['text'][0]

dataset = dataset.train_test_split(test_size=0.1)

"""After tokenizing the text, the tokenizer returns a dictionary, typically containing:

input_ids: The tokenized text (numerical tokens).
attention_mask: A mask indicating which tokens are padding.

lora_alpha:

Higher values (e.g., 32 or 64) give stronger adaptation, useful for tasks that differ significantly from the pre-trained model.
Lower values (e.g., 8 or 16) provide more subtle fine-tuning, beneficial for tasks closer to the original model’s purpose or with limited data.

LoRA Dropout: Adds regularization specifically to LoRA updates, reducing overfitting risk.
Range: Between 0.0 and 1.0, with typical values around 0.1 to 0.3.
Choosing a Value: Start with 0.1 and adjust based on dataset size and overfitting tendencies; larger datasets can often handle lower dropout values.

In each training batch, dropout randomly drops out a subset of neurons, making the network effectively use a different subset of neurons each time.

and then scaling the active neurons after that. no scaling no dropout in inference or validation

Causal models generate text by predicting the next token without looking ahead, using only past context.
Autoregressive means the model generates each token in sequence, feeding each new token back into the model for the next prediction.
"""

from peft import LoraConfig, TaskType
#lora configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False, #whether we are using the model for inference or not
    r=16, #dimension of low rank matrices/adapters
    lora_alpha=32, #range from 16-64 #the strength of the adapter #the scaling factor for low rank matrices/adapters
    lora_dropout=0.05, #range from 0-1: probability of temporarily dropping out #the drop out probabilty for lora layers/adapters/low rank matrices
    bias="none",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'], #The names of the modules to apply the adapter to  #whats adapter
    init_lora_weights=False## to initiate with random weights
    )

"""get the model's module names to know which ones should be the target modules"""

# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8b-Instruct")
# for name, module in model.named_modules():
#     print(name)

"""Quantization for memory efficiency"""

!pip install accelerate

# !pip uninstall bitsandbytes

!pip install -U bitsandbytes

import torch
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

config = BitsAndBytesConfig(
    load_in_4bit=True, #load the model and its parameters like weights in 4 bit precision so it fits into our vram :if 7b param in 16 bit(2byte) precision: 7*2= 14 GB in Fp16
    bnb_4bit_quant_type="nf4",#nf4 uses a floating point representation in 4 bits so more accurate and more range #more accuracy than traditional integer-based 4-bit formats(INT4)
    bnb_4bit_use_double_quant=True, #double quantization, even the quantization params are quantized for lower space in memory
    bnb_4bit_compute_dtype=torch.float16,# we are gonna load the weights in 4 bit but store in 16 bit so we have more accuracy
)

"""
 should call the prepare_model_for_kbit_training() function to preprocess the quantized model for training.

"""

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",quantization_config=config,device_map="auto",attn_implementation="eager")
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

#Wrap the base model and peft_config with the get_peft_model() function to create a PeftModel.
from peft import get_peft_model

lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

"""to train with the Trainer class, setup a TrainingArguments class with some training hyperparameters.

Increasing ** gradient_accumulation_steps** saves VRAM because you can use a smaller batch size for each step, then accumulate gradients over multiple steps to simulate a larger batch.

logging_steps: Controls how often training metrics (e.g., loss) are logged, in steps. It’s purely for monitoring.
save_strategy: Controls when model checkpoints are saved, usually after each epoch or step. Saving the model is distinct from logging metrics, and they don’t conflict.

When group_by_length=True, the data loader groups inputs of similar lengths together in each batch.
This minimizes the amount of padding required, as sequences within a batch will be closer in length.
Reducing padding helps the model process batches faster and more efficiently, as the model doesn’t waste time on unnecessary padded tokens.
"""

from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="nilamasrouri98/hermes-function-calling/LoRA-FineTune",#where the model checkpoints and logs are saved
    learning_rate=2e-4, #1e-3 can be large and its useful for adapting quickly on small tasks
    per_device_train_batch_size=1,#10 #using size 1 for batch saves vram
    per_device_eval_batch_size=1,#not needed we dont have eval
    gradient_accumulation_steps=2,#added #controls how many batches the model should process before it performs an optimization (weight update) step
    optim="paged_adamw_8bit",#changed 32 bit to 8 to save vram
    eval_steps=0.2, #1000: if number means after how many, if fraction mean after what percentage of epoch, #added #evaluation after how many steps
    num_train_epochs=2,#4 #full path through dataset
    # weight_decay=0.01, #regularization to penalize large weights by adding penalty to loss function
    evaluation_strategy="steps",#"step" #epoch means evaluate after each epoch, step means after each step
    logging_steps=1,#added #The frequency of logging training metrics, in steps
    logging_strategy="steps",
    # lr_scheduler_type="linear",#added #adjusts and reduce the learning rate gradually which helps the model converge without drastic changes
    warmup_steps=10,#added #num of initial steps during which the learning rate gradually increases from 0 to the set learning rate: preventing sudden large updates
    report_to="wandb", #added #log to wheights and biases which tracks and visualizes the training metrics
    save_strategy="steps",#how often model checkpoints are saved #should match eval strategy, changed from epoch
    group_by_length=True,#added #data loader groups similar length of inputs in each batch for faster training and less memory(less padding)
    # load_best_model_at_end=True,#loads the best final model
    bf16=True,
    fp16=False,  # Enable mixed-precision (FP16) training #apply 16-bit (half-precision) precision specifically for the training updates # default is Fp32 full precision for all new parameters, gradients, and training updates.
)

"""Add adapter because of this error without adapter after quantizing
ValueError: You cannot perform fine-tuning on purely quantized models. Please attach trainable adapters on top of the quantized model to correctly perform fine-tuning. Please see: https://huggingface.co/docs/transformers/peft for more details

This error occures because
occurs because when using quantization (particularly 4-bit or 8-bit), the model weights are frozen and cannot be fine-tuned directly. Quantization reduces memory usage and increases inference speed but limits the ability to update model weights. To enable fine-tuning, trainable adapters need to be attached on top of the quantized model.
"""

lora_model.gradient_checkpointing_enable()
lora_model.config.use_cache = False

"""Pass the model, training arguments, dataset, tokenizer, and any other necessary component to the Trainer

Can You Use Trainer for LoRA or QLoRA Fine-Tuning?
Yes, you can use Trainer for LoRA or QLoRA fine-tuning, but with some considerations:

Trainer Compatibility: Trainer doesn’t have built-in support for PEFT (parameter-efficient fine-tuning) configurations like LoRA or QLoRA. However, you can still fine-tune a model with LoRA or QLoRA using Trainer if you manually apply the LoRA or QLoRA layers to the model before passing it to Trainer.

How to Make It Work:

You’d need to set up the LoRA or QLoRA configurations separately, apply them to the model(get_peft_model), and then load this modified model into Trainer.
This approach is commonly used, especially if you want to take advantage of Trainer's other features like distributed training, mixed precision, or custom training schedules.
In short: While SFTTrainer is specifically built to handle LoRA, Trainer can still fine-tune LoRA or QLoRA models with manual setup.
"""

# Define the compute_metrics function
import evaluate
import numpy as np
# metric = evaluate.load("accuracy")


#SFTTrainer instead of trainer
from trl import SFTTrainer
trainer = SFTTrainer(
    model=lora_model,
    # train_dataset=tokenized_dataset["train"],
    # eval_dataset=tokenized_dataset["test"],
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    max_seq_length=1024,#maximum length of input sequences (in tokens) and if the input is longer, it gets truncated
    dataset_text_field="text",#which field in the dataset contains the text data
    tokenizer=tokenizer,
    args=training_args,
    # compute_metrics=compute_metrics,
    packing=False #Sequence Packing combines shorter sequences into a single sequence up to max_seq_length by inserting special tokens (like [SEP]) between them.This minimizes wasted space in each batch, reducing the amount of padding and making computations more efficient.
)
#now we train
trainer.train()




