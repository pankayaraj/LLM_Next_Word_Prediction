from datasets import load_dataset, DatasetDict, load_from_disk
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig,PeftConfig, PeftModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import wandb

parser = argparse.ArgumentParser(description='Watermarking arguments')
parser.add_argument("--model", type=str, default="llama-30b")
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args()


wandb.init(
    # set the wandb project where this run will be logged
    project="Pre Training",
    
    # track hyperparameters and run metadata
    config={
    "model": args.model,
    }
)

if args.model == "llama-13b":
    
    path = "/cmlscratch/pan/Watermarking/models/llama-13b"

    peft_config = LoraConfig(
             r=16,
             lora_alpha=32,
             lora_dropout=0.05,
             target_modules=["q_proj", "v_proj"],
             bias="none",
             task_type="CAUSAL_LM",
    )


    max_length = 2048
    model = AutoModelForCausalLM.from_pretrained(path,
                                         device_map="auto",
                                         torch_dtype=torch.bfloat16,)
    
    model.add_adapter(peft_config)

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

elif args.model == "llama-30b":
    
    path = "/cmlscratch/pan/Watermarking/models/llama-30b"

    peft_config = LoraConfig(
             r=8,
             lora_alpha=16,
             lora_dropout=0.05,
             target_modules=["q_proj", "v_proj"],
             bias="none",
             task_type="CAUSAL_LM",
    )


    max_length = 2048
    model = AutoModelForCausalLM.from_pretrained(path,
                                          device_map="auto",
                                          torch_dtype=torch.bfloat16,)
    model.add_adapter(peft_config)

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

elif args.model == "gpt-neox-20b":
    path = "/cmlscratch/pan/Watermarking/models/gpt-neox-20b"

    peft_config = LoraConfig(
             r=8,
             lora_alpha=16,
             lora_dropout=0.05,
             target_modules=["query_key_value"],
             bias="none",
             task_type="CAUSAL_LM",
    )


    max_length = 2048
    model = AutoModelForCausalLM.from_pretrained(path,
                                         device_map="auto",
                                         torch_dtype=torch.bfloat16,)
    
    model.add_adapter(peft_config)

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token


elif args.model == "opt-2.7b":
    path = "/cmlscratch/pan/Watermarking/models/opt-2.7b"

    peft_config = LoraConfig(
             r=16,
             lora_alpha=32,
             lora_dropout=0.05,
             target_modules=["q_proj", "v_proj"],
             bias="none",
             task_type="CAUSAL_LM",
    )


    max_length = 2048
    model = AutoModelForCausalLM.from_pretrained(path,
                                         device_map="auto",
                                         torch_dtype=torch.bfloat16,)
    
    model.add_adapter(peft_config)



    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token

elif args.model == "pythia-2.8b":
    path = "/cmlscratch/pan/Watermarking/models/pythia-2.8b"

    peft_config = LoraConfig(
             r=16,
             lora_alpha=32,
             lora_dropout=0.05,
             target_modules=["query_key_value"],
             bias="none",
             task_type="CAUSAL_LM",
    )


    max_length = 2048
    model = AutoModelForCausalLM.from_pretrained(path,
                                         device_map="auto",
                                         torch_dtype=torch.bfloat16,)
    
    print(model)
    model.add_adapter(peft_config)

    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token





#dataset with both data seen in training and not seen in traininig
dataset = load_from_disk("/cmlscratch/pan/Watermarking/Datasets/dataset/bookMIA")

#dataset that is seen in training
training_dataset = dataset["seen_in_training"]



def tokenize(element):
    outputs = tokenizer(
        element["snippet"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
        padding='max_length'
    )
    #input_batch = []
   
    #idx = 0
    # for length, input_ids in zip(outputs["length"], outputs["input_ids"]): 
    #     idx += 1   
    #     if length == max_length:
    #         input_batch.append(input_ids)

    #input_batch.append(outputs["input_ids"][0])

    return {"input_ids": outputs["input_ids"][0]}


#the dataset to train on 
tokenized_training_dataset = training_dataset.map(
    tokenize, batched=False, remove_columns=training_dataset.column_names
)

print(model)
print(training_dataset)
print(tokenized_training_dataset)


learning_rate = 5e-4

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="/cmlscratch/pan/Watermarking/models/trained/" + str(args.model) + str(learning_rate),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    evaluation_strategy="steps",
    eval_steps=int(5135/(32*args.epochs)),
    logging_first_step=True,
    logging_steps=int(5135/(32*args.epochs)), 
    num_train_epochs=args.epochs,
    weight_decay=0.1,
    warmup_steps=150,
    lr_scheduler_type="cosine",
    learning_rate=learning_rate,
    save_steps=int(5135/(32*args.epochs)),
    bf16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_training_dataset,
    eval_dataset=tokenized_training_dataset,
)

trainer.train()