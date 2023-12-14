from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
import os, torch

def collate(elements, tokenizer):
	tokenlist=[e["input_ids"] for e in elements]
	tokens_maxlen=max([len(t) for t in tokenlist])

	input_ids,labels,attention_masks = [],[],[]
	for tokens in tokenlist:
		pad_len=tokens_maxlen-len(tokens)

		# pad input_ids with pad_token, labels with ignore_index (-100) and set attention_mask 1 where content otherwise 0
		input_ids.append( tokens + [tokenizer.pad_token_id]*pad_len )   
		labels.append( tokens + [-100]*pad_len )    
		attention_masks.append( [1]*len(tokens) + [0]*pad_len ) 

	batch={
		"input_ids": torch.tensor(input_ids),
		"labels": torch.tensor(labels),
		"attention_mask": torch.tensor(attention_masks)
	}
	return batch


def get_dataloader(
    dataset,
    batch_size,
    collator,
    fsdp_info,
    shuffle=False,
    seed=42,
):
	fsdp_rank, fsdp_world_size = fsdp_info

	sampler = DistributedSampler(dataset=dataset, rank=fsdp_rank, num_replicas=fsdp_world_size, shuffle=False)
	loader = DataLoader(
		dataset,
		shuffle=False,
		pin_memory=True,
		batch_size=batch_size,
		collate_fn=collator,
		sampler=sampler,
	)
	return sampler, loader

def tokenize(element, max_length):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

fsdp_rank, fsdp_world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

tokenizer = AutoTokenizer.from_pretrained("models/Mistral-7B-v0.1", use_fast=False)
tokenizer.pad_token_id = tokenizer.eos_token_id

samples=[
	{"text": "sample 1"},
	{"text": "sample 2"},
	{"text": "sample 3"},
	{"text": "sample 4"},
	{"text": "sample 5"},
	{"text": "sample 6"},
	{"text": "sample 7"},
	{"text": "sample 8"},
	{"text": "sample 9"},
]

dataset = Dataset.from_list(samples)
dataset_tokenized = dataset.map(
    partial(tokenize, max_length=512), 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)

train_sampler, train_loader = get_dataloader(
    dataset=dataset_tokenized,
    batch_size=2,
    collator=partial(collate, tokenizer=tokenizer),
    fsdp_info=[fsdp_rank, fsdp_world_size]
)

for epoch in range(0, 2):
	train_sampler.set_epoch(epoch)

	for step, batch in enumerate(train_loader):
		print(
			f"rank {str(fsdp_rank)}, epoch {epoch}, step {step}: ",
			tokenizer.batch_decode(batch["input_ids"])
		)

