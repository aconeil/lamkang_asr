'''
This script loads a dataset from a folder using the CSV in the folder.
The dataset is used to fine-tune and test a wav2vec2 model
python w2v2_finetune.py dataset_directory output_directory hf_repo_name
'''

import re
import numpy as np
from huggingface_hub import login
import random
import pandas as pd
import torch
import sys
from evaluate import load
from datasets import load_dataset, Audio, concatenate_datasets
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, TrainingArguments, Trainer

#Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

#Third argument on CL is HF repo name
repo_name = sys.argv[3]

#Second argument on CL is desired output directory
output_dir = sys.argv[2]

base_model = "facebook/wav2vec2-large-xlsr-53"

#Note: ' is a character in Lamkang and not removed
chars_to_remove_regex = "[\n\,\?\.\!\-\;\:\"\“\%\‘\”\�\»\«\’\@\<\>\(\)]"

def remove_special_characters(batch):
    # remove special characters
    batch["transcription"] = re.sub(chars_to_remove_regex, "", batch["transcription"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    batch["labels"] = processor(text=batch["transcription"]).input_ids
    return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",)
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",)
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

login('INSERT_TOKEN_HERE')

dir_name = sys.argv[1]

lmk_train = load_dataset(dir_name, split="train")

lmk_test = load_dataset(dir_name, split="test")

lmk_train = lmk_train.map(remove_special_characters)

lmk_test = lmk_test.map(remove_special_characters)

vocab_train = lmk_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=lmk_train.column_names)


vocab_test = lmk_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=lmk_test.column_names)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w") as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

tokenizer.push_to_hub(repo_name)

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

processor.push_to_hub(repo_name)

lmk_train = lmk_train.cast_column("audio", Audio(sampling_rate=16000))

lmk_test = lmk_test.cast_column("audio", Audio(sampling_rate=16000))

#uncomment to upload train datset to HuggingFace
#lmk_train.push_to_hub("aconeil/manual_align_updated")

lmk_train = lmk_train.map(prepare_dataset, remove_columns=lmk_train.column_names)

lmk_test = lmk_test.map(prepare_dataset, remove_columns=lmk_test.column_names)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load("wer")

cer_metric = load("cer")

#Adjust hyperparameters as fit
model = Wav2Vec2ForCTC.from_pretrained(
    base_model,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    gradient_checkpointing=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

#Adjust hyperparameters as fit
training_args = TrainingArguments(
  output_dir=output_dir,
  group_by_length=True,
  per_device_train_batch_size=8,
  gradient_accumulation_steps=2,
  eval_strategy="steps",
  num_train_epochs=100,
  gradient_checkpointing=True,
  fp16=True,
  save_steps=400,
  eval_steps=100,
  logging_steps=50,
  learning_rate=1e-4,
  warmup_steps=300,
  #weight_decay=0.01,
  save_total_limit=10,
  push_to_hub=True,
  load_best_model_at_end=True,
  metric_for_best_model="cer",
  greater_is_better=False,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=lmk_train,
    eval_dataset=lmk_test,
    tokenizer=processor.feature_extractor,
)

#Option to resume from checkpoint here
trainer.train()#resume_from_checkpoint='PATH_TO_CHECKPOINT/')

trainer.push_to_hub()

