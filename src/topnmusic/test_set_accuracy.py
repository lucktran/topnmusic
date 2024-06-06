#!/bin/env python3

import json

import accelerate
from datasets import Audio
from datasets import load_dataset
import evaluate
import librosa
import numpy as np
from transformers import ASTFeatureExtractor
from transformers import ASTForAudioClassification
from transformers import ASTConfig
from transformers import models
from transformers import AutoConfig
from transformers import Trainer
from transformers import TrainingArguments


# load in gtzan dataset
gtzan = load_dataset("marsyas/gtzan", "all", trust_remote_code=True)

# create function mapping label integer to genre name
id2label_fn = gtzan["train"].features["genre"].int2str

# get model's feature extractor
feature_extractor = ASTFeatureExtractor()

# cast train and test datasets to sampling rate used in pre-trained model
sampling_rate = feature_extractor.sampling_rate
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

# preprocess audio
def preprocess_function(examples):
    # examples = gtzan["train"]
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        raw_speech=audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors='pt'
    )
    return inputs

gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)

# change column name from genre to label
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
# map label to id, id to label
id2label = {
    str(i): id2label_fn(i)
    for i in range(len(gtzan_encoded["train"].features["label"].names))
}
label2id = {v: k for k, v in id2label.items()}

# split into 70-20-10 train, val, test datasets
test_split = gtzan_encoded["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
train_val_split = test_split["train"].train_test_split(seed=42, shuffle=True, test_size=0.222)

# TODO: make graphs of the dataset split

# load in the pre-trained model configuration
model_id = "connordilgren/ast_finetuned_audioset_finetuned_gtzan"
model = ASTForAudioClassification.from_pretrained(
    model_id,
)

# create a dummy trainer
training_args = TrainingArguments("test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer = Trainer(
    model,
    training_args,
    train_dataset=test_split["train"],
    eval_dataset=test_split["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

# evaluate trainer on test set
results = trainer.evaluate()

# save results
with open("model_params_compare/checkpoint_model_lr_5e_05_bs_16/checkpoint-1144/test_set_accuracy.json", "w") as outfile:
    json.dump(results, outfile)
