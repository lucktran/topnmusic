#!/bin/env python3

# !pip install accelerate -U
# !pip install datasets
# !pip install evaluate
# !pip install git+https://github.com/huggingface/transformers

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

# split into train and test datasets
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)

# create function mapping label integer to genre name
id2label_fn = gtzan["train"].features["genre"].int2str

# def generate_audio():
#     # gets sampling rate, signal, and genre of a random sample in training set
#     example = gtzan["train"].shuffle()[0]
#     audio = example["audio"]
#     return (
#         audio["sampling_rate"],
#         audio["array"],
#     ), id2label_fn(example["genre"])

# get model's feature extractor
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
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

# load in the pre-trained model configuration
model = ASTForAudioClassification.from_pretrained(
    model_id,
)

# initialize linear classification layer, random weights and biases
num_labels = len(id2label)
config = AutoConfig.from_pretrained(model_id)
config.num_labels = num_labels
config.label2id = label2id
config.id2label = id2label

config = ASTConfig(
    num_labels = len(id2label),
    label2id = label2id,
    id2label = id2label,
)

classifier = models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer.ASTMLPHead(config)

# replace existing classifier in pre-trained model with new one
model.classifier = classifier
model.config = config
model.num_labels = num_labels

# train model
model_name = model_id.split("/")[-1]
batch_size = 8
gradient_accumulation_steps = 1
num_train_epochs = 10

training_args = TrainingArguments(
    f"{model_name}-finetuned-gtzan",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

kwargs = {
    "dataset_tags": "marsyas/gtzan",
    "dataset": "GTZAN",
    "model_name": f"{model_name}-finetuned-gtzan",
    "finetuned_from": model_id,
    "tasks": "audio-classification",
}

model.save_pretrained('saved_model')
