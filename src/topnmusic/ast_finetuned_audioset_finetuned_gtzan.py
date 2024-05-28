#!/bin/env python3

import os
import sys

import accelerate
from datasets import Audio
from datasets import load_dataset
import evaluate
import librosa
import librosa.display
import numpy as np
from transformers import ASTFeatureExtractor
from transformers import ASTForAudioClassification
from transformers import ASTConfig
from transformers import models
from transformers import AutoConfig
from transformers import Trainer
from transformers import TrainingArguments
import matplotlib.pyplot as plt


# learning_rate = 5e-5
# batch_size = 8

mel_spectrograms_db = []

# load in hyperparams from batch script
learning_rate = float(sys.argv[1])
batch_size = int(sys.argv[2])
learning_rate_str = str(learning_rate).replace('-', '_') 

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

    # Get mel spectrograms from the inputs
    for audio_array in audio_arrays:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_array,
            sr=feature_extractor.sampling_rate,
            n_fft=feature_extractor.feature_extractor.spectrogram_config.n_fft,
            hop_length=feature_extractor.feature_extractor.spectrogram_config.hop_length,
            n_mels=feature_extractor.feature_extractor.spectrogram_config.feat_extract_cfg.num_feature_bins,
            fmin=feature_extractor.feature_extractor.spectrogram_config.feat_extract_cfg.f_min,
            fmax=feature_extractor.feature_extractor.spectrogram_config.feat_extract_cfg.f_max
        )
        # Convert mel spectrogram to decibel scale
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        # Append to the global variable
        mel_spectrograms_db.append(mel_spectrogram_db)

    return inputs

gtzan_encoded = gtzan.map(
    preprocess_function,
    remove_columns=["audio", "file"],
    batched=True,
    batch_size=100,
    num_proc=1,
)

# Plot mel spectrograms
for mel_spectrogram in mel_spectrogram_db[:3]:  # Plot first 3 examples
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram, sr=feature_extractor.sampling_rate, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()

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

# Calculate the sizes of train, val, and test sets
train_size = len(train_val_split["train"])
val_size = len(train_val_split["test"])
test_size = len(test_split["test"])

# Data to plot
labels = 'Train', 'Validation', 'Test'
sizes = [train_size, val_size, test_size]
colors = ['gold', 'yellowgreen', 'lightcoral']
explode = (0.1, 0, 0)  # explode 1st slice

# Plot
plt.figure(figsize=(7, 7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Dataset Split')
plt.show()

# load in the pre-trained model configuration
model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
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
gradient_accumulation_steps = 1
num_train_epochs = 50  # will keep model at epoch with highest accuracy

# save model checkpoints during training
checkpoint_folder_name = f'checkpoint_model_lr_{learning_rate_str}_bs_{batch_size}'
checkpoint_folder_path = os.path.join('.', 'checkpoints', checkpoint_folder_name)
os.mkdir(checkpoint_folder_path)

training_args = TrainingArguments(
    output_dir=checkpoint_folder_path,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    logging_steps=5,
    save_total_limit=1,
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
    train_dataset=train_val_split["train"],
    eval_dataset=train_val_split["test"],
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

# save the model and its configuration
save_folder_name = f'saved_model_lr_{learning_rate_str}_bs_{batch_size}'
save_folder_path = os.path.join('.', 'end_models', save_folder_name)
os.mkdir(save_folder_path)

model.save_pretrained(save_folder_path)
