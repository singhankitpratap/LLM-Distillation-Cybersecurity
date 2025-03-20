from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from transformers import DataCollatorWithPadding

# Load dataset
dataset = load_from_disk("data/phishing_dataset")

# Load tokenizer and model
model_checkpoint = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Define label mappings
id2label = {0: "Safe", 1: "Not Safe"}
label2id = {"Safe": 0, "Not Safe": 1}

# Initialize model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# Print model parameters
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Freeze base model parameters
for name, param in model.base_model.named_parameters():
    param.requires_grad = False

# Unfreeze pooler layers
for name, param in model.base_model.named_parameters():
    if "pooler" in name:
        param.requires_grad = True

# Print model parameters after modification
for name, param in model.named_parameters():
    print(name, param.requires_grad)

# Define tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True)

# Apply tokenization to dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Initialize data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load evaluation metrics
accuracy_metric = evaluate.load("accuracy")
auc_metric = evaluate.load("roc_auc")

# Define metric computation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Convert logits to probabilities
    probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    pos_probs = probs[:, 1]

    # Compute AUC score
    auc = np.round(auc_metric.compute(prediction_scores=pos_probs, references=labels)['roc_auc'], 3)

    # Compute accuracy
    preds = np.argmax(logits, axis=1)
    acc = np.round(accuracy_metric.compute(predictions=preds, references=labels)['accuracy'], 3)

    return {"Accuracy": acc, "AUC": auc}

# Define training parameters
learning_rate = 2e-4
batch_size = 8
epochs = 10

training_args = TrainingArguments(
    output_dir="bert-phishing-classifier_teacher",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to=None
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate model on validation set
predictions = trainer.predict(tokenized_dataset["validation"])

# Compute metrics
logits, labels = predictions.predictions, predictions.label_ids
metrics = compute_metrics((logits, labels))
print(metrics)

# Save model and tokenizer
save_path = "bert-phishing-classifier_teacher"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
