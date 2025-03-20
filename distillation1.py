from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertForSequenceClassification, DistilBertConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set device
device = torch.device("cuda")

# Load dataset
dataset = load_from_disk("data/phishing_dataset")

# Load teacher model and tokenizer
teacher_model_path = "/content/bert-phishing-classifier_teacher"
tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_model_path).to(device)

# Define student model configuration (reduced number of heads and layers)
student_config = DistilBertConfig(n_heads=8, n_layers=4)
student_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", config=student_config
).to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")

    return accuracy, precision, recall, f1

# Compute loss function
def distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):
    soft_targets = nn.functional.softmax(teacher_logits / temperature, dim=1)
    student_soft = nn.functional.log_softmax(student_logits / temperature, dim=1)

    distill_loss = nn.functional.kl_div(student_soft, soft_targets, reduction="batchmean") * (temperature**2)
    hard_loss = nn.CrossEntropyLoss()(student_logits, true_labels)

    return alpha * distill_loss + (1.0 - alpha) * hard_loss

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
epochs = 5
temperature = 2.0
alpha = 0.5

# Optimizer
optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

# Create dataloaders
train_loader = DataLoader(tokenized_dataset["train"], batch_size=batch_size)
test_loader = DataLoader(tokenized_dataset["test"], batch_size=batch_size)

# Training loop
student_model.train()
for epoch in range(epochs):
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            teacher_logits = teacher_model(input_ids, attention_mask=attention_mask).logits

        student_logits = student_model(input_ids, attention_mask=attention_mask).logits
        loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} completed with loss: {loss.item()}")

    # Evaluate models
    teacher_acc, teacher_prec, teacher_rec, teacher_f1 = evaluate_model(teacher_model, test_loader)
    student_acc, student_prec, student_rec, student_f1 = evaluate_model(student_model, test_loader)

    print(f"Teacher (test) - Accuracy: {teacher_acc:.4f}, Precision: {teacher_prec:.4f}, Recall: {teacher_rec:.4f}, F1 Score: {teacher_f1:.4f}")
    print(f"Student (test) - Accuracy: {student_acc:.4f}, Precision: {student_prec:.4f}, Recall: {student_rec:.4f}, F1 Score: {student_f1:.4f}")
    print("\n")

# Save student model
student_model_path = "/content/bert-phishing-classifier_student"
student_model.save_pretrained(student_model_path)

# Validation evaluation
validation_loader = DataLoader(tokenized_dataset["validation"], batch_size=8)

teacher_acc, teacher_prec, teacher_rec, teacher_f1 = evaluate_model(teacher_model, validation_loader)
student_acc, student_prec, student_rec, student_f1 = evaluate_model(student_model, validation_loader)

print(f"Teacher (validation) - Accuracy: {teacher_acc:.4f}, Precision: {teacher_prec:.4f}, Recall: {teacher_rec:.4f}, F1 Score: {teacher_f1:.4f}")
print(f"Student (validation) - Accuracy: {student_acc:.4f}, Precision: {student_prec:.4f}, Recall: {student_rec:.4f}, F1 Score: {student_f1:.4f}")
