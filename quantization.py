from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BitsAndBytesConfig

# Set device
device = torch.device("cuda")

# Load dataset
dataset = load_from_disk("data/phishing_dataset")

# Load tokenizer and student model
teacher_model_path = "/content/bert-phishing-classifier_teacher"
student_model_path = "/content/bert-phishing-classifier_student"

tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
model = AutoModelForSequenceClassification.from_pretrained(student_model_path).to(device)

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

# Create validation data loader
validation_loader = DataLoader(tokenized_dataset["validation"], batch_size=128)

# Evaluate original model before quantization
accuracy, precision, recall, f1 = evaluate_model(model, validation_loader)
print("Pre-quantization Performance")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# Load student model with 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

quantized_model = AutoModelForSequenceClassification.from_pretrained(
    student_model_path, device_map=device, quantization_config=quant_config
)

# Evaluate quantized model
quantized_accuracy, quantized_precision, quantized_recall, quantized_f1 = evaluate_model(quantized_model, validation_loader)

print("Post-quantization Performance")
print(f"Accuracy: {quantized_accuracy:.4f}, Precision: {quantized_precision:.4f}, Recall: {quantized_recall:.4f}, F1 Score: {quantized_f1:.4f}")
