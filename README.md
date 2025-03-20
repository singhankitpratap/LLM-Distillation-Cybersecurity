# LLM-Distillation-Cybersecurity

This project demonstrates how **Large Language Models (LLMs)** can be optimized for **phishing URL detection** in cybersecurity. Using **Model Distillation** and **Quantization**, we achieve a **51% reduction in model size** while maintaining (or improving) accuracy.

---

## Workflow
1. **Prepare Dataset** → `data.py`
2. **Train Teacher Model (BERT)** → `teacher_training.py`
3. **Distill to Smaller Model (DistilBERT)** → `distillation.py`
4. **Quantize Model for Efficiency** → `quantization.py`

---

## Files
| File                 | Description |
|----------------------|-------------|
| `data.py`           | Prepares phishing dataset for training. |
| `teacher_training.py` | Trains a **BERT-based model** for phishing detection. |
| `distillation.py`   | Uses **Knowledge Distillation** with **Hugging Face Transformers & PyTorch** to train a **DistilBERT student model**. |
| `quantization.py`   | Uses **BitsAndBytes 4-bit quantization (NF4)** to optimize the student model for efficient deployment. |

---

## Key Concepts
### **Model Distillation**
- Implemented using **Hugging Face Transformers & PyTorch**.
- A **large teacher model** (BERT) guides a **smaller student model** (DistilBERT) using **soft labels** instead of direct ground truth.
- **Reduces model size** while **retaining high accuracy**.

### **Model Quantization**
- Implemented using **BitsAndBytes library** from Hugging Face.
- Uses **4-bit Normal Float (NF4) quantization** to **reduce memory footprint** and **speed up inference**.
- Reduces precision while keeping performance **almost identical**.

---

## How to Run
1️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```
2️⃣ **Run scripts in order**  
```bash
python data.py
python teacher_training.py
python distillation.py
python quantization.py
```

---

## 📌 Libraries Used
- **[Transformers](https://huggingface.co/docs/transformers/index)** → Model training & distillation  
- **[Datasets](https://huggingface.co/docs/datasets/index)** → Efficient dataset handling  
- **[Torch](https://pytorch.org/)** → Deep learning framework for training  
- **[BitsAndBytes](https://huggingface.co/blog/4bit-transformers-bitsandbytes)** → 4-bit quantization for optimized inference  
- **[scikit-learn](https://scikit-learn.org/)** → Evaluation metrics  
