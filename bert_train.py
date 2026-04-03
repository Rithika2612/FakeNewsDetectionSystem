import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# =============================
# 1. Load Dataset
# =============================
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0  # Fake = 0
true["label"] = 1  # Real = 1

df = pd.concat([fake, true]).sample(frac=1, random_state=42).reset_index(drop=True)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# =============================
# 2. Tokenizer
# =============================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# =============================
# 3. Model
# =============================
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# =============================
# 4. TrainingArguments
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"

training_args = TrainingArguments(
    output_dir="./models/bert_finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,       
    logging_dir="./logs",
    logging_steps=200,
    save_steps=500,
    save_total_limit=1,
)

# =============================
# 5. Metrics function
# =============================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# =============================
# 6. Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =============================
# 7. Train
# =============================
trainer.train()

# =============================
# 8. Evaluate
# =============================
metrics = trainer.evaluate()
print("Evaluation metrics:", metrics)

# =============================
# 9. Save model & tokenizer
# =============================
trainer.save_model("./models/bert_finetuned")
tokenizer.save_pretrained("./models/bert_finetuned")

print("✅ DistilBERT fine-tuning complete! Model saved to ./models/bert_finetuned")

