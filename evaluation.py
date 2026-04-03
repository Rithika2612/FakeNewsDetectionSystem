# evaluation.py
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ==========================
# 1. Load Dataset
# ==========================
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0  # Fake
true_df["label"] = 1  # Real

df = pd.concat([fake_df, true_df], axis=0).sample(frac=1, random_state=42)
X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 2. Load Classical Models
# ==========================
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/LrModel.pkl", "rb") as f:
    lr_model = pickle.load(f)

with open("models/DtModel.pkl", "rb") as f:
    dt_model = pickle.load(f)

with open("models/RfModel.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/GBModel.pkl", "rb") as f:
    gb_model = pickle.load(f)

# Vectorize test data for classical models
X_test_vec = vectorizer.transform(X_test)

# ==========================
# 3. Load Fine-tuned BERT
# ==========================
BERT_DIR = "models/bert_finetuned"
tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_DIR, local_files_only=True)
bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR, local_files_only=True)
bert_model.eval()

def predict_with_bert_batch(texts, batch_size=8, max_len=256):
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            list(batch),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        ).to("cpu")
        with torch.no_grad():
            outputs = bert_model(**inputs)
            batch_preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
    return preds

# ==========================
# 4. Compute Predictions
# ==========================
results = {
    "Logistic Regression": lr_model.predict(X_test_vec),
    "Decision Tree": dt_model.predict(X_test_vec),
    "Random Forest": rf_model.predict(X_test_vec),
    "Gradient Boosting": gb_model.predict(X_test_vec),
    "BERT": predict_with_bert_batch(X_test)
}

# ==========================
# 5. Confusion Matrices
# ==========================
for name, preds in results.items():
    print(f"\n{name}:\n")
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# ==========================
# 6. Performance Metrics
# ==========================
metrics = {"Accuracy": {}, "Precision": {}, "Recall": {}, "F1-score": {}}

for name, preds in results.items():
    metrics["Accuracy"][name] = accuracy_score(y_test, preds)
    metrics["Precision"][name] = precision_score(y_test, preds, zero_division=0)
    metrics["Recall"][name] = recall_score(y_test, preds, zero_division=0)
    metrics["F1-score"][name] = f1_score(y_test, preds, zero_division=0)

metrics_df = pd.DataFrame(metrics)

# ==========================
# 7. Detailed Combined Bar Chart
# ==========================
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")

# Plot each metric as separate bar with spacing
bar_width = 0.15
x = range(len(metrics_df.index))
for i, metric in enumerate(metrics_df.columns):
    plt.bar(
        [p + bar_width*i for p in x],
        metrics_df[metric].values,
        width=bar_width,
        label=metric
    )

plt.xticks([p + 1.5*bar_width for p in x], metrics_df.index, rotation=30)
plt.ylim(0.7, 1.0)  # show minute differences clearly
plt.ylabel("Score")
plt.title("Detailed Performance Metrics Comparison of Models")
plt.legend(title="Metrics")
plt.tight_layout()
plt.savefig("detailed_performance_comparison.png")
plt.show()

print("✅ Evaluation completed. Charts saved as detailed_performance_comparison.png")



