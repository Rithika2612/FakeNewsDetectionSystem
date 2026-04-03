import os
import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, TextClassificationPipeline
import torch
import requests

# ==============================
# Flask App Setup
# ==============================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "f50665e9214590e20f97d7aa29368d512852869fb5ba185fd176bdc6ded8c518")  # store your key securely


# ==============================
# Route to get latest news
# ==============================
@app.route("/get_latest_news")
def get_latest_news():
    try:
        params = {
            "engine": "google_news",
            "country": "in",   # India news
            "api_key": "f50665e9214590e20f97d7aa29368d512852869fb5ba185fd176bdc6ded8c518"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        data = response.json()
        
        # Extract top 10 news titles and links
        articles = data.get("news_results", [])[:10]
        news_list = []
        for a in articles:
            title = a.get("title") or a.get("snippet") or "loading..."
            link = a.get("link") or "#"
            news_list.append({"title": title, "link": link})

        return jsonify(news_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# User Store (in-memory demo)
# ==============================
users = {}  # {email: password}

# ==============================
# Load Classical ML Models
# ==============================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

LR_PATH = os.path.join(MODEL_DIR, "LrModel.pkl")
DT_PATH = os.path.join(MODEL_DIR, "DtModel.pkl")
RF_PATH = os.path.join(MODEL_DIR, "RfModel.pkl")
GB_PATH = os.path.join(MODEL_DIR, "GBModel.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

with open(LR_PATH, "rb") as f:
    lr_model = pickle.load(f)
with open(DT_PATH, "rb") as f:
    dt_model = pickle.load(f)
with open(RF_PATH, "rb") as f:
    rf_model = pickle.load(f)
with open(GB_PATH, "rb") as f:
    gb_model = pickle.load(f)
with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

# ==============================
# Load Pretrained DistilBERT
# ==============================
BERT_DIR = os.path.join(MODEL_DIR, "bert_finetuned").replace("\\", "/")  # convert backslashes
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_DIR, local_files_only=True)
bert_model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR, local_files_only=True)

bert_model.eval()

def predict_with_bert(text: str):
    try:
        snippet = text[:512]  # truncate long input
        with torch.no_grad():
            inputs = bert_tokenizer(snippet, truncation=True, padding=True, max_length=512, return_tensors="pt")
            outputs = bert_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            bert_prob = probs[0][1].item()  # probability of Real
            bert_pred = int(bert_prob >= 0.5)
        return bert_pred, bert_prob
    except Exception as e:
        return 0, 0.0 
# ==============================
# Routes
# ==============================
@app.route("/") 
def home_page(): 
    if "user" not in session:
         return redirect(url_for("login_page")) 
    return render_template("index.html") 
@app.route("/signup", methods=["GET", "POST"]) 
def signup_page(): 
    error = None 
    if request.method == "POST": 
        email = request.form.get("email", "").strip() 
        password = request.form.get("password", "").strip() 
        if not email or not password: 
            error = "Please enter both email and password." 
        elif email in users: 
            error = "Email already registered." 
        else: 
            users[email] = password 
            session["user"] = email 
            return redirect(url_for("home_page")) 
    return render_template("signup.html", error=error) 
@app.route("/login", methods=["GET", "POST"]) 
def login_page(): 
    error = None 
    if request.method == "POST": 
        email = request.form.get("email", "").strip() 
        password = request.form.get("password", "").strip() 
        if users.get(email) == password: 
            session["user"] = email 
            return redirect(url_for("home_page")) 
        else: 
            error = "Invalid email or password." 
    return render_template("login.html", error=error) 
@app.route("/logout") 
def logout_page(): 
    session.pop("user", None) 
    return redirect(url_for("login_page")) 
@app.route("/index") 
def dashboard_page(): 
    return render_template("index.html") 
@app.route("/predict", methods=["POST"]) 
def predict_route(): 
    try: 
        news_text = request.form.get("news_text", "").strip() 
        if not news_text: return jsonify({"error": "Please provide news text."}), 400        # ========================
        # Classical ML Predictions
        # ========================
        X = vectorizer.transform([news_text])

        # If your classical models support probability estimates
        lr_prob = lr_model.predict_proba(X)[0][1] if hasattr(lr_model, "predict_proba") else float(lr_model.predict(X)[0])
        dt_prob = dt_model.predict_proba(X)[0][1] if hasattr(dt_model, "predict_proba") else float(dt_model.predict(X)[0])
        rf_prob = rf_model.predict_proba(X)[0][1] if hasattr(rf_model, "predict_proba") else float(rf_model.predict(X)[0])
        gb_prob = gb_model.predict_proba(X)[0][1] if hasattr(gb_model, "predict_proba") else float(gb_model.predict(X)[0])

        # Convert probability to binary prediction for reference
        lr_pred = int(lr_prob >= 0.5)
        dt_pred = int(dt_prob >= 0.5)
        rf_pred = int(rf_prob >= 0.5)
        gb_pred = int(gb_prob >= 0.5)

        bert_pred, bert_prob = predict_with_bert(news_text)
        # ========================
        # Weighted Ensemble
        # ========================
        # Weights for each model
        weights = {
            "LogisticRegression": 1,
            "DecisionTree": 1,
            "RandomForest": 2,
            "GradientBoosting": 2,
            "BERT": 5,
        }

        # Weighted probability
        weighted_sum = (
            lr_prob * weights["LogisticRegression"] +
            dt_prob * weights["DecisionTree"] +
            rf_prob * weights["RandomForest"] +
            gb_prob * weights["GradientBoosting"] +
            bert_prob * weights["BERT"] 
        )
        final_prob = weighted_sum / sum(weights.values())
        final_result = "Real" if final_prob >= 0.5 else "Fake"
        
        return jsonify({
            "LogisticRegression": lr_pred,
            "DecisionTree": dt_pred,
            "RandomForest": rf_pred,
            "GradientBoosting": gb_pred,
            "BERT": bert_pred,
            "Final_probability": round(final_prob, 3),
            "FinalResult": final_result
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
