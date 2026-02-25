from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import os
import joblib
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from googletrans import Translator

app = Flask(__name__)
app.secret_key = "supersecretkey"

translator = Translator()

# ------------------------------
# LOAD & TRAIN MODEL (OR LOAD SAVED)
# ------------------------------
model_file = "fake_news_model.pkl"
vectorizer_file = "vectorizer.pkl"

if os.path.exists(model_file) and os.path.exists(vectorizer_file):
    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    accuracy = 95  # Placeholder if loading saved model
else:
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test, pred) * 100, 2)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)

# ------------------------------
# NEWS API CONFIG
# ------------------------------
API_KEY = "YOUR_NEWSAPI_KEY"

def get_live_news():
    url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    return data["articles"][0]["title"]

# ------------------------------
# LOGIN ROUTE
# ------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return "Invalid credentials"

    return render_template("login.html")

# ------------------------------
# HOME ROUTE
# ------------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = ""

    if request.method == "POST":

        if "live" in request.form:
            news = get_live_news()
        else:
            news = request.form["news"]

        # Translate to English
        try:
            translated = translator.translate(news, dest='en')
            news = translated.text
        except:
            pass

        news_vector = vectorizer.transform([news])
        result = model.predict(news_vector)

        if result[0] == 1:
            prediction = "✅ This News is REAL"
        else:
            prediction = "❌ This News is FAKE"

    return render_template("index.html",
                           prediction=prediction,
                           accuracy=accuracy)

# ------------------------------
# LOGOUT
# ------------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ------------------------------
# RUN APP
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)