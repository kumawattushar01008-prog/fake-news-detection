from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import os
import joblib
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from googletrans import Translator

app = Flask(__name__)
app.secret_key = "supersecretkey"

translator = Translator()

# -----------------------------
# LOAD SAVED MODEL
# -----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -----------------------------
# NEWS API (Using Environment Variable)
# -----------------------------
API_KEY = os.environ.get("API_KEY")

def get_live_news():
    try:
        url = f"https://newsapi.org/v2/top-headlines?country=in&apiKey={API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data["status"] == "ok":
            return data["articles"][0]["title"]
        else:
            return "Unable to fetch live news"
    except:
        return "Error fetching live news"

# -----------------------------
# LOGIN ROUTE
# -----------------------------
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

# -----------------------------
# HOME ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = ""
    news_text = ""

    if request.method == "POST":

        # If Live News Button Clicked
        if "live" in request.form:
            news_text = get_live_news()
        else:
            news_text = request.form["news"]

        # Translate to English
        try:
            translated = translator.translate(news_text, dest='en')
            news_text = translated.text
        except:
            pass

        # Predict
        news_vector = vectorizer.transform([news_text])
        result = model.predict(news_vector)

        if result[0] == 1:
            prediction = "✅ This News is REAL"
        else:
            prediction = "❌ This News is FAKE"

    return render_template("index.html",
                           prediction=prediction,
                           news=news_text)

# -----------------------------
# LOGOUT
# -----------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)