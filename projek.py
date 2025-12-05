import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

st.set_page_config(page_title="Prediksi Stress Level", layout="wide")

# =========================
# LOAD DATASET
# =========================
st.title("📊 Dashboard Prediksi Stress Level")

url = "https://raw.githubusercontent.com/emiliojuni0r/penambangan-data-ppd/refs/heads/main/StressLevelDataset.csv"
df = pd.read_csv(url)

st.header("Dataset (5 Data Teratas)")
st.dataframe(df.head())

# =========================
# EDA
# =========================
st.subheader("Deskripsi Statistik")
st.write(df.describe())

st.subheader("Distribusi Target Stress Level")
fig1, ax1 = plt.subplots()
df['stress_level'].value_counts().plot(kind='pie', autopct="%.2f%%", ax=ax1)
st.pyplot(fig1)

st.subheader("Heatmap Korelasi Fitur")
corr = df.corr(numeric_only=True)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# =========================
# PREPROCESSING
# =========================
df_X = df.drop(['blood_pressure', 'stress_level'], axis=1)
df_y = df['stress_level']

X = df_X.astype(float)
y = df_y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# =========================
# MODELING FUNCTION
# =========================
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro")
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    st.write(f"### 📌 {model_name}")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")

    # Confusion Matrix
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", ax=ax)
    st.pyplot(fig)

    return model_name, acc, prec, rec, f1

# =========================
# TRAINING MODELS
# =========================
st.header("📌 Modeling & Evaluasi")

models = [
    (LogisticRegression(max_iter=1000), "Logistic Regression"),
    (DecisionTreeClassifier(), "Decision Tree"),
    (RandomForestClassifier(), "Random Forest"),
    (KNeighborsClassifier(n_neighbors=10), "KNN"),
    (AdaBoostClassifier(), "AdaBoost"),
    (XGBClassifier(),
     "XGBoost"),
    (GaussianNB(), "Naive Bayes"),
]

results = []

for model, name in models:
    model_result = train_and_evaluate(model, name)
    results.append(model_result)

# =========================
# PERBANDINGAN MODEL
# =========================
st.header("📊 Perbandingan Model")

df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
st.dataframe(df_results)

fig3 = px.bar(
    df_results,
    x="Model",
    y=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    title="Perbandingan Performa Model"
)
st.plotly_chart(fig3)
