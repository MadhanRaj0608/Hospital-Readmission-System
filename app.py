import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import re
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import shap
from streamlit_lottie import st_lottie
from openai import OpenAI

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
client1 = OpenAI(
        api_key="AIzaSyBnyIj10Y5THLtnZIYzC1-RC1SAfo486M4",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

# --- MongoDB Imports ---
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime

# --- MongoDB Connection ---
# Replace <system140> with your actual password.
uri = "mongodb+srv://Lakshanya:system140@cluster0.3m28r7d.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

DB_NAME = "hospital_predictions"
COLLECTION_NAME = "readmission_predictions"

try:
    client = MongoClient(uri, server_api=ServerApi('1'))
    db = client[DB_NAME]
    predictions_collection = db[COLLECTION_NAME]
    client.admin.command('ping')
    #st.success("Connected to MongoDB successfully!")
except Exception as e:
    st.error(f"Could not connect to MongoDB: {e}")
    client = None


# ----------- Modern CSS Styling -----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f8f9fa;
    }

    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1200px;
        margin: auto;
    }

    h1, h2, h3, h4 {
        font-weight: 700;
        color: #ffffff;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }

    .card {
        background: rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 25px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.15);
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 12px;
        border: none;
        color: white;
        font-weight: 600;
        padding: 12px 32px;
        transition: all 0.3s ease;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    section[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: bold;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #4e54c8, #8f94fb);
        transform: scale(1.05);
        cursor: pointer;
        box-shadow: 0px 6px 14px rgba(0,0,0,0.3);
    }

    .stCheckbox label {
        color: #eee;
        font-weight: 500;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c5364, #203a43, #0f2027);
        color: white;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3 {
        color: white;
    }

    .css-1d391kg {  /* form labels */
        font-weight: 600;
        font-size: 16px;
        color: #f1f1f1 !important;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 22px;
        font-weight: 700;
        color: #00e6e6;
    }
</style>
""", unsafe_allow_html=True)


# ---------- Helper Functions (unchanged) ----------
# ... (all your helper functions here, unchanged) ...
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None


def clean_column_names(columns):
    return [col.replace("[", "_").replace("]", "_").replace("<", "").replace(">", "") for col in columns]


def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

        
def load_json(filename):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {}


def save_model(model, filename):
    joblib.dump(model, filename)


def load_model(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    return None


def reset_models():
    saved_models = load_json("trained_models.json")
    for mid in saved_models.keys():
        model_file = f"{mid}.pkl"
        if os.path.exists(model_file):
            os.remove(model_file)
    if os.path.exists("trained_models.json"):
        os.remove("trained_models.json")


def preprocess_data(df, target_col, return_raw=False):
    raw_input = None
    if return_raw:
        # Save the first row (or modify if you want a specific row)
        raw_input = df.iloc[0].to_dict()

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        else:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # One-hot encode categorical columns
    encoded_df = pd.get_dummies(df.drop(columns=[target_col]))
    encoded_df.columns = clean_column_names(encoded_df.columns)

    # Scale numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col != target_col and col in encoded_df.columns:
            scaler = StandardScaler()
            encoded_df[col] = scaler.fit_transform(df[[col]])

    # Encode target
    target = df[target_col]
    if target.dtype == 'object':
        le = LabelEncoder()
        target = le.fit_transform(target)
    else:
        le = None

    if return_raw:
        return encoded_df, target, le, raw_input
    else:
        return encoded_df, target, le



def train_and_eval_model(model_name, X_train, y_train, X_test, y_test):
    X_train.columns = clean_column_names(X_train.columns)
    X_test.columns = clean_column_names(X_test.columns)
    if model_name == "Balanced Random Forest":
        model = BalancedRandomForestClassifier(random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif model_name == "SVM":
        model = SVC(probability=True, random_state=42)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    else:
        st.error("Unsupported model")
        return None, None
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    metrics = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds, average='weighted'),
        "Precision": precision_score(y_test, preds, average='weighted'),
        "Recall": recall_score(y_test, preds, average='weighted'),
        "Confusion Matrix": confusion_matrix(y_test, preds).tolist()
    }
    return model, metrics
    

    


def display_metrics(metrics):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Model Performance Metrics")

    # High-level metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['Accuracy'] * 100:.2f}%")
    c2.metric("F1 Score", f"{metrics['F1 Score'] *100:.4f}%")
    c3.metric("Precision", f"{metrics['Precision']*100:.4f}%")
    c4.metric("Recall", f"{metrics['Recall']*100:.4f}%")

    # Confusion Matrix Heatmap
    st.markdown("#### Confusion Matrix")
    cm = np.array(metrics['Confusion Matrix'])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=ax,
                xticklabels=["No Readmission", "Readmission"],
                yticklabels=["No Readmission", "Readmission"])
    st.pyplot(fig)

    # Textual Explanation right after graph
    st.markdown("### Confusion Matrix Details")
    st.write(f"✅ **True Positives (TP):** {tp} → Correctly predicted readmissions")
    st.write(f"✅ **True Negatives (TN):** {tn} → Correctly predicted non-readmissions")
    st.write(f"❌ **False Positives (FP):** {fp} → Predicted readmission but actually not")
    st.write(f"❌ **False Negatives (FN):** {fn} → Predicted no readmission but actually yes")
    st.markdown("### Performance Metrics")

    st.write(f"**Accuracy:** {metrics['Accuracy']:.2%} → Overall percentage of predictions that were correct "
            "(both readmission and non-readmission).")

    st.write(f"**Precision (weighted):** {metrics['Precision']:.2%} → Out of all patients predicted as readmitted, "
            "this is the percentage that were actually correct. High precision means fewer false alarms.")

    st.write(f"**Recall (weighted):** {metrics['Recall']:.2%} → Out of all patients who were truly readmitted, "
            "this is the percentage that the model successfully identified. High recall means fewer missed cases.")

    st.write(f"**F1 Score (weighted):** {metrics['F1 Score']:.2%} → Balance between Precision and Recall. "
            "It’s useful when both false positives and false negatives are important.")



    st.markdown("</div>", unsafe_allow_html=True)
    


def plot_eda(df):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🔍 Dataset Statistics")
    st.write(f"Shape: {df.shape}")
    st.write(f"Null Values:\n{df.isnull().sum()}")
    st.write(f"Unique Values per Column:\n{df.nunique()}")
    st.markdown("### 📄 Sample Data")
    st.dataframe(df.head())
    st.markdown("### 📈 Distribution Plots")
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax, color="#4e79a7")
        st.pyplot(fig)
    st.markdown("### 🔗 Correlation Heatmap")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)        
        st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------ Main App Flow (rest of your code remains same) --------------------

# Load cool AI Lottie animation
lottie_ai = load_lottie_url("https://assets3.lottiefiles.com/packages/lf20_x62chJ.json")


# Authentication data
admin_username = "admin"
admin_password = "admin123"
user_username = "user"
user_password = "user123"


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "role" not in st.session_state:
    st.session_state.role = None


def login():
    # Use columns to put logo and title side by side
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            st.image("image.jpg", width=140)  # Adjust if file path/name different
        except Exception:
            st.warning("Logo image not found.")
    with col2:
        st.markdown(
            "<h1 style='font-weight:bold; margin-bottom:0;'>Health Radar</h1>"
            "<h3 style='margin-top:-10px;'>Predict.. Prevent.. Protect..</h3>",
            unsafe_allow_html=True
        )
    st_lottie(lottie_ai, speed=1, height=150)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == admin_username and password == admin_password:
            st.session_state.logged_in = True
            st.session_state.role = "admin"
        elif username == user_username and password == user_password:
            st.session_state.logged_in = True
            st.session_state.role = "user"
        else:
            st.error("Invalid username or password")


def admin_dashboard():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Hospital Data Administrator")
    st.markdown(
        """
        <a href='https://charts.mongodb.com/charts-healthradar-uvfogsk/dashboards/5663a6a7-8f31-4a25-8688-676a5910a419' target='_blank'>
            <button style='
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 12px;
                border: none;
                color: white;
                font-weight: 600;
                padding: 12px 32px;
                transition: all 0.3s ease;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
                cursor: pointer;
                margin-bottom: 20px;
            '>
                View Dashboard
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
    
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    data_changed = False
    if uploaded_file:
        if 'uploaded_name' not in st.session_state or st.session_state.uploaded_name != uploaded_file.name:
            st.session_state.uploaded_name = uploaded_file.name
            data_changed = True
            reset_models()
            st.session_state['dataset'] = None
            st.session_state['X_processed'] = None
            st.session_state['y_processed'] = None
            st.session_state['label_enc'] = None
            st.session_state['target_col'] = None
            st.session_state['feature_descriptions'] = {}
            st.success("New dataset detected. Previous trained models cleared.")
        if st.session_state['dataset'] is None:
            df = pd.read_csv(uploaded_file)
            st.session_state['dataset'] = df
            st.success(f"Dataset uploaded ({uploaded_file.name})")


    if 'dataset' in st.session_state and st.session_state['dataset'] is not None:
        df = st.session_state['dataset']
        tab1, tab2, tab3 = st.tabs(["Data & EDA", "Feature Descriptions", "Train & Manage Models"])
        with tab1:
            target_col = st.selectbox("Select Target Column", df.columns.tolist())
            st.session_state['target_col'] = target_col
            show_eda = st.checkbox("Show Data Exploration (EDA)")
            if show_eda:
                plot_eda(df)


            if st.button("Preprocess Dataset"):
                X_processed, y_processed, label_enc = preprocess_data(df, target_col)
                st.session_state['X_processed'] = X_processed
                st.session_state['y_processed'] = y_processed
                st.session_state['label_enc'] = label_enc
                st.success("Dataset preprocessed")


        with tab2:
            st.markdown("### Feature Descriptions")
            desc_file = "feature_descriptions.json"
            descriptions = load_json(desc_file)
            if "feature_descriptions" not in st.session_state:
                st.session_state['feature_descriptions'] = descriptions if descriptions else {}
            descs = {}
            for col in df.columns:
                val = st.text_input(f"Description for '{col}'", value=st.session_state['feature_descriptions'].get(col, ""))
                descs[col] = val
            if st.button("Save Feature Descriptions"):
                save_json(descs, desc_file)
                st.session_state['feature_descriptions'] = descs
                st.success("Saved feature descriptions")


        with tab3:
            model_choices = ["Balanced Random Forest", "Decision Tree", "XGBoost", "SVM", "Naive Bayes"]
            selected_model = st.selectbox("Select model to train", model_choices)
            if st.session_state.get('X_processed') is not None and st.session_state.get('y_processed') is not None:
                if st.button("Train Model"):
                    X_train, X_test, y_train, y_test = train_test_split(
                        st.session_state['X_processed'], st.session_state['y_processed'], test_size=0.2, random_state=42)
                    model, metrics = train_and_eval_model(selected_model, X_train, y_train, X_test, y_test)
                    if model:
                        saved_models = load_json("trained_models.json")
                        if saved_models == {}:
                            saved_models = {}
                        model_id = f"{selected_model}_{len(saved_models)+1}"
                        joblib.dump(model, f"{model_id}.pkl")
                        saved_models[model_id] = {
                            "model_name": selected_model,
                            "metrics": metrics,
                            "active": False
                        }
                        save_json(saved_models, "trained_models.json")
                        st.success(f"Model trained and saved as {model_id}")
                        display_metrics(metrics)


            st.markdown("---")
            st.markdown("### Model Manager")
            saved_models = load_json("trained_models.json")
            if saved_models:
                for mid, mval in saved_models.items():
                    st.write(f"*Model ID:* {mid} — {mval['model_name']}")
                    display_metrics(mval['metrics'])
                    if not mval.get('active', False):
                        if st.button(f"Mark {mid} as Active"):
                            for k in saved_models.keys():
                                saved_models[k]['active'] = False
                            saved_models[mid]['active'] = True
                            save_json(saved_models, "trained_models.json")
                            st.success(f"{mid} marked as active model — logging out and redirecting to User Login…")
                            st.session_state.logged_in = False
                            st.session_state.role = None
                            st.session_state["preferred_user_login"] = True
                            st.stop()
            else:
                st.info("No trained models yet.")
    else:
        st.info("Upload a dataset to start.")
    st.markdown("</div>", unsafe_allow_html=True)





def user_dashboard():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Patient Risk Evaluation Interface")
    df = st.session_state.get('dataset', None)
    if df is None:
        st.warning("No dataset found. Please ask admin to upload the dataset.")
        return


    saved_models = load_json("trained_models.json")
    active_model_id = None
    for mid, mval in saved_models.items():
        if mval.get("active", False):
            active_model_id = mid
            break
    if active_model_id is None:
        st.warning("No active model available. Please ask admin to activate a model.")
        return


    model = load_model(f"{active_model_id}.pkl")
    target_col = st.session_state.get('target_col', None)
    if target_col is None:
        st.warning("Target column not defined. Please ask admin to select target column.")
        return


    desc_file = "feature_descriptions.json"
    descriptions = load_json(desc_file)


    st.subheader("Input Features")
    user_input = {}
    for col in df.columns:
        if col == target_col:
            continue
        desc = descriptions.get(col, "")
        dtype = df[col].dtype
        label = f"{col} - {desc}" if desc else col
        if np.issubdtype(dtype, np.number):
            val = st.number_input(label, value=float(df[col].median()))
            user_input[col] = val
        else:
            options = df[col].dropna().unique().tolist()
            val = st.selectbox(label, options)
            user_input[col] = val


    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        X_train_cols = st.session_state['X_processed'].columns
        input_encoded = pd.get_dummies(input_df)
        for col in X_train_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[X_train_cols]
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        for col in numeric_cols:
            if col != target_col and col in input_encoded.columns:
                med = df[col].median()
                std = df[col].std() if df[col].std() > 0 else 1
                input_encoded[col] = (input_encoded[col] - med) / std


        pred = model.predict(input_encoded)[0]
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_encoded)[0][1]


        label_enc = st.session_state.get('label_enc', None)
        if label_enc:
            pred_label = label_enc.inverse_transform([pred])[0]
        else:
            pred_label = pred


        st.success(f"Prediction: {target_col} = {pred_label}")
        if proba is not None:
            st.info(f"Probability: {proba*100:.2f}%")
        
        # --- NEW: Store the user's data and prediction in MongoDB ---
        if client is not None:
            try:
                prediction_record = {
                    "prediction_date": datetime.now(),
                    "model_used": saved_models[active_model_id]['model_name'],
                    "predicted_readmission": int(pred_label) if isinstance(pred_label, (np.integer, np.int64, np.int32)) else pred_label,
                    "dataset_source": st.session_state.get('uploaded_name', 'Unknown'),
                    "input_data": user_input,
                    "probabilities": float(proba) if proba is not None else "N/A"
                }
                predictions_collection.insert_one(prediction_record)
                st.success("User prediction data saved to MongoDB!")
            except Exception as e:
                st.error(f"Failed to save prediction to MongoDB: {e}")


        # Store encoded input and prediction info in session for explanation reuse
        st.session_state['last_input_encoded'] = input_encoded
        st.session_state['last_raw_input'] = user_input
        st.session_state['last_prediction'] = pred
        st.session_state['last_model'] = model
        st.session_state['last_model_name'] = saved_models[active_model_id]['model_name']


    # Explain Prediction button (separate, so user clicks after a prediction)
    if st.button("Explain Prediction"):
        if 'last_input_encoded' not in st.session_state or 'last_model' not in st.session_state:
            st.warning("Please make a prediction first.")
        else:
            with st.spinner("Generating explanation..."):
                try:
                    model = st.session_state['last_model']
                    input_encoded = st.session_state['last_input_encoded']
                    model_name = st.session_state['last_model_name']

                    # --- Select SHAP Explainer ---
                    if model_name in ["Balanced Random Forest", "Decision Tree", "XGBoost"]:
                        explainer = shap.TreeExplainer(model)
                    else:
                        sample_bg = shap.sample(st.session_state['X_processed'], 100)
                        explainer = shap.KernelExplainer(model.predict_proba, sample_bg)

                    shap_values = explainer.shap_values(input_encoded)
                    # --- Get predicted class first ---
                    pred_class = int(model.predict(input_encoded)[0])
                    # --- Handle binary classification shap output ---
                    if isinstance(shap_values, list):
                        pred_class = int(model.predict(input_encoded))
                        shap_vals = shap_values[pred_class][0]  # explanation for predicted class
                    else:
                        shap_vals = shap_values[0]

                    feature_names = list(input_encoded.columns)

                    # --- Most Influential Feature ---
                    # 🐞 This is the corrected section
                    abs_vals = np.abs(shap_vals[0]) 
                    max_idx = abs_vals.argmax()
                    most_feat = input_encoded.columns[max_idx]
                    contribution = shap_vals[0][max_idx]


                    
                    # ✅ Ensure contribution is a scalar
                    if isinstance(contribution, np.ndarray):
                        contribution = contribution.item()

                    # --- Show result ---
                    st.markdown("### 🏆 Most Influential Feature")
                    st.write(f"{most_feat}** contributed the most to this prediction "
                             f"(SHAP value = {contribution:.4f})")

                    # --- Simple bar plot ---
                    plt.figure(figsize=(4,2))
                    plt.barh([most_feat], [contribution], color="skyblue")
                    plt.xlabel("SHAP Value (Impact)")
                    st.pyplot(plt.gcf())
                    plt.clf()

                    # --- Textual Explanations ---
                    st.markdown("### Key Factors Influencing Prediction")
                    raw_input = st.session_state.get('last_raw_input', {})
                    explanations = generate_text_explanation(input_encoded, shap_vals, feature_names, raw_input, top_n=5)
                    for msg in explanations:
                        st.write(msg)

                    

                    # --- AI Explanation via Gemini ---
                    st.markdown("### Narrative Explanation")

                    # Build a nice SHAP text (top 5 features)
                    shap_pairs = []
                    for i in abs_vals.argsort()[-5:][::-1]:
                        feat_name = feature_names[i]
                        shap_val = shap_vals[0][i]

                        # Map to raw input
                        if "__" in feat_name:  # categorical
                            orig_col, category = feat_name.split("__", 1)
                            if raw_input.get(orig_col) == category:
                                feat_val = category
                            else:
                                continue
                        else:  # numeric
                            feat_val = raw_input.get(feat_name, "N/A")

                        shap_pairs.append(f"- {feat_name}: value={feat_val}, SHAP={shap_val:.4f}")
                    shap_text = "\n".join(shap_pairs)


                    # Safer probability (some models may not have predict_proba)
                    proba = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_encoded)[0][pred_class]

                    prompt = f"""
                    You are an AI medical assistant helping to explain a machine learning prediction
                    about whether a patient will be readmitted to the hospital.

                    Prediction: {pred_class} (Probability: {proba*100:.2f}% if available)

                    Here are the SHAP feature contributions for this patient:
                    {shap_text}

                    Explain clearly in 5–7 sentences:
                    1. Highlight the top 2–3 most influential features (with their values).
                    2. Explain how they increase or decrease the likelihood of readmission.
                    3. Keep it patient-friendly (avoid technical jargon).
                    4. End with a short summary about what the model "thinks" regarding this case.
                    """

                    response = client1.chat.completions.create(
                        model="gemini-2.5-flash",
                        messages=[
                            {"role": "system", "content": "You are a trusted doctor assistant. Be concise, clear, and medically accurate."},
                            {"role": "user", "content": prompt}
                        ]
                    )

                    ai_expl = response.choices[0].message.content
                    st.write(ai_expl)


                except Exception as e:
                    st.error(f"Failed to generate explanation: {e}")



    st.markdown("</div>", unsafe_allow_html=True)


import numpy as np

def generate_text_explanation(input_data, shap_values, feature_names, raw_input, top_n=5):
    """
    Generate human-readable explanations for a single prediction based on SHAP values.
    Uses raw_input values instead of encoded/scaled ones.
    """
    shap_values = np.array(shap_values).flatten()

    abs_shap = [(feature, abs(float(shap_values[feature_names.index(feature)])))
                 for feature in feature_names]
    abs_shap.sort(key=lambda x: x[1], reverse=True)

    explanation_msgs = []
    for feature, _ in abs_shap:
        if len(explanation_msgs) >= top_n:
            break

        shap_val = shap_values[feature_names.index(feature)]
        if isinstance(shap_val, np.ndarray):
            shap_val = shap_val.item()
        else:
            shap_val = float(shap_val)

        direction = "increases" if shap_val > 0 else "decreases"

        # Check if feature is one-hot encoded
        if "__" in feature:  # e.g., age__70_80
            orig_col, category = feature.split("__", 1)
            raw_val = raw_input.get(orig_col, None)
            if raw_val == category:
                msg = f"Being in the category *{orig_col} = {category}* {direction} the predicted readmission risk."
            else:
                continue  # inactive category
        else:
            raw_val = raw_input.get(feature, None)
            msg = f"Feature *{feature}* with value {raw_val} {direction} the predicted readmission risk."

        explanation_msgs.append(msg)

    return explanation_msgs


# ================== MEDICAL ASSISTANT (Gemini) ==================
# ---------- Helpers for dataset analytics ----------
def _get_df():
    df = st.session_state.get("dataset")
    if df is None:
        st.warning("No dataset loaded. Upload a CSV in the Admin Dashboard → Data & EDA.")
        return None
    # drop duplicated column names, keep first occurrence
    df = df.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def _detect_readmit_col(df: pd.DataFrame):
    # prefer explicit target chosen in UI
    tgt = st.session_state.get("target_col", None)
    if tgt and tgt in df.columns:
        return tgt
    # fallbacks by name
    for c in df.columns:
        if re.search(r"(readmit|readmission|readmitted)", str(c), re.I):
            return c
    return None

def _ensure_series(x) -> pd.Series:
    """Make sure we always operate on a Series (never a 2D DataFrame)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series([], dtype=object)
        return x.iloc[:, 0]
    return x

def _binarize_readmit(series) -> pd.Series:
    """
    Convert readmission column to 0/1 robustly.
    Handles: 'yes'/'no', '<30'/'>'/'no', booleans, numeric 0/1, etc.
    """
    s = _ensure_series(series)

    # Early numeric/boolean path
    if pd.api.types.is_numeric_dtype(s):
        # If it's already 0/1 or close, coerce and clip.
        sn = pd.to_numeric(s, errors="coerce").fillna(0)
        uniq = set(pd.unique(sn.dropna()))
        if uniq <= {0, 1}:
            return sn.astype(int)
        # treat any non-zero as 1
        return (sn != 0).astype(int)

    # General string path
    s = s.astype(str).str.strip().str.lower()

    # Common encodings
    vals = set(s.unique())
    if {"yes", "no"} & vals:
        return s.map({"yes": 1, "no": 0}).fillna(0).astype(int)

    # Diabetes 130-US hospitals style: NO / <30 / >30 → any not NO is positive
    if {"<30", ">30", "no"} & vals:
        return s.map({"no": 0}).fillna(1).astype(int)

    # true/false
    if {"true", "false"} & vals:
        return s.map({"true": 1, "false": 0}).fillna(0).astype(int)

    # 'readmitted'/'not readmitted' style (rare)
    if {"readmitted", "not readmitted"} & vals:
        return s.map({"readmitted": 1, "not readmitted": 0}).fillna(0).astype(int)

    # fallback: anything not 'no' becomes 1
    return (s != "no").astype(int)

def _pick_col(df, options):
    """Return the first column that exists (case-insensitive) from options list or regex."""
    # direct case-insensitive match
    lowered = {str(c).lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in lowered:
            return lowered[opt.lower()]
    # regex contains match
    for opt in options:
        for c in df.columns:
            if re.search(opt, str(c), re.I):
                return c
    return None

def _rate_by(df, by_col, readmit_col):
    """Tabulate readmission count & rate grouped by a column (categorical or binned)."""
    # Work on a minimal copy and guarantee 1-D key + 1-D label
    tmp = df[[by_col, readmit_col]].copy()

    # Build robust 1-D Series for group key and outcome
    key = tmp[by_col]
    if isinstance(key, pd.DataFrame):
        key = key.iloc[:, 0]
    y = _binarize_readmit(tmp[readmit_col])

    tmp = pd.DataFrame({
        "__key__": key.values,
        "__y__": y.values
    })

    grp = tmp.groupby("__key__", dropna=False)["__y__"].agg(
        count="size",
        readmitted="sum",
        readmit_rate="mean",
    ).reset_index().rename(columns={"__key__": str(by_col)})

    grp = grp.sort_values("readmit_rate", ascending=False)
    grp["readmit_rate"] = (grp["readmit_rate"] * 100).round(2).astype(str) + "%"
    return grp

def _avg_stay_by_readmit(df, readmit_col):
    if "time_in_hospital" not in df.columns:
        return None
    tmp = df[[readmit_col, "time_in_hospital"]].copy()
    tmp["readmitted"] = _binarize_readmit(tmp[readmit_col])
    out = tmp.groupby("readmitted")["time_in_hospital"].agg(
        count="size", mean="mean", median="median"
    ).reset_index()
    out["readmitted"] = out["readmitted"].map({0: "No", 1: "Yes"})
    out["mean"] = out["mean"].round(2)
    out["median"] = out["median"].round(2)
    return out[["readmitted", "count", "mean", "median"]]

def _overall_readmit(df, readmit_col):
    y = _binarize_readmit(df[readmit_col])
    rate = float(y.mean()) if len(y) else 0.0
    return pd.DataFrame([{
        "total_records": int(len(df)),
        "readmitted": int(y.sum()),
        "not_readmitted": int((1 - y).sum()),
        "readmit_rate": f"{rate*100:.2f}%"
    }])

def _bucketize_numeric(df, col, bins=5, label_prefix=None):
    s = pd.to_numeric(df[col], errors="coerce")
    # Let pandas label automatically; it's clearer (e.g., (0.0, 3.0])
    return pd.cut(s, bins=bins, duplicates="drop")

def _top_categories_by_readmit(df, cat_col, readmit_col, topn=10, min_count=20):
    tmp = df[[cat_col, readmit_col]].copy()
    tmp["y"] = _binarize_readmit(tmp[readmit_col])
    agg = tmp.groupby(cat_col)["y"].agg(count="size", readmit_rate="mean").reset_index()
    agg = agg[agg["count"] >= min_count]
    agg = agg.sort_values(["readmit_rate", "count"], ascending=[False, False]).head(topn)
    agg["readmit_rate"] = (agg["readmit_rate"] * 100).round(2).astype(str) + "%"
    return agg.rename(columns={cat_col: "category"})

# ---------- The Chatbot ----------
def admin_chatbot():
    st.markdown(
        "<div class='card'><h2>💬 Hospital Data Assistant</h2>"
        "<p>Ask about dataset, trained models, MongoDB readmissions, or analytics like "
        "<i>age group with highest readmission</i>, <i>readmission by specialty</i>, "
        "<i>average stay for readmitted</i>, <i>readmission by gender</i>, etc.</p></div>",
        unsafe_allow_html=True
    )

    if "admin_chat_history" not in st.session_state:
        st.session_state.admin_chat_history = []

    # Show history
    for msg in st.session_state.admin_chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    # Input
    query = st.chat_input("Ask about dataset, models, MongoDB, or analytics…")
    if not query:
        return
    st.session_state.admin_chat_history.append({"role": "user", "content": query})
    q = query.lower()

    reply = None  # default text reply

    # ===== Dataset-size / columns =====
    if any(k in q for k in ["dataset", "features", "columns", "rows", "shape"]):
        df = _get_df()
        if df is None:
            return
        reply = (
            f"Dataset has **{len(df)} rows** and **{df.shape[1]} columns**. "
            f"Sample columns: {', '.join(map(str, df.columns[:min(10, len(df.columns))]))}."
        )

    # ===== Trained models (list) =====
    elif any(k in q for k in ["trained", "available models", "saved models", "list models"]):
        saved = load_json("trained_models.json")
        if not saved:
            reply = "No trained models are available."
        else:
            models = [f"{mid} ({mval['model_name']})" for mid, mval in saved.items()]
            reply = f"There are **{len(models)} trained models**: {', '.join(models)}."

    # ===== Active model =====
    elif any(k in q for k in ["active", "live", "running", "current model"]):
        saved = load_json("trained_models.json")
        active = [mid for mid, mval in saved.items() if mval.get("active")]
        if active:
            mid = active[0]
            acc = saved[mid]["metrics"]["Accuracy"]
            reply = f"The active model is **{mid}**, serving live predictions with accuracy **{acc:.2f}**."
        else:
            reply = "No model is currently active."

    # ===== Specific model metrics =====
    elif any(k in q for k in ["metrics", "accuracy", "precision", "recall", "f1"]):
        saved = load_json("trained_models.json")
        found = None
        for mid, mval in saved.items():
            if re.search(mid.lower(), q) or re.search(mval["model_name"].lower(), q):
                m = mval["metrics"]
                found = (
                    f"**{mid} ({mval['model_name']})** → "
                    f"Accuracy {m['Accuracy']:.2f}, F1 {m['F1 Score']:.2f}, "
                    f"Precision {m['Precision']:.2f}, Recall {m['Recall']:.2f}."
                )
                break
        reply = found if found else "Please specify which model you want metrics for (ID or name)."

    # ===== Compare models / best =====
    elif any(k in q for k in ["compare", "best", "better", "higher", "outperform"]):
        saved = load_json("trained_models.json")
        if not saved:
            reply = "No models available for comparison."
        else:
            best_mid, _ = max(saved.items(), key=lambda x: x[1]["metrics"]["Accuracy"])
            reply = (
                f"**{best_mid} ({saved[best_mid]['model_name']})** "
                f"is best by accuracy **{saved[best_mid]['metrics']['Accuracy']:.2f}**."
            )

    # ===== MongoDB predictions (table always) =====
    elif any(k in q for k in ["mongodb", "prediction", "stored", "saved", "db", "database", "count"]):
        try:
            count = predictions_collection.count_documents({})
            if count == 0:
                reply = "No predictions have been stored in MongoDB yet."
            else:
                # detect count intent
                limit = 3
                m = re.search(r"last\s+(\d+)", q)
                if "all" in q:
                    limit = count
                elif m:
                    limit = int(m.group(1))
                elif "last 5" in q:
                    limit = 5

                preds = list(
                    predictions_collection.find().sort("prediction_date", -1).limit(limit)
                )
                table = pd.DataFrame(
                    [{
                        "Date": str(p.get("prediction_date")),
                        "Model Used": p.get("model_used"),
                        "Readmitted": p.get("predicted_readmission"),
                        "Probability": p.get("probabilities", "N/A"),
                        "Dataset": p.get("dataset_source", "Unknown")
                    } for p in preds]
                )
                st.session_state.admin_chat_history.append({
                    "role": "assistant",
                    "content": f"MongoDB holds **{count}** predictions. Showing last **{len(table)}**:"
                })
                st.chat_message("assistant").write(
                    f"MongoDB holds **{count}** predictions. Showing last **{len(table)}**:"
                )
                st.table(table)
                return
        except Exception as e:
            reply = f"Could not fetch MongoDB predictions. Error: {e}"

    # ======== Dataset analytics Q&A ========
    else:
        df = _get_df()
        if df is None:
            return
        readmit_col = _detect_readmit_col(df)
        if not readmit_col:
            st.session_state.admin_chat_history.append(
                {"role": "assistant", "content": "Could not locate a readmission column in the dataset."}
            )
            st.chat_message("assistant").write("Could not locate a readmission column in the dataset.")
            return

        # 1) Age group readmission rate
        if ("age" in q and "readmit" in q) or ("age group" in q):
            age_col = _pick_col(df, ["age"])
            if age_col:
                st.chat_message("assistant").write("Readmission rate by age group:")
                st.table(_rate_by(df, age_col, readmit_col))
                return
            reply = "No 'age' column found to compute age-group readmission rates."

        # 2) Readmission by medical specialty
        elif ("specialty" in q or "medical specialty" in q) and "readmit" in q:
            spec_col = _pick_col(df, ["medical_specialty", "specialty"])
            if spec_col:
                st.chat_message("assistant").write("Readmission rate by medical specialty:")
                st.table(_rate_by(df, spec_col, readmit_col).head(15))
                return
            reply = "No medical specialty column found."

        # 3) Top diagnoses leading to readmission (diag_1)
        elif ("diagnosis" in q or "diagnoses" in q or "diag" in q) and "readmit" in q:
            diag_col = _pick_col(df, ["diag_1", "primary_diagnosis", r"diag[_\s]?1"])
            if diag_col:
                st.chat_message("assistant").write("Top diagnosis categories by readmission rate (min 20 cases):")
                st.table(_top_categories_by_readmit(df, diag_col, readmit_col, topn=10, min_count=20))
                return
            reply = "No diagnosis column (diag_1) found."

        # 4) Readmission rate by A1C / glucose / diabetes_med / change
        elif "a1c" in q:
            a1c_col = _pick_col(df, ["A1Ctest", "a1c_result", "a1c_test"])
            if a1c_col:
                st.chat_message("assistant").write("Readmission rate by A1C test:")
                st.table(_rate_by(df, a1c_col, readmit_col))
                return
            reply = "No A1C test column found."

        elif "glucose" in q or "glu" in q:
            glu_col = _pick_col(df, ["glucose_test", "max_glu_serum", "max_glucose"])
            if glu_col:
                st.chat_message("assistant").write("Readmission rate by glucose test/result:")
                st.table(_rate_by(df, glu_col, readmit_col))
                return
            reply = "No glucose-related column found."

        elif "diabetes" in q:
            dm_col = _pick_col(df, ["diabetes_med", "diabetesmed"])
            if dm_col:
                st.chat_message("assistant").write("Readmission rate by diabetes_med:")
                st.table(_rate_by(df, dm_col, readmit_col))
                return
            reply = "No diabetes_med column found."

        elif "change" in q and "readmit" in q:
            ch_col = _pick_col(df, ["change"])
            if ch_col:
                st.chat_message("assistant").write("Readmission rate by 'change' flag:")
                st.table(_rate_by(df, ch_col, readmit_col))
                return
            reply = "No 'change' column found."

        # 5) Average stay for readmitted vs not
        elif any(x in q for x in ["average stay", "avg stay", "time in hospital"]) and \
             any(x in q for x in ["readmit", "readmitted", "by readmit"]):
            table = _avg_stay_by_readmit(df, readmit_col)
            if table is not None:
                st.chat_message("assistant").write("Average time_in_hospital by readmission outcome:")
                st.table(table)
                return
            reply = "No 'time_in_hospital' column to compute average stay."

        # 6) Readmission rate by number of medications (binned)
        elif ("medication" in q or "n_medications" in q or "num_medications" in q) and "readmit" in q:
            meds_col = _pick_col(df, ["n_medications", "num_medications"])
            if meds_col:
                df["_meds_bin"] = _bucketize_numeric(df, meds_col, bins=6)
                table = _rate_by(df, "_meds_bin", readmit_col).rename(columns={"_meds_bin": "n_medications_bin"})
                st.chat_message("assistant").write("Readmission rate by medication count (binned):")
                st.table(table)
                df.drop(columns=["_meds_bin"], inplace=True)
                return
            reply = "No numeric medications column found."

        # 7) Overall readmission rate
        elif "overall" in q and "readmit" in q:
            st.chat_message("assistant").write("Overall readmission summary:")
            st.table(_overall_readmit(df, readmit_col))
            return

        # 8) Generic: "by <column>" intent (e.g., "readmission by gender")
        elif "by " in q and "readmit" in q:
            m = re.search(r"by\s+([a-zA-Z0-9_]+)", q)
            if m:
                col_guess = m.group(1)
                col = _pick_col(df, [col_guess])
                if col:
                    st.chat_message("assistant").write(f"Readmission rate by {col}:")
                    st.table(_rate_by(df, col, readmit_col))
                    return
                reply = f"Column '{col_guess}' not found."

        # Fallback to LLM if none matched
        if reply is None:
            response = client1.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {"role": "system", "content": "You are a concise admin assistant. Reply with short analytical summaries only."},
                    {"role": "user", "content": query}
                ]
            )
            reply = response.choices[0].message.content

    # Final text reply (non-tabular)
    st.session_state.admin_chat_history.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)


def medical_chatbot():
    st.markdown("<div class='card'><h2>💬 Medical Assistant</h2><p>Ask health-related questions. Answers are short, doctor-style suggestions. Always consult a physician.</p></div>", unsafe_allow_html=True)

    # Gemini client
    client1 = OpenAI(
        api_key="AIzaSyBnyIj10Y5THLtnZIYzC1-RC1SAfo486M4",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a trusted doctor assistant. Provide concise, medically accurate answers. Always suggest consulting a physician."}
        ]

    for msg in st.session_state.chat_history[1:]:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("Ask your medical question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        response = client1.chat.completions.create(
            model="gemini-2.5-flash",
            messages=st.session_state.chat_history
        )

        reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)


# ------------------ Main app --------------------

st.set_page_config(page_title="Hospital Readmission AI Dashboard", layout="wide", page_icon="🤖")


st.sidebar.title("Navigation")
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    if st.session_state.get("preferred_user_login", False):
        page = "Login"
        st.session_state["preferred_user_login"] = False
    else:
        page = st.sidebar.selectbox("Select Page", ["Login"])
else:
    page = st.sidebar.selectbox("Select Page", ["Hospital Data Administrator", "Patient Risk Evaluation Interface","Medical Assistant","Logout"])


def logout_and_stop():
    st.session_state.logged_in = False
    st.session_state.role = None
    st.stop()


if page == "Login":
    login()
elif page == "Logout":
    if st.sidebar.button("Confirm Logout"):
        logout_and_stop()
elif page == "Hospital Data Administrator":
    if st.session_state.logged_in and st.session_state.role == "admin":
        admin_dashboard()
    else:
        st.warning("You need admin login to access this page.")
elif page == "Patient Risk Evaluation Interface":
    if st.session_state.logged_in and st.session_state.role == "user":
        user_dashboard()
    else:
        st.warning("You need user login to access this page.")
elif page == "Medical Assistant":
    if st.session_state.role == "admin":
        admin_chatbot()
    elif st.session_state.role == "user":
        medical_chatbot()

