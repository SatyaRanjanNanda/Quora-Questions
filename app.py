import streamlit as st
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Models
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('xgboost_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

with open('sbert_model.pkl', 'rb') as f:
    sbert_model = pickle.load(f)

# 2. Streamlit UI
st.title("Quora Question Duplicate Detector")
st.write("This app checks if two questions are duplicates.")

# 3. Input Questions
q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check for Duplicate"):
    if q1 and q2:
        # 4. Preprocess Questions
        q1_embed = sbert_model.encode([q1])[0]
        q2_embed = sbert_model.encode([q2])[0]

        # 5. Feature Engineering
        cos_sim = cosine_similarity(q1_embed.reshape(1, -1), q2_embed.reshape(1, -1))[0][0]
        combined_features = np.hstack([
            [cos_sim],
            (q1_embed + q2_embed) / 2
        ]).reshape(1, -1)

        # 6. Predict using Random Forest
        with st.spinner('Predicting with Random Forest...'):
            rf_pred = rf_model.predict(combined_features)[0]
            rf_pred_prob = float(rf_model.predict_proba(combined_features)[:, 1][0])

        # 7. Predict using XGBoost
        with st.spinner('Predicting with XGBoost...'):
            xgb_pred = xgb_model.predict(combined_features)[0]
            xgb_pred_prob = float(xgb_model.predict_proba(combined_features)[:, 1][0])

        # 8. Show Progress Bar
        progress = st.progress(0)
        for i in range(100):
            progress.progress(i + 1)

        # 9. Final Decision (prefer XGBoost more)
        final_decision = None
        if rf_pred == xgb_pred:
            final_decision = rf_pred  # Both models agree
        else:
            final_decision = xgb_pred  # Trust XGBoost more

        # 10. Display results
        if final_decision == 1:
            st.success(f"✅ Final Prediction: Duplicate (XGBoost Confidence: {xgb_pred_prob * 100:.2f}%)")
        else:
            st.error(f"❌ Final Prediction: Not Duplicate (XGBoost Confidence: {xgb_pred_prob * 100:.2f}%)")

        st.write(f"**Random Forest Prediction:** {'Duplicate' if rf_pred == 1 else 'Not Duplicate'}")
        st.write(f"**Random Forest Confidence:** {rf_pred_prob * 100:.2f}%")
        st.write(f"**XGBoost Prediction:** {'Duplicate' if xgb_pred == 1 else 'Not Duplicate'}")
        st.write(f"**XGBoost Confidence:** {xgb_pred_prob * 100:.2f}%")

    else:
        st.warning("Please enter both questions.")
