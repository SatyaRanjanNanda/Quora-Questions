import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
df = pd.read_csv('train.csv')
df = df.sample(n=50000, random_state=42).reset_index(drop=True)  # 50k samples

df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')
df['combined'] = df['question1'] + " " + df['question2']

# 2. Load SBERT
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. Encode separately
q1_embeds = sbert_model.encode(df['question1'].tolist(), batch_size=64, show_progress_bar=True)
q2_embeds = sbert_model.encode(df['question2'].tolist(), batch_size=64, show_progress_bar=True)

# 4. Feature engineering
cosine_similarities = [cosine_similarity(q1_embeds[i].reshape(1, -1), q2_embeds[i].reshape(1, -1))[0][0]
                       for i in range(len(q1_embeds))]

# 5. Final Feature Matrix
X = np.hstack([
    np.array(cosine_similarities).reshape(-1, 1),
    (q1_embeds + q2_embeds)/2  # Average of embeddings
])

y = df['is_duplicate']

# 6. Train/Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train, y_train)

# 8. Train XGBoost
positive_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=positive_weight,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 9. Validation Accuracy
rf_preds = rf_model.predict(X_val)
xgb_preds = xgb_model.predict(X_val)

print(f"✅ Random Forest Accuracy: {accuracy_score(y_val, rf_preds):.4f}")
print(f"✅ XGBoost Accuracy: {accuracy_score(y_val, xgb_preds):.4f}")

# 10. Save models
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

print("✅ Models retrained and saved successfully!")
