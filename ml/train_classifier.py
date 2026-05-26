import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocess import preprocess, FEATURE_COLS, TARGET_COL

# Load data
df = pd.read_csv('data/raw/sales_transactions.csv')
df = preprocess(df)

# Drop any remaining nulls
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50   # prints every 50 rounds
)

# ── Evaluation ─────────────────────────────────────────────────
y_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test, y_pred)

print("\n" + "="*50)
print(f"Training Accuracy : {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy     : {test_acc:.4f} ({test_acc*100:.2f}%)")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Sales', 'High Sales']))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save model bundle ──────────────────────────────────────────
bundle = {
    'model': model,
    'feature_cols': FEATURE_COLS
}
with open('ml/saved_models/sales_classifier.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print("\nModel saved to ml/saved_models/sales_classifier.pkl")