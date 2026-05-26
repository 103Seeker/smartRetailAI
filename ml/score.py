import json, pickle, numpy as np
import os

def init():
    global model, feature_cols
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'sales_classifier.pkl')
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    model = bundle['model']
    feature_cols = bundle['feature_cols']

def run(raw_data):
    data = json.loads(raw_data)['data']
    features = np.array([[d[f] for f in feature_cols] for d in data])
    preds = model.predict(features).tolist()
    probs = model.predict_proba(features).max(axis=1).tolist()
    return json.dumps({'predictions': preds, 'probabilities': probs})

