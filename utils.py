import os
import joblib
from datetime import datetime

def save_model(clf, thr_op,methode, base_path="models"):
    os.makedirs(base_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{methode}_{timestamp}.joblib"

    file_path = os.path.join(base_path,file_name)
    
    production_artifact = {
        "model": clf,
        "optimal_threshold": thr_op,
    }
    
    joblib.dump(production_artifact, file_path)
    return file_path

def load_model(path):
    artifact = joblib.load(path)

    model = artifact["model"]
    threshold = artifact["optimal_threshold"]
    return model,threshold