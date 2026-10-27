"""save/load the trained model + the stuff needed to actually use it"""
from __future__ import annotations
 
from pathlib import Path
 
import joblib
 
from .features import FEATURE_NAMES, model_input
 
DEFAULT_MODEL_PATH = Path("models/model.joblib")

# what the model expects to get. change this if that ever changes so an old
# model gets refused instead of giving garbage
INPUT_FORMAT = "features+url"
 
 
def save_model(model, path: Path, metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # store the feature names with it so I can tell later if features.py changed
    bundle = {"model": model, "feature_names": FEATURE_NAMES,
              "input_format": INPUT_FORMAT, "metadata": metadata}
    joblib.dump(bundle, path)
 
 
def load_model(path: Path = DEFAULT_MODEL_PATH):
    bundle = joblib.load(path)
    # old model + new features = garbage predictions, so just refuse to load
    if bundle["feature_names"] != FEATURE_NAMES or bundle.get("input_format") != INPUT_FORMAT:
        raise RuntimeError(
            "The features in phishdetect/features.py changed since this model was "
            "trained. Retrain it: python -m scripts.train"
        )
    return bundle
 
 
def predict_proba(bundle, urls):
    # chance each url is phishing, 0 to 1
    X = model_input(urls)
    return bundle["model"].predict_proba(X)[:, 1]
 
