import argparse 
import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json 
import time 
from dataclasses import dataclass 
from typing import Dict, Tuple, List 
 
import numpy as np 
import pandas as pd 
 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import RobustScaler 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.metrics import ( 
    precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, 
    confusion_matrix, roc_curve, precision_recall_curve 
) 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier, IsolationForest 
from sklearn.utils.class_weight import compute_class_weight 
 
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.pipeline import Pipeline as ImbPipeline 
 
import matplotlib 
matplotlib.use("Agg") 
import matplotlib.pyplot as plt 
import joblib 
 
import tensorflow as tf 
from tensorflow.keras import layers, models, callbacks 
 
RANDOM_STATE = 42 
ARTIFACT_DIR = "artifacts" 
PLOTS_DIR = os.path.join(ARTIFACT_DIR, "plots") 
 
os.makedirs(ARTIFACT_DIR, exist_ok=True) 
os.makedirs(PLOTS_DIR, exist_ok=True) 
 
@dataclass 
class EvalResult: 
    model_name: str 
    threshold: float 
    precision: float 
    recall: float 
    f1: float 
    roc_auc: float 
    pr_auc: float 
    cm: np.ndarray 
 
 
def load_data(path: str) -> pd.DataFrame: 
    df = pd.read_csv(path) 
    assert set(["Time", "Amount", "Class"]).issubset(df.columns), "CSV must contain Time, Amount, Class" 
    return df 
 
 
def perform_eda(df: pd.DataFrame) -> None: 
    
    fraud_ratio = df["Class"].mean() 
 
    plt.figure() 
    df["Class"].value_counts().sort_index().plot(kind="bar") 
    plt.title(f"Class Distribution (Fraud ratio: {fraud_ratio:.4f})") 
    plt.xlabel("Class (0=Legit, 1=Fraud)") 
    plt.ylabel("Count") 
    plt.tight_layout() 
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png")) 
    plt.close() 
 
    plt.figure() 
    df["Amount"].plot(kind="hist", bins=100) 
    plt.title("Transaction Amount Distribution") 
    plt.xlabel("Amount") 
    plt.ylabel("Frequency") 
    plt.tight_layout() 
    plt.savefig(os.path.join(PLOTS_DIR, "amount_hist.png")) 
    plt.close() 
 
    plt.figure() 
    df["Time"].plot(kind="hist", bins=100) 
    plt.title("Transaction Time Distribution") 
    plt.xlabel("Time (seconds from first txn)") 
    plt.ylabel("Frequency") 
    plt.tight_layout() 
    plt.savefig(os.path.join(PLOTS_DIR, "time_hist.png")) 
    plt.close() 
 
     
    plt.figure() 
    df.boxplot(column="Amount", by="Class") 
    plt.title("Amount by Class") 
    plt.suptitle("") 
    plt.xlabel("Class") 
    plt.ylabel("Amount") 
    plt.tight_layout() 
    plt.savefig(os.path.join(PLOTS_DIR, "amount_by_class.png")) 
    plt.close() 
 
 
def make_splits(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
np.ndarray, np.ndarray, List[str]]: 
    X = df.drop(columns=["Class"]) 
    y = df["Class"].astype(int) 
 
    feature_names = X.columns.tolist() 
 
    X_train_val, X_test, y_train_val, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE ) 
    X_train, X_val, y_train, y_val = train_test_split( X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=RANDOM_STATE )   
 
     
    scaler = RobustScaler() 
    X_train_s = scaler.fit_transform(X_train) 
    X_val_s = scaler.transform(X_val) 
    X_test_s = scaler.transform(X_test) 
 
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.joblib")) 
 
    return X_train_s, X_val_s, X_test_s, y_train.values, y_val.values, y_test.values, feature_names 
 
 
 
def train_logreg(X_tr, y_tr, use_smote=False, under_ratio=None): 
    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), 
y=y_tr) 
    cw = {0: class_weights[0], 1: class_weights[1]} 
 
    clf = LogisticRegression(max_iter=200, n_jobs=None, random_state=RANDOM_STATE, 
class_weight=cw) 
 
    if use_smote: 
        steps = [("smote", SMOTE(random_state=RANDOM_STATE))] 
        if under_ratio: 
            steps.append(("under", RandomUnderSampler(sampling_strategy=under_ratio, 
random_state=RANDOM_STATE))) 
        steps.append(("clf", clf)) 
        pipe = ImbPipeline(steps) 
        pipe.fit(X_tr, y_tr) 
        return pipe 
    else: 
        clf.fit(X_tr, y_tr) 
        return clf 
 
 
def train_rf(X_tr, y_tr, use_smote=False, under_ratio=None): 
    clf = RandomForestClassifier( 
        n_estimators=300, 
        max_depth=None, 
        n_jobs=-1, 
        random_state=RANDOM_STATE, 
        class_weight="balanced_subsample", 
    ) 
    if use_smote: 
        steps = [("smote", SMOTE(random_state=RANDOM_STATE))] 
        if under_ratio: 
            steps.append(("under", RandomUnderSampler(sampling_strategy=under_ratio, 
random_state=RANDOM_STATE))) 
        steps.append(("clf", clf)) 
        pipe = ImbPipeline(steps) 
        pipe.fit(X_tr, y_tr) 
        return pipe 
    else: 
        clf.fit(X_tr, y_tr) 
        return clf 
 
 
 
 
def train_isolation_forest(X_tr, y_tr): 
 
    X_norm = X_tr[y_tr == 0] 
    iso = IsolationForest( 
        n_estimators=300, 
        max_samples="auto", 
        contamination=0.001,   
        random_state=RANDOM_STATE, 
        n_jobs=-1, 
    ) 
    iso.fit(X_norm) 
    return iso 
 
 
def train_autoencoder(X_tr, y_tr, epochs=20, batch_size=512): 
    X_norm = X_tr[y_tr == 0] 
    input_dim = X_norm.shape[1] 
 
    enc_dim = 14 
    model = models.Sequential([ 
layers.Input(shape=(input_dim,)), 
        layers.Dense(24, activation="relu"), 
        layers.Dense(enc_dim, activation="relu"), 
        layers.Dense(24, activation="relu"), 
        layers.Dense(input_dim, activation="linear"), 
    ]) 
    model.compile(optimizer="adam", loss="mse") 
 
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=3, 
restore_best_weights=True)] 
    model.fit( 
        X_norm, 
        X_norm, 
        validation_split=0.1, 
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=1, 
        callbacks=cb, 
        shuffle=True, 
    ) 
 
    recon_train = np.mean(np.square(X_norm - model.predict(X_norm, verbose=0)), axis=1) 
    threshold = np.percentile(recon_train, 99.5) 
 
    model.save(os.path.join(ARTIFACT_DIR, "autoencoder.keras")) 
    with open(os.path.join(ARTIFACT_DIR, "autoencoder_threshold.json"), "w") as f: 
        json.dump({"threshold": float(threshold)}, f) 
 
    return model, threshold 
 
 
 
def score_supervised(clf, X): 
    if hasattr(clf, "predict_proba"): 
        return clf.predict_proba(X)[:, 1] 
    elif hasattr(clf, "decision_function"): 
        s = clf.decision_function(X) 
        s = (s - s.min()) / (s.max() - s.min() + 1e-8) 
        return s 
    else: 
        return clf.predict(X) 
 
 
def score_isolation_forest(iso: IsolationForest, X): 
 
    s = -iso.score_samples(X)   
    s = (s - s.min()) / (s.max() - s.min() + 1e-8) 
    return s 
 
 
def score_autoencoder(model: tf.keras.Model, X, threshold: float) -> Tuple[np.ndarray, 
np.ndarray]: 
    recon = np.mean(np.square(X - model.predict(X, verbose=0)), axis=1) 
    s = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8) 
    y_pred = (recon >= threshold).astype(int) 
    return s, y_pred 
def evaluate_scores(y_true, scores, model_name: str) -> EvalResult: 
    precision, recall, thr = precision_recall_curve(y_true, scores) 
    f1s = 2 * precision * recall / (precision + recall + 1e-12) 
    best_idx = int(np.nanargmax(f1s)) 
    best_thr = thr[best_idx] if best_idx < len(thr) else 0.5 
    y_pred = (scores >= best_thr).astype(int) 
 
    pr = precision_score(y_true, y_pred, zero_division=0) 
    rc = recall_score(y_true, y_pred, zero_division=0) 
    f1 = f1_score(y_true, y_pred, zero_division=0) 
    try: 
        roc = roc_auc_score(y_true, scores) 
    except Exception: 
        roc = float("nan") 
    pr_auc = average_precision_score(y_true, scores) 
    cm = confusion_matrix(y_true, y_pred) 
 
    return EvalResult(model_name, float(best_thr), float(pr), float(rc), float(f1), float(roc), 
float(pr_auc), cm) 
 
 
def plot_curves(y_true, scores, name: str): 
    fpr, tpr, _ = roc_curve(y_true, scores) 
    prec, rec, _ = precision_recall_curve(y_true, scores) 
 
    plt.figure() 
    plt.plot(fpr, tpr) 
    plt.xlabel("FPR") 
    plt.ylabel("TPR") 
    plt.title(f"ROC: {name}") 
    plt.tight_layout() 
    plt.savefig(os.path.join(PLOTS_DIR, f"ROC_{name}.png")) 
    plt.close() 
 
    plt.figure() 
    plt.plot(rec, prec) 
    plt.xlabel("Recall") 
    plt.ylabel("Precision") 
    plt.title(f"PR: {name}") 
    plt.tight_layout() 
    plt.savefig(os.path.join(PLOTS_DIR, f"PR_{name}.png")) 
    plt.close() 
 
 
 
def train_and_evaluate(df: pd.DataFrame, use_smote: bool, under_ratio: float) -> None: 
    perform_eda(df) 
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = make_splits(df) 
 
    results: List[EvalResult] = [] 
 
    logreg = train_logreg(X_tr, y_tr, use_smote=use_smote, under_ratio=under_ratio) 
    scores_lr = score_supervised(logreg, X_val) 
    res_lr = evaluate_scores(y_val, scores_lr, "LogReg") 
    plot_curves(y_val, scores_lr, "LogReg") 
    results.append(res_lr) 
 
    rf = train_rf(X_tr, y_tr, use_smote=use_smote, under_ratio=under_ratio) 
    scores_rf = score_supervised(rf, X_val) 
    res_rf = evaluate_scores(y_val, scores_rf, "RandomForest") 
    plot_curves(y_val, scores_rf, "RandomForest") 
    results.append(res_rf) 
 
    iso = train_isolation_forest(X_tr, y_tr) 
    scores_iso = score_isolation_forest(iso, X_val) 
    res_iso = evaluate_scores(y_val, scores_iso, "IsolationForest") 
    plot_curves(y_val, scores_iso, "IsolationForest") 
    results.append(res_iso) 
 
    ae_model, ae_thr = train_autoencoder(X_tr, y_tr) 
    ae_scores_val, ae_pred_val = score_autoencoder(ae_model, X_val, ae_thr) 
    res_ae = evaluate_scores(y_val, ae_scores_val, "Autoencoder") 
    plot_curves(y_val, ae_scores_val, "Autoencoder") 
    results.append(res_ae) 
 
    results_sorted = sorted(results, key=lambda r: r.f1, reverse=True) 
    best = results_sorted[0] 
    print("\nValidation results (sorted by F1):") 
    for r in results_sorted: 
        print(
        f"{r.model_name}: "
        f"F1={r.f1:.4f}, "
        f"Precision={r.precision:.4f}, "
        f"Recall={r.recall:.4f}, "
        f"PR-AUC={r.pr_auc:.4f}"
)
 
    model_to_save = None 
    meta = {"best_model": best.model_name, "threshold": best.threshold} 
 
    if best.model_name == "LogReg": 
        model_to_save = logreg 
    elif best.model_name == "RandomForest": 
        model_to_save = rf 
    else: 
        model_to_save = rf 
        joblib.dump(iso, os.path.join(ARTIFACT_DIR, "iso_forest.joblib")) 
        meta["unsupervised_saved"] = "iso_forest.joblib" 
 
    joblib.dump(model_to_save, os.path.join(ARTIFACT_DIR, "best_supervised.joblib")) 
    with open(os.path.join(ARTIFACT_DIR, "model_meta.json"), "w") as f: 
        json.dump(meta, f, indent=2) 
 
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib")) 
    X_te_s = X_te   
    scores_test = score_supervised(model_to_save, X_te_s) 
    res_test = evaluate_scores(y_te, scores_test, f"Test-{best.model_name}") 
    plot_curves(y_te, scores_test, f"Test-{best.model_name}") 
 
    print("\nTest set:") 
    print(
    f"{res_test.model_name}: "
    f"F1={res_test.f1:.4f}, "
    f"Precision={res_test.precision:.4f}, "
    f"Recall={res_test.recall:.4f}, "
    f"PR-AUC={res_test.pr_auc:.4f}"
)

    print("Confusion Matrix:\n", res_test.cm) 
 
 
from fastapi import FastAPI 
from pydantic import BaseModel 
 
app = FastAPI(title="Fraud Detection API", version="1.0") 
 
class Txn(BaseModel): 
    Time: float 
    Amount: float 
    V1: float = 0.0 
    V2: float = 0.0 
    V3: float = 0.0 
    V4: float = 0.0 
    V5: float = 0.0 
    V6: float = 0.0 
    V7: float = 0.0 
    V8: float = 0.0 
    V9: float = 0.0 
    V10: float = 0.0 
    V11: float = 0.0 
    V12: float = 0.0 
    V13: float = 0.0 
    V14: float = 0.0 
    V15: float = 0.0 
    V16: float = 0.0 
    V17: float = 0.0 
    V18: float = 0.0 
    V19: float = 0.0 
    V20: float = 0.0 
    V21: float = 0.0 
    V22: float = 0.0 
    V23: float = 0.0 
    V24: float = 0.0 
    V25: float = 0.0 
    V26: float = 0.0 
    V27: float = 0.0 
    V28: float = 0.0 
 
 
def load_serving_artifacts(): 
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.joblib")) 
    model = joblib.load(os.path.join(ARTIFACT_DIR, "best_supervised.joblib")) 
    meta_path = os.path.join(ARTIFACT_DIR, "model_meta.json") 
    if os.path.exists(meta_path): 
        with open(meta_path, "r") as f: 
            meta = json.load(f) 
    else: 
        meta = {"best_model": "Unknown", "threshold": 0.5} 
    return scaler, model, meta 
 
 
@app.post("/score") 
async def score(txn: Txn): 
    scaler, model, meta = load_serving_artifacts() 
    feat_order = [ 
        "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", 
 "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", 
"V26", "V27", "V28", "Amount" 
    ] 
    x = np.array([[getattr(txn, k) for k in feat_order]], dtype=float) 
    x_s = scaler.transform(x) 
    score = float(score_supervised(model, x_s)[0]) 
    flag = bool(score >= meta.get("threshold", 0.5)) 
    return {"model": meta.get("best_model", "Unknown"), "score": score, "flag": flag, 
"threshold": meta.get("threshold", 0.5)} 
 
 
def stream_alerts(csv_path: str, max_rows: int = 5000, sleep: float = 0.0): 
    scaler, model, meta = load_serving_artifacts() 
    df_iter = pd.read_csv(csv_path, chunksize=1) 
    count = 0 
    for chunk in df_iter: 
        count += 1 
        if count > max_rows: 
            break 
        row = chunk.iloc[0] 
        x = row.drop(labels=["Class"]).values.reshape(1, -1) 
        x_s = scaler.transform(x) 
        score = float(score_supervised(model, x_s)[0]) 
        flag = score >= meta.get("threshold", 0.5) 
        if flag: 
            print(f"ALERT txn_id={count} score={score:.4f} amount={row['Amount']:.2f} time={row['Time']}") 
        if sleep > 0: 
            time.sleep(sleep) 
def main(): 
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline") 
    parser.add_argument("--data_path", type=str, help="Path to creditcard.csv") 
    parser.add_argument("--train", action="store_true", help="Run EDA + training + evaluation") 
    parser.add_argument("--smote", action="store_true", help="Use SMOTE oversampling") 
    parser.add_argument("--under_ratio", type=float, default=None, help="Optional undersampling ratio for majority class after SMOTE (e.g., 0.5)") 
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server for real-time scoring") 
    parser.add_argument("--stream", action="store_true", help="Simulate streaming alerts from CSV (requires trained artifacts)") 
    parser.add_argument("--max_rows", type=int, default=5000, help="Max rows to stream in simulation") 
    parser.add_argument("--sleep", type=float, default=0.0, help="Sleep between streamed rows (seconds)") 
 
    args = parser.parse_args() 
 
    if args.train: 
        assert args.data_path, "--data_path is required for training" 
        df = load_data(args.data_path) 
        train_and_evaluate(df, use_smote=args.smote, under_ratio=args.under_ratio) 
 
    if args.stream: 
        assert args.data_path, "--data_path is required for streaming" 
        stream_alerts(args.data_path, max_rows=args.max_rows, sleep=args.sleep) 
    if args.serve: 
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000) 
 
 
if __name__ == "__main__": 
    main() 
  