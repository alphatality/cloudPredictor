import numpy as np
import pandas as pd
import utils
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score,average_precision_score,precision_recall_curve,confusion_matrix,brier_score_loss)
from sklearn.utils.class_weight import compute_sample_weight

seed = 438
methode = "gradient"  # "tree" or "gradient"
target_recall = 0.70
 
df = pd.read_csv("data/synthetic_cloud_data.csv")
 
STAT_NAMES = ["mean", "std", "min", "max", "last", "slope", "p95", "range", "delta"]
metric_cols = ["cpu", "memory", "latency"]
W = 120
H = 10
n_estimators=300
 
 
def window_features(w):
    W = len(w)
    x = np.arange(W, dtype=float)
    slope = np.polyfit(x, w, 1)[0]
    return np.array([w.mean(),w.std(),w.min(),w.max(),w[-1],slope,np.percentile(w, 95),w.max() - w.min(),w[-1] - w[0]])
 
 
def build_windows(df,W= 60,H = 10, metric_cols= None):
    if metric_cols is None:
        metric_cols = ["cpu", "memory", "latency"]
 
    vals = df[metric_cols].values.astype(float)
    inc =df["incident"].values
    T,K = vals.shape
 
    feat_names = [f"{col}_{stat}" for col in metric_cols for stat in STAT_NAMES]
 
    X_rows = []
    y_rows = []
 
    for t in range(W, T - H):
        window = vals[t - W : t]
        label = int(inc[t : t + H].max())        # 1 if any incident ahead
 
        feats = np.concatenate([window_features(window[:, k]) for k in range(K)])
        X_rows.append(feats)
        y_rows.append(label)
 
    return np.array(X_rows), np.array(y_rows), feat_names
 
def threshold_for_recall(y_true, y_prob, target_recall=0.85):
    _, rec, thresholds = precision_recall_curve(y_true, y_prob)
    valid = thresholds[rec[:-1] >= target_recall]
    if len(valid) == 0:
        return float(thresholds[0]) 
    return float(valid.max())
 
def evaluate(y_true, y_prob, threshold, label=""):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp / max(tp + fp, 1)
    rec= tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "label":     label,
        "threshold": round(threshold, 3),
        "AUC-ROC":round(roc_auc_score(y_true, y_prob), 4),
        "AUC-PR":round(average_precision_score(y_true, y_prob), 4),
        "Brier":round(brier_score_loss(y_true, y_prob), 4),
        "F1":round(f1, 4),
        "Precision":round(prec, 4),
        "Recall":round(rec, 4),
        "FPR":round(fp / max(fp + tn, 1), 4),
        "TP": int(tp), 
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
 
def main(df, methode="tree") :
    X, y, feat_names = build_windows(df, W=W, H=H)
 
    # temporal split
    split  = int(0.75 * len(y))
    X_tr, X_te = X[:split-H], X[split:]
    y_tr, y_te = y[:split-H], y[split:]
 
    print(f"Dataset : {len(df):,} steps | {len(y):,} windows")
    print(f"Labels  : {y.mean()*100:.1f}% positive")
    print(f"Train / Test : {len(y_tr):,} / {len(y_te):,}\n")
 
    if methode == "gradient": 
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_tr)

        clf = GradientBoostingClassifier(
            n_estimators=n_estimators*5, 
            learning_rate=0.01,
            max_depth=4,
            max_features="sqrt",
            min_samples_leaf=10,
            subsample=0.8,
            random_state=seed,
            validation_fraction=0.1,
            n_iter_no_change=20,     
            tol=0.01
        )
        
        clf.fit(X_tr, y_tr, sample_weight=sample_weights)
        print(f"\n[Info] Trees build : {clf.n_estimators_} / {5*n_estimators}")
    else:
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
    
    y_prob = clf.predict_proba(X_te)[:, 1]
    thr_default = 0.5
    thr_op=threshold_for_recall(y_te, y_prob, target_recall=target_recall)
    utils.save_model(clf, methode=methode, thr_op=thr_op,)
    res_default= evaluate(y_te, y_prob, thr_default,"Default   t=0.50")
    res_op = evaluate(y_te, y_prob, thr_op,f"Op. point t={thr_op:.2f} (recall≥{int(100*target_recall)}%)")
 
    # --- print results ---
    cols = ["AUC-ROC", "AUC-PR", "Brier", "F1", "Precision", "Recall", "FPR"]
    header = f"{'':28}" + "".join(f"{c:>10}" for c in cols)
    sep    = "─" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in [res_default, res_op]:
        print(f"  {r['label']:<26}" + "".join(f"{r[c]:>10.4f}" for c in cols))
    print(sep)
 
    r = res_op
    print(f"\nConfusion matrix @ t={thr_op:.2f}")
    print(f"  {'':14} Pred-0   Pred-1")
    print(f"  {'Actual-0':14} {r['TN']:6}   {r['FP']:6}   FPR    = {r['FPR']:.3f}")
    print(f"  {'Actual-1':14} {r['FN']:6}   {r['TP']:6}   Recall = {r['Recall']:.3f}")

main(df,methode)