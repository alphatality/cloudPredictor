# Time-Series Incident Prediction

An **early-warning machine learning pipeline** designed to predict system incidents (e.g., CPU spikes, memory leaks, cascading failures) from time-series metrics.

---

## Problem Formulation

This problem is framed as a **sliding-window binary classification task**.  
Instead of reacting to incidents as they occur, the system predicts them **in advance**.

- **Look-back window (W):** 120 time steps (historical context)  
- **Prediction horizon (H):** 10 time steps (early warning lead time)  
- **Target:**  
  - `1` → if any incident occurs within the prediction horizon  
  - `0` → otherwise  

---

## Repository Structure

├── datagen.py # Synthetic  time-series generator
├── predictor.py # Feature extraction, model training, evaluation
└── utils.py # Model serialization utilities

## Modeling Choices

### 1. Feature Extraction vs. Raw Sequences

Instead of feeding raw time-series into the model, the pipeline computes **9 statistical features per metric** within each window:

- mean, std, min, max  
- last value  
- slope (trend)  
- 95th percentile  
- range  
- delta (last - first)  

**Why this approach?**

Tree-based models do not naturally capture temporal structure from raw sequences.  
Explicit features like **slope** and **delta** encode trends and significantly improve the detection of early warning signals while:
- reducing dimensionality  
- improving generalization  
- limiting overfitting 
Also added a cpu/latency to capture the correlation between the two 

---

### 2. Algorithm Selection

Two models are implemented:

- **Gradient Boosting Classifier**
- **Random Forest**

**Why Gradient Boosting?**
- Usually the best on such cases 
- Captures non-linear interactions between metrics  

**Why Random Forest**
- Easy rules
- low quantity of data

---

### 3. Handling Class Imbalance

Incidents represent approximately **3.5% of the dataset**.

**Approach:**
- Use `compute_sample_weight("balanced")`
- Pass weights directly to the training loss

**Why:**
- Prevents collapse into majority-class predictions  
- Strongly penalizes missed incidents (false negatives)  

---

## Evaluation Setup

### Strict Chronological Split

Random splits introduce **data leakage** in time-series problems.

We enforce:
- **75% training (past)**
- **25% testing (future)**

This ensures realistic evaluation on **unseen future data**.

---

### Metric Selection & Thresholding

Traditional metrics like accuracy and ROC-AUC are insufficient under heavy class imbalance.

#### Primary Metric
- **Precision-Recall AUC (PR-AUC)**  
  → Focuses on performance over the minority (incident) class  

#### Threshold Strategy
- Default threshold (`0.5`) would **not be used in production**
- Instead, we scan the Precision-Recall curve to enforce:

> **Target Recall = 70%**

This ensures the system prioritizes **incident detection over missed events**
I prefer having some missed accidents than triggering an alarm avery 5 minutes, because people would just not care anymore and shut it off.

---

### Model 1: Random Forest

| Metric | Default (t=0.50) | at racall = 70% (t=0.49) |
|---|---|---|
| **AUC-ROC** | 0.8684 | 0.8684 |
| **AUC-PR** | 0.8380 | 0.8380 |
| **Brier Score** | 0.1309 | 0.1309 |
| **F1 Score** | 0.7316 | 0.7337 | 
| **Precision** | 77.32% | 77.02% | 
| **Recall** | 69.42% | 70.05% | 
| **FPR** | 11.42% | 11.72% |


### Model 2: Gradient Boosting

| Metric | Default (t=0.50) | at racall = 70% (t=0.43) | 
|---|---|---|
| **AUC-ROC** | 0.8717 | 0.8717 |
| **AUC-PR** | 0.8322 | 0.8322 |
| **Brier Score** | 0.1301 | 0.1301 |
| **F1 Score** | 0.7276 | 0.7099 | 
| **Precision** | 89.14% | 71.96% |
| **Recall** | 61.47% | 70.05% | 
| **FPR** | 4.20% | 15.31% | 



### Interpretation

- The model is pretty sure when it detects a crisis but dont detect them all. To further improve the metrics we could monitor this scores for each type of incident


---

## Usage

### 1. Generate synthetic dataset

```bash
python datagen.py
```
```bash
python predictor.py
```
