from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

#----------LOAD DATA------------#
train_features = np.load("UAS_ML_TEAM_7/features/train_features.npy", allow_pickle=True)
print("Train Feature loaded")
train_labels   = np.load("UAS_ML_TEAM_7/features/train_labels.npy", allow_pickle=True)
print("Train Label loaded")
test_features  = np.load("UAS_ML_TEAM_7/features/test_features.npy",  allow_pickle=True)
print("Test Feature loaded")
test_labels    = np.load("UAS_ML_TEAM_7/features/test_labels.npy",  allow_pickle=True)
print("Test label loaded")
class_names    = np.load("UAS_ML_TEAM_7/features/class_names.npy",  allow_pickle=True)
print("Class names loaded")


train_features = train_features.reshape(train_features.shape[0], -1)
test_features  = test_features.reshape(test_features.shape[0], -1)
 
print(f"Train: {train_features.shape}, Test: {test_features.shape}")
print(f"Classes: {class_names}")

#----------TQDM CALLBACK--------#
class TqdmCallback(xgb.callback.TrainingCallback):
    def __init__(self, total):
        self.pbar = tqdm(total=total, desc="Training XGBoost", unit="round")

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

#----------TRAIN XGBOOST--------#
N_ESTIMATORS = 100

model = xgb.XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=6,
    learning_rate=0.1,
    objective="multi:softmax",
    num_class=len(class_names),
    use_label_encoder=False,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    callbacks=[TqdmCallback(total=N_ESTIMATORS)]
)

model.fit(train_features, train_labels)

#----------SAVE MODEL-----------#
model.save_model("UAS_ML_TEAM_7/Model/xgboost_model.json")
print("Model saved to /Model/xgboost_model.json")

#----------PREDICT--------------#
y_pred = model.predict(test_features)

#----------EVALUATE-------------#
f1     = f1_score(test_labels, y_pred, average="weighted")
cm     = confusion_matrix(test_labels, y_pred)
report = classification_report(test_labels, y_pred, target_names=class_names)

print(f"\nF1 Score (weighted): {f1:.4f}")
print("\nClassification Report:")
print(report)

#----------PLOT CONFUSION MATRIX#
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix - XGBoost")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("UAS_ML_TEAM_7/features/confusion_matrix_xgboost.png", dpi=150)
plt.show()
print("Confusion matrix saved to features/confusion_matrix_xgboost.png")