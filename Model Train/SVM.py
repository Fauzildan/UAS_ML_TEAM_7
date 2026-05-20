import numpy as np
import os
import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

base_dir = Path(__file__).resolve().parent.parent
fitur_dir = base_dir / "features"
model_dir = base_dir / "Model"

print("Loading data...")
X_train      = np.load(fitur_dir/"train_features.npy")
X_test       = np.load(fitur_dir/"test_features.npy")
y_train      = np.load(fitur_dir/"train_labels.npy")
y_test       = np.load(fitur_dir/"test_labels.npy")
class_names  = np.load(fitur_dir/"class_names.npy", allow_pickle=True)

print(f"X_train shape : {X_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"y_test  shape : {y_test.shape}")
print(f"Jumlah kelas  : {len(class_names)}")
print(f"Kelas         : {list(class_names)}")


# Normalisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

joblib.dump(scaler, model_dir / "scaler.joblib")
print("Fitur dinormalisasi dan scaler disimpan.")

# Training SVM
print("Training SVM...")

svm_model = SVC(
    kernel = "rbf",       
    C = 10.0,        
    gamma = "scale",     
    class_weight = "balanced",  
    probability = True,        
    random_state = 42,
    verbose = True,
    decision_function_shape = "ovr"
)
 
t0 = time.time()
svm_model.fit(X_train, y_train)
elapsed = time.time() - t0

print(f"Training selesai dalam {elapsed:.2f} detik.")

# Prediksi dan evaluasi
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(model_dir / "confusion_matrix.png")

# Simpan model SVM
joblib.dump(svm_model, model_dir / "svm_model.joblib")
print("Model SVM disimpan.")


