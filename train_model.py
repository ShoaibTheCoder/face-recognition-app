"""
train_model.py
--------------
Trains Logistic Regression, kNN, and Decision Tree models on your
face dataset, evaluates all three, and saves the best one as
models/sklearn_model.pkl

Run AFTER you have collected at least 50 images per class:
  python train_model.py
"""

import os
import cv2
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR  = "dataset"
MODEL_OUT    = "models/sklearn_model.pkl"
IMG_SIZE     = (64, 64)   # resize all faces to this
LABELS       = {"ME": 1, "NOT_ME": 0}
# ──────────────────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_face_features(img_path: str) -> np.ndarray | None:
    """Load image → detect face → resize → flatten to feature vector."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
    if len(faces) == 0:
        # Fall back: use full image resized (no face detected)
        gray_resized = cv2.resize(gray, IMG_SIZE)
    else:
        x, y, w, h = faces[0]
        gray_resized = cv2.resize(gray[y:y+h, x:x+w], IMG_SIZE)
    return gray_resized.flatten().astype(np.float32) / 255.0


def load_dataset():
    X, y = [], []
    for label_name, label_val in LABELS.items():
        folder = os.path.join(DATASET_DIR, label_name)
        if not os.path.exists(folder):
            print(f"[WARN] Folder not found: {folder} — skipping")
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"[INFO] Loading {len(files)} images from '{label_name}'...")
        for fname in files:
            feat = extract_face_features(os.path.join(folder, fname))
            if feat is not None:
                X.append(feat)
                y.append(label_val)
            else:
                print(f"  [SKIP] {fname}")
    return np.array(X), np.array(y)


def train_and_evaluate():
    print("\n" + "="*55)
    print("  FACE RECOGNITION — SKLEARN TRAINING PIPELINE")
    print("="*55)

    X, y = load_dataset()
    print(f"\n[INFO] Dataset loaded: {len(X)} samples, "
          f"{y.tolist().count(1)} ME / {y.tolist().count(0)} NOT_ME\n")

    if len(X) < 20:
        print("[ERROR] Not enough data. Collect at least 50 images per class.")
        return

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Models ──────────────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0,
                                                   random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10,
                                                       random_state=42),
    }

    results = {}
    print(f"{'Model':<25} {'CV Acc':>8} {'Test Acc':>10}")
    print("-" * 45)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        test_acc  = accuracy_score(y_test, y_pred)
        results[name] = {
            "model":    model,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "test_acc": test_acc,
            "y_pred":   y_pred,
        }
        print(f"  {name:<23} {cv_scores.mean():.4f}    {test_acc:.4f}")

    # ── Best model ──────────────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["test_acc"])
    best      = results[best_name]
    print(f"\n[BEST] {best_name} (test accuracy: {best['test_acc']:.4f})\n")

    print(f"Classification Report — {best_name}")
    print(classification_report(y_test, best["y_pred"],
                                 target_names=["NOT_ME", "ME"]))

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    payload = {
        "model":        best["model"],
        "scaler":       scaler,
        "model_name":   best_name,
        "img_size":     IMG_SIZE,
        "labels":       {v: k for k, v in LABELS.items()},  # {0:"NOT_ME",1:"ME"}
        "test_accuracy": best["test_acc"],
    }
    joblib.dump(payload, MODEL_OUT)
    print(f"[SAVED] Model → {MODEL_OUT}")

    # ── Plots ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart comparing all models
    names  = list(results.keys())
    accs   = [results[n]["test_acc"] for n in names]
    colors = ["#4CAF50" if n == best_name else "#90CAF9" for n in names]
    axes[0].bar(names, accs, color=colors)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Model Comparison — Test Accuracy")
    axes[0].set_ylabel("Accuracy")
    for i, v in enumerate(accs):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["NOT_ME", "ME"])
    disp.plot(ax=axes[1], colorbar=False)
    axes[1].set_title(f"Confusion Matrix — {best_name}")

    plt.tight_layout()
    plot_path = "models/training_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[SAVED] Plot  → {plot_path}")
    plt.show()
    print("\n[DONE] Training complete.")


if __name__ == "__main__":
    train_and_evaluate()
