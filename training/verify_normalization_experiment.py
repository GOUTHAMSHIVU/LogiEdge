import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, recall_score

def main():
    try:
        # Load trained Keras model, normalized validation features, and ground truth
        model = tf.keras.models.load_model("training/models/baseline_mlp.keras")
        X_val_norm = np.load("training/X_val.npy")   # Pre-normalized validation split
        y_val_int = np.load("training/y_val.npy")    # Ground truth labels (0, 1, 2)
    except FileNotFoundError as e:
        print(f"[ERROR] Required asset missing: {e}")
        return

    print("=================================================================")
    print("TASK C2: NORMALIZATION PERTURBATION EXPERIMENT VERIFICATION")
    print("=================================================================\n")

    # --- 1. BASELINE EVALUATION (Raw model outputs, no multiplier) ---
    raw_probs_correct = model.predict(X_val_norm, verbose=0)
    preds_correct = np.argmax(raw_probs_correct, axis=1)

    acc_correct = accuracy_score(y_val_int, preds_correct)
    rec_correct = recall_score(y_val_int, preds_correct, labels=[0, 1, 2], average=None)

    print("--- 1. BASELINE EVALUATION (Correct stats) ---")
    print(f"Overall Accuracy:           {acc_correct * 100:.2f}%")
    print(f"Normal (0) Accuracy/Recall:  {rec_correct[0] * 100:.2f}%")
    print(f"Warning (1) Accuracy/Recall: {rec_correct[1] * 100:.2f}%")
    print(f"Critical (2) Recall:         {rec_correct[2] * 100:.2f}%\n")

    # --- 2. SHIFTED EVALUATION (+3 Sigma Shifted Mean -> X_norm - 3.0) ---
    X_val_shifted = X_val_norm - 3.0

    raw_probs_shifted = model.predict(X_val_shifted, verbose=0)
    preds_shifted = np.argmax(raw_probs_shifted, axis=1)

    acc_shifted = accuracy_score(y_val_int, preds_shifted)
    rec_shifted = recall_score(y_val_int, preds_shifted, labels=[0, 1, 2], average=None)

    print("--- 2. SHIFTED EVALUATION (+3 Sigma Shifted stats) ---")
    print(f"Overall Accuracy:           {acc_shifted * 100:.2f}%")
    print(f"Normal (0) Accuracy/Recall:  {rec_shifted[0] * 100:.2f}%")
    print(f"Warning (1) Accuracy/Recall: {rec_shifted[1] * 100:.2f}%")
    print(f"Critical (2) Recall:         {rec_shifted[2] * 100:.2f}%\n")

    # --- SUMMARY TABLE VALUES FOR REPORT ---
    print("=================================================================")
    print("SUMMARY TABLE VALUES FOR REPORT (TASK C2)")
    print("=================================================================")
    print(f"Normal (0) Delta:   {rec_correct[0]*100:.2f}%  -->  {rec_shifted[0]*100:.2f}%  (Drop: {(rec_correct[0]-rec_shifted[0])*100:.2f}%)")
    print(f"Warning (1) Delta:  {rec_correct[1]*100:.2f}%  -->  {rec_shifted[1]*100:.2f}%  (Drop: {(rec_correct[1]-rec_shifted[1])*100:.2f}%)")
    print(f"Critical (2) Delta: {rec_correct[2]*100:.2f}%  -->  {rec_shifted[2]*100:.2f}%  (Drop: {(rec_correct[2]-rec_shifted[2])*100:.2f}%)")
    print(f"Overall Acc Delta:  {acc_correct*100:.2f}%  -->  {acc_shifted*100:.2f}%  (Drop: {(acc_correct-acc_shifted)*100:.2f}%)")
    print("=================================================================")

if __name__ == "__main__":
    main()