import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def main():
    # Load dataset generated in step 1
    try:
        X = np.load("training/X_train.npy")
        y = np.load("training/y_train.npy")
        stats = np.load("data_pipeline/training_stats.npy", allow_pickle=True).item()
    except FileNotFoundError:
        print("[ERROR] Execute generate_dataset.py before training.")
        return

    # Normalize the data using the baseline Normal class stats (Task C2 requirement)
    X_normalized = (X - stats["mean"]) / (stats["std"] + 1e-8)

    # Convert labels to one-hot vectors for multiclass categorical crossentropy
    y_onehot = to_categorical(y, num_classes=3)

    # 20% validation split as mandated by brief
    # NOTE: split X_normalized/y (int labels) AND y_onehot together so both
    # representations stay row-aligned with the same held-out samples.
    X_train, X_val, y_train, y_val, y_train_int, y_val_int = train_test_split(
        X_normalized, y_onehot, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build standard 2-hidden-layer MLP network architecture
    model = Sequential([
        Dense(32, activation='relu', input_shape=(6,)), # Hidden Layer 1: 32 units
        Dense(16, activation='relu'),                   # Hidden Layer 2: 16 units
        Dense(3, activation='softmax')                  # Output Layer: 3 target classes
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("[TRAINING] Commencing model optimization...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=1
    )

    val_acc = history.history['val_accuracy'][-1]
    print(f"\n[TRAINING] Final Validation Accuracy: {val_acc * 100:.2f}%")

    if val_acc >= 0.88:
        print("[SUCCESS] Quality threshold satisfied (>88%). Saving model structural files...")
        model.save("training/models/baseline_mlp.keras")

        # Persist the held-out split so downstream scripts (benchmark.py, PSI
        # monitor, etc.) evaluate on data the model never trained on.
        # - X_val / y_val_onehot: for Keras .evaluate() (M1)
        # - y_val_int: plain class indices, for the TFLite argmax-comparison
        #   loops used to evaluate M2/M3
        np.save("training/X_val.npy", X_val)
        np.save("training/y_val_onehot.npy", y_val)
        np.save("training/y_val.npy", y_val_int)
        print("[SUCCESS] Held-out validation split saved to training/X_val.npy, "
              "training/y_val.npy, training/y_val_onehot.npy")
    else:
        print("[FAIL] Accuracy fell below the 88% structural threshold. Re-verify feature scaling.")

if __name__ == "__main__":
    main()