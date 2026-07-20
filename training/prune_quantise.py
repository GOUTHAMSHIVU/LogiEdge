import os
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.model_selection import train_test_split

def representative_data_gen():
    # Calibration only needs input statistics (min/max ranges) for quantization —
    # no labels or gradients involved, so using the full training pool here is
    # fine and consistent with convert_ptq.py's calibration approach.
    X = np.load("training/X_train.npy")
    stats = np.load("data_pipeline/training_stats.npy", allow_pickle=True).item()
    X_normalized = (X - stats["mean"]) / (stats["std"] + 1e-8)
    X_normalized = X_normalized.astype(np.float32)

    for i in range(min(250, len(X_normalized))):
        yield [np.expand_dims(X_normalized[i], axis=0)]

def main():
    print("[OPTIMIZATION] Commencing 35% Structured Pruning + Full INT8 PTQ Pipeline...")

    keras_model_path = "training/models/baseline_mlp.keras"
    if not os.path.exists(keras_model_path):
        print(f"[ERROR] Baseline model not found at {keras_model_path}. Run train_model.py first.")
        return

    # 1. Load data and reproduce the EXACT same train/val split used in
    #    train_model.py (same random_state + stratify), so the pruning
    #    fine-tune below never sees the rows held out in X_val.npy/y_val.npy.
    #    Fine-tuning on validation rows would leak them into the model and
    #    inflate M3's benchmarked accuracy artificially.
    X = np.load("training/X_train.npy")
    y = np.load("training/y_train.npy")
    stats = np.load("data_pipeline/training_stats.npy", allow_pickle=True).item()
    X_normalized = (X - stats["mean"]) / (stats["std"] + 1e-8)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=3)

    X_train, X_val, y_train, y_val = train_test_split(
        X_normalized, y_onehot, test_size=0.20, random_state=42, stratify=y
    )

    base_model = tf.keras.models.load_model(keras_model_path)

    # 2. Define the structural 35% pruning schedule constraint
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.35,  # Target 35% structural pruning
            begin_step=0,
            end_step=100
        )
    }

    # Apply pruning wrappers to the network layers
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
    pruned_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Short internal calibration fine-tune to restore structure thresholds —
    # now trained ONLY on X_train, never on the held-out validation rows
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5, batch_size=16, callbacks=callbacks, verbose=0
    )

    val_loss, val_acc = pruned_model.evaluate(X_val, y_val, verbose=0)
    print(f"[OPTIMIZATION] Post-pruning validation accuracy (held-out, uncontaminated): {val_acc * 100:.2f}%")

    # Strip the pruning meta-wrappers to leave clean pruned weight matrices
    stripped_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # 3. Apply Post-Training Quantization over the pruned structure
    converter = tf.lite.TFLiteConverter.from_keras_model(stripped_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_pruned_int8 = converter.convert()

    # Store locally for the future Pareto frontier benchmarking metrics step
    output_dir = "training/models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pruned_quantized_model.tflite")

    with open(output_path, "wb") as f:
        f.write(tflite_pruned_int8)

    print(f"[SUCCESS] M3 Pruned + INT8 TFLite model written to: {output_path}")

if __name__ == "__main__":
    main()