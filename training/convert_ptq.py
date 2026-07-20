import os
import numpy as np
import tensorflow as tf

def representative_data_gen():
    # Load raw training arrays to extract calibration metrics
    X = np.load("training/X_train.npy")
    stats = np.load("data_pipeline/training_stats.npy", allow_pickle=True).item()
    
    # Normalize features using saved clean metrics (Task C2 Requirement)
    X_normalized = (X - stats["mean"]) / (stats["std"] + 1e-8)
    X_normalized = X_normalized.astype(np.float32)
    
    # Mandated constraint: provide >= 200 calibration samples for full INT8 PTQ
    for i in range(min(250, len(X_normalized))):
        yield [np.expand_dims(X_normalized[i], axis=0)]

def main():
    print("[OPTIMIZATION] Commencing Post-Training Quantization (PTQ) to Full INT8...")
    
    keras_model_path = "training/models/baseline_mlp.keras"
    if not os.path.exists(keras_model_path):
        print(f"[ERROR] Baseline model not found at {keras_model_path}. Run train_model.py first.")
        return

    # 1. Load the baseline FP32 Keras model
    model = tf.keras.models.load_model(keras_model_path)

    # 2. Configure the TFLite Converter for full fixed-point INT8 execution
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Enforce strict integer execution paths for edge hardware compatibility
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 3. Convert and write the optimized binary configuration
    tflite_model_int8 = converter.convert()
    
    output_path = "inference/model.tflite"
    with open(output_path, "wb") as f:
        f.write(tflite_model_int8)
        
    print(f"[SUCCESS] Full INT8 Quantized model written directly to: {output_path}")

if __name__ == "__main__":
    main()