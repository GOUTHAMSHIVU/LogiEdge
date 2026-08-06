# monitoring/generate_reference_dist.py
import json
import numpy as np
import tensorflow.lite as tflite

MODEL_PATH = "inference/model.tflite"
STATS_PATH = "data_pipeline/training_stats.npy"
OUTPUT_PATH = "monitoring/reference_dist.json"
N_SAMPLES = 300
BIN_EDGES = [0.0, 0.25, 0.50, 0.75, 1.0]

def generate_clean_normal_features(n):
    # Same distribution as generate_dataset.py's "none" mode
    t_mean = np.random.normal(4.0, 0.05, n)
    t_std = np.random.normal(0.3, 0.02, n)
    t_rate = np.random.normal(0.0, 0.01, n)
    v_rms = np.random.normal(0.45, 0.02, n)
    v_peak = v_rms * 1.414
    v_kurt = np.random.normal(0.0, 0.1, n)
    return np.stack([t_mean, t_std, t_rate, v_rms, v_peak, v_kurt], axis=1)

def main():
    stats = np.load(STATS_PATH, allow_pickle=True).item()
    X = np.load("training/windows_none.npy")
    print(f"Number of normal windows: {len(X)}")
    if len(X) > N_SAMPLES:
        idx = np.random.choice(len(X), N_SAMPLES, replace=False)
        X = X[idx]
    X_norm = ((X - stats["mean"]) / (stats["std"] + 1e-8)).astype(np.float32)
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    in_d = interpreter.get_input_details()
    out_d = interpreter.get_output_details()
    in_scale, in_zp = in_d[0]["quantization"]
    out_scale, out_zp = out_d[0]["quantization"]

    confidences = []
    for row in X_norm:
        q_in = np.clip((row / in_scale) + in_zp, -128, 127).astype(np.int8)
        q_in = np.expand_dims(q_in, axis=0)
        interpreter.set_tensor(in_d[0]["index"], q_in)
        interpreter.invoke()
        q_out = interpreter.get_tensor(out_d[0]["index"])[0]
        probs = (q_out.astype(np.float32) - out_zp) * out_scale
        confidences.append(float(np.max(probs)))

    counts, _ = np.histogram(confidences, bins=BIN_EDGES)
    pct = (counts / len(confidences)).tolist()

    ref = {
        "bin_edges": BIN_EDGES,
        "bin_percentages": pct,
        "n_samples": len(X)
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"[SUCCESS] Reference distribution saved to {OUTPUT_PATH}: {pct}")
    print("Confidence statistics")
    print("Min :", np.min(confidences))
    print("Max :", np.max(confidences))
    print("Mean:", np.mean(confidences))
    print("Unique (rounded):", np.unique(np.round(confidences, 3)))

if __name__ == "__main__":
    main()