import os
import time
import csv
import numpy as np
import psutil
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Ensure output directories exist structurally
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "benchmark_results.csv")
CHART_PATH = os.path.join(RESULTS_DIR, "pareto_chart.png")

# Lab 2 methodology constants
WARMUP_RUNS = 10
TIMED_RUNS = 200

# Laptop TDP estimate for energy calculation (adjust to your actual hardware)
LAPTOP_TDP_WATTS = 15.0


def get_file_size(path):
    if not os.path.exists(path):
        return 0.0
    return os.path.getsize(path) / 1024.0  # Convert to KB


def _cycle_indices(n_available, n_needed):
    """Cycle through a (possibly small) validation set to reach n_needed
    sample draws, e.g. a 60-row val set repeated to cover 210 runs."""
    return [i % n_available for i in range(n_needed)]


def evaluate_keras(model_path, X_val, y_val_onehot):
    if not os.path.exists(model_path):
        print(f"[WARN] Target model missing: {model_path}")
        return 0.0, 0.0, 0.0, 0.0

    model = load_model(model_path)
    n = len(X_val)
    idxs = _cycle_indices(n, WARMUP_RUNS + TIMED_RUNS)

    # Warmup — excluded from timing
    for i in idxs[:WARMUP_RUNS]:
        _ = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)

    # Timed runs — pure inference call only, no accuracy bookkeeping inside
    latencies = []
    cpu_before = psutil.cpu_percent(interval=None)
    for i in idxs[WARMUP_RUNS:]:
        t0 = time.perf_counter()
        _ = model.predict(np.expand_dims(X_val[i], axis=0), verbose=0)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
    cpu_after = psutil.cpu_percent(interval=None)

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Accuracy measured separately, on the FULL held-out set, outside the timed loop
    _, acc = model.evaluate(X_val, y_val_onehot, verbose=0)

    # Energy per inference: E = P * t (mJ), using measured CPU% as a load proxy
    avg_cpu_frac = max(cpu_after, 1.0) / 100.0  # avoid a zero-power estimate
    energy_mj = LAPTOP_TDP_WATTS * avg_cpu_frac * (mean_latency / 1000.0) * 1000.0

    return acc * 100.0, mean_latency, p95_latency, energy_mj


def evaluate_tflite(model_path, X_val, y_val_int):
    if not os.path.exists(model_path):
        print(f"[WARN] Target model missing: {model_path}")
        return 0.0, 0.0, 0.0, 0.0

    import tensorflow.lite as tflite
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_scale, in_zp = input_details[0]['quantization']
    out_scale, out_zp = output_details[0]['quantization']

    def run_one(row):
        q_in = (row / in_scale) + in_zp
        q_in = np.clip(q_in, -128, 127).astype(np.int8)
        q_in = np.expand_dims(q_in, axis=0)
        interpreter.set_tensor(input_details[0]['index'], q_in)
        interpreter.invoke()
        q_out = interpreter.get_tensor(output_details[0]['index'])[0]
        return (q_out.astype(np.float32) - out_zp) * out_scale

    n = len(X_val)
    idxs = _cycle_indices(n, WARMUP_RUNS + TIMED_RUNS)

    # Warmup — excluded from timing
    for i in idxs[:WARMUP_RUNS]:
        _ = run_one(X_val[i])

    # Timed runs — pure inference call only, no accuracy bookkeeping inside
    latencies = []
    cpu_before = psutil.cpu_percent(interval=None)
    for i in idxs[WARMUP_RUNS:]:
        t0 = time.perf_counter()
        _ = run_one(X_val[i])
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms
    cpu_after = psutil.cpu_percent(interval=None)

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    # Accuracy measured separately, on the FULL held-out set, outside the timed loop
    correct = 0
    for i in range(n):
        probs = run_one(X_val[i])
        if np.argmax(probs) == y_val_int[i]:
            correct += 1
    accuracy = (correct / n) * 100.0

    avg_cpu_frac = max(cpu_after, 1.0) / 100.0
    energy_mj = LAPTOP_TDP_WATTS * avg_cpu_frac * (mean_latency / 1000.0) * 1000.0

    return accuracy, mean_latency, p95_latency, energy_mj


def generate_pareto_chart(results):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    models = [r["Model Variant"] for r in results]
    latencies = [r["Mean Latency (ms)"] for r in results]
    sizes = [r["Size (KB)"] for r in results]
    accuracies = [r["Accuracy (%)"] for r in results]

    color = '#1f77b4'
    ax1.set_xlabel('Model Architectures & Compressions', fontweight='bold', labelpad=12)
    ax1.set_ylabel('Mean Latency (ms per inference)', color=color, fontweight='bold')
    bars = ax1.bar(models, latencies, color=color, alpha=0.6, width=0.4, label='Latency (ms)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    for bar, size in zip(bars, sizes):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(latencies) * 0.02 + 1e-6),
                  f"{size:.1f} KB", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Validation Accuracy (%)', color=color, fontweight='bold')
    ax2.plot(models, accuracies, color=color, marker='o', linewidth=2.5, markersize=8, label='Accuracy (%)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(min(accuracies) - 5, 105)

    plt.title('LogiBridge Edge AI Pareto Optimization Frontier — Held-Out Validation Set',
              fontweight='bold', fontsize=14, pad=15)
    fig.tight_layout()

    plt.savefig(CHART_PATH, dpi=300)
    print(f"[SUCCESS] Optimization visualization rendered directly to: {CHART_PATH}")
    plt.close()


def main():
    print("=" * 70)
    print("Executing Structural Pareto Metrics Profiling (held-out validation set)...")
    print("=" * 70)

    try:
        X_val = np.load("training/X_val.npy")
        y_val_onehot = np.load("training/y_val_onehot.npy")
        y_val = np.load("training/y_val.npy")
    except FileNotFoundError:
        print("[ERROR] Held-out validation split not found. Run train_model.py first "
              "(it now saves training/X_val.npy, y_val.npy, y_val_onehot.npy).")
        return

    m1_acc, m1_lat, m1_p95, m1_energy = evaluate_keras(
        "training/models/baseline_mlp.keras", X_val, y_val_onehot)
    m1_size = get_file_size("training/models/baseline_mlp.keras")

    m2_acc, m2_lat, m2_p95, m2_energy = evaluate_tflite(
        "inference/model.tflite", X_val, y_val)
    m2_size = get_file_size("inference/model.tflite")

    m3_acc, m3_lat, m3_p95, m3_energy = evaluate_tflite(
        "training/models/pruned_quantized_model.tflite", X_val, y_val)
    m3_size = get_file_size("training/models/pruned_quantized_model.tflite")

    results = [
        {"Model Variant": "M1: FP32 Baseline", "Accuracy (%)": m1_acc,
         "Mean Latency (ms)": m1_lat, "p95 Latency (ms)": m1_p95,
         "Size (KB)": m1_size, "Energy (mJ)": m1_energy},
        {"Model Variant": "M2: PTQ INT8", "Accuracy (%)": m2_acc,
         "Mean Latency (ms)": m2_lat, "p95 Latency (ms)": m2_p95,
         "Size (KB)": m2_size, "Energy (mJ)": m2_energy},
        {"Model Variant": "M3: Pruned + INT8", "Accuracy (%)": m3_acc,
         "Mean Latency (ms)": m3_lat, "p95 Latency (ms)": m3_p95,
         "Size (KB)": m3_size, "Energy (mJ)": m3_energy},
    ]

    with open(CSV_PATH, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[SUCCESS] Tabular evaluation data saved to: {CSV_PATH}")

    generate_pareto_chart(results)


if __name__ == "__main__":
    main()