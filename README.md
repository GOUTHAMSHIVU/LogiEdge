
## 👥 Group Details (Group 22)

* **GOUTHAM S** (2024AD05034)
* **MAUSAM JAIN** (2024AD05001)
* **SINGH YASHPAL JILA** (2024AC05496)
* **SUDHARSON R R** (2024AC05952)

---

```markdown
# LogiBridge — Edge AI & Automated DevOps for Cold-Chain Logistics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Compose](https://img.shields.io/badge/Docker--Compose-v2-blue)](https://docs.docker.com/compose/)
[![Ansible](https://img.shields.io/badge/Ansible-IaC-red)](https://www.ansible.com/)
[![TFLite](https://img.shields.io/badge/TensorFlow--Lite-INT8-orange)](https://www.tensorflow.org/lite)

LogiBridge is a microservice-based, containerized Edge AI telemetry monitoring system designed for cold-chain logistics. By processing multi-sensor telemetry directly at the edge, LogiBridge eliminates cloud latency and network vulnerabilities, executing real-time anomaly detection and statistical model drift analysis on local nodes.

---

## 🏛️ System Architecture

LogiBridge deploys four isolated Docker microservices communicating asynchronously via an internal MQTT bridge network (`logibridge_net`):

```text
+------------------------+      Raw Telemetry      +-----------------------+
|  logibridge_simulator  | ----------------------> |    logibridge_mqtt    |
| (Data / Fault Pipeline)|                         |   (Mosquitto Broker)  |
+------------------------+                         +-----------+-----------+
                                                               ^     |
                                              Telemetry (Sub)  |     | Predictions (Sub)
                                            Predictions (Pub)  v     v
                            +------------------------------+   +----------------------------------+
                            |     logibridge_inference     |   |    logibridge_drift_monitor     |
                            |   (TFLite Engine & Fusion)   |   |   (KS Drift & Baseline Stat)     |
                            +------------------------------+   +----------------------------------+

```

### Microservice Specifications

| Container Name | Technology Stack | Role & Function |
| --- | --- | --- |
| **`logibridge_mqtt`** | Eclipse Mosquitto (v2.0) | Central MQTT message broker handling internal sensor/inference routing (`1883/TCP`). |
| **`logibridge_simulator`** | Python 3.10, Paho MQTT | Simulates multi-sensor streams (temperature, vibration, door state) & fault injections. |
| **`logibridge_inference`** | Python 3.10, TFLite Runtime | Aggregates sliding window features, normalizes inputs, and runs local TFLite forward pass. |
| **`logibridge_drift_monitor`** | Python 3.10, SciPy, NumPy | Computes real-time Kolmogorov-Smirnov (KS) and PSI tests to flag input telemetry drift. |

---

## ⚡ Edge AI Model Optimization Benchmarks

LogiBridge implements an optimized **Train $\rightarrow$ Prune $\rightarrow$ Quantize** pipeline to fit stringent edge hardware budgets without sacrificing accuracy:

| Model Variant | Accuracy (%) | Critical Recall (%) | Mean Latency (ms) | Model Size (KB) | Energy / Inf (mJ) |
| --- | --- | --- | --- | --- | --- |
| **M1: FP32 Baseline** | $94.92\%$ | $88.24\%$ | $52.87\text{ ms}$ | $32.78\text{ KB}$ | $274.41\text{ mJ}$ |
| **M2: PTQ INT8** | $91.53\%$ | $82.35\%$ | $0.032\text{ ms}$ | $3.38\text{ KB}$ | $0.474\text{ mJ}$ |
| **M3: Pruned + INT8** *(Deployed)* | **$93.22\%$** | **$82.35\%$** | **$0.033\text{ ms}$** | **$3.38\text{ KB}$** | **$0.005\text{ mJ}$** |

* **Footprint Reduction:** **$89.7\%$** size reduction ($32.78\text{ KB} \rightarrow 3.38\text{ KB}$).
* **Latency & Energy:** Slashing inference times to **$\sim 0.033\text{ ms}$** per prediction with **$>99.9\%$ energy savings**.

---

## 🚀 Quick Start

### Prerequisites

* Docker Engine (v20.10+) & Docker Compose v2
* Python 3.10+ (for local development/dataset generation)
* Ansible (optional, for automated IaC deployment)

### 1. Automated IaC Deployment via Ansible (Recommended)

Deploy the full stack idempotently using Ansible:

```bash
ansible-playbook logibridge_deploy.yml

```

### 2. Manual Docker Deployment

Alternatively, bring up the microservice stack directly via Docker Compose:

```bash
# Clone repository
git clone [https://github.com/your-org/logibridge.git](https://github.com/your-org/logibridge.git)
cd logibridge

# Build and launch microservices
docker compose up -d --build

# Verify container status
docker ps

```

---

## 🧪 Fault Injection & Verification

Validate the edge inference engine and real-time drift monitor by injecting simulated operational failures into the running simulator service:

### 1. Thermal Drift Injection

Simulates a refrigeration cooling failure (linear increment of $+0.08^\circ\text{C}$ per sample):

```bash
docker exec -it logibridge_simulator python3 data_pipeline/simulator.py --anomaly temp_drift

```

### 2. Combined Vibration & Door Fault

Simulates compressor bearing wear combined with door latch failure:

```bash
docker exec -it logibridge_simulator python3 data_pipeline/simulator.py --anomaly combined

```

### 3. Observe Inference & Drift Output

Inspect incoming feature vectors and state classifications in real time:

```bash
docker logs -f logibridge_inference

```

Inspect Kolmogorov-Smirnov statistical divergence alerts:

```bash
docker logs -f logibridge_drift_monitor

```

---

## 📂 Project Structure

```text
logibridge/
├── ansible/
│   └── logibridge_deploy.yml   # Infrastructure as Code deployment playbook
├── data_pipeline/
│   ├── generate_dataset.py     # Synthetic data & statistical parameter generator
│   ├── simulator.py            # Real-time MQTT telemetry & fault injector
│   └── training_stats.npy      # Feature normalization calibration parameters (Mean/Std)
├── inference/
│   ├── inference.py            # TFLite sliding window inference service
│   └── model.tflite            # Quantized INT8 edge model asset
├── models/
│   ├── train_model.py          # Baseline float32 training pipeline
│   ├── prune.py                # Structural weight pruning
│   └── convert_ptq.py          # Post-Training Quantization (PTQ) conversion
├── monitoring/
│   └── drift_monitor.py        # Real-time KS & PSI statistical drift service
├── docker-compose.yml          # Multi-container orchestration specification
└── README.md

```

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

```

***

<ElicitationsGroup message="What would you like to do next?">
<Elicitation label="Add a License section or LICENSE file template" query="Generate an MIT License file for LogiBridge" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Create a GitHub Actions CI/CD pipeline for docker build verification" query="Draft a GitHub Actions workflow for LogiBridge Docker builds" query_intent="CLICKABLE_SUGGESTION" />
<Elicitation label="Generate environment variable template file (.env.example)" query="Draft a .env.example file for LogiBridge" query_intent="CLICKABLE_SUGGESTION" />
</ElicitationsGroup>

```
