Markdown
# Component B: Hardware Selection and Justification

## 1. Constraint Triangle Application

To determine the optimal edge compute node for the FreightBridge cold-chain deployment, we evaluate three hardware candidates across the primary vertices of the Edge AI Constraint Triangle: **Compute Performance (SLA/Latency)**, **Power Consumption (Thermal/TDP)**, and **Fleet-Scale Cost**.

              Compute Performance (SLA / 90s)
                            /\
                           /  \
                          /    \
                         /      \
        (Option 2)      /________\      (Option 1)
       Jetson Orin Nano            Raspberry Pi 5 + Hailo-8L
             /                        \
            /                          \
           /____________________________\
Cost (Fleet)                                 Power (10W Budget)
(Option 3: STM32 MCU)


### Candidate Evaluation Matrix

| Metric / Constraint | Option 1: Raspberry Pi 5 + Hailo-8L | Option 2: Jetson Orin Nano Super | Option 3: STM32H7 Custom MCU |
| :--- | :--- | :--- | :--- |
| **Compute Capacity** | 13 TOPS (Dedicated NPU) | 67 TOPS (GPU Architecture) | Core-coupled CPU (No NPU) |
| **Power Draw (TDP)** | ~7.5W | ~15W (Moderate Load) | ~0.4W |
| **Unit Cost (INR)** | ~₹15,000 / truck | ~₹45,000 / truck | ~₹3,500 / truck |
| **85-Truck Pilot Cost**| ₹12,75,000 | ₹38,25,000 | ₹2,97,500 |
| **265-Fleet Scale Cost**| ₹39,75,000 | ₹1,19,25,000 | ₹9,27,500 |

### Operational Trade-off Analysis
The **dominant constraint vertex** for this deployment is **Compute Performance balanced with the physical 10W Automotive Power Budget**.

*   **Argument FOR Option 1 (Raspberry Pi 5 + Hailo-8L NPU):** This configuration represents the optimal Pareto-efficient choice for FreightBridge. At 7.5W, it operates safely within the 10W power budget supplied via the vehicle's DC-DC converter, avoiding high thermal stress in a sealed truck cabin. The 13 TOPS Hailo-8L coprocessor easily handles the 500 Hz multi-channel vibration stream and sliding-window feature extractions in milliseconds, guaranteeing deterministic adherence to the 90-second fault detection SLA. Financially, it scales within a viable CapEx profile (₹39.75 Lakhs for the full fleet).
*   **Argument AGAINST Option 2 (Jetson Orin Nano Super):** The Orin Nano is disqualified due to severe **Cost** and **Power** overruns. A total fleet rollout cost of ₹1.19 Crores introduces negative operational ROI for a 3-class classification problem. Furthermore, its 15W typical draw violates the physical 10W constraint, risking vehicle battery drain during ignition-off monitoring cycles and demanding complex active cooling configurations.
*   **Argument AGAINST Option 3 (STM32H7 MCU):** While highly cost-effective and low-power, the custom MCU fails the **Compute Performance** vertex. It lacks the SRAM capacity to buffer concurrent 30-second sliding windows for high-frequency (500 Hz) vibration data while maintaining a local MQTT broker layer, Python preprocessing dependencies, and the containerized MLOps architecture required by the pilot program. Thread starvation would trigger SLA breaches.

---

## 2. Arithmetic Intensity and Roofline Analysis

To predict how the model performs on the Raspberry Pi 5 host CPU, we map its computational profile against the broad hardware boundaries of the processor ($\text{Peak Performance} = 16\text{ GFLOP/s}$ via NEON SIMD; $\text{Memory Bandwidth} = 12\text{ GB/s}$ via LPDDR4X).

### Mathematical Calculations
*   **Model Computational Work ($W$):** 45 MFLOPs = $45 \times 10^6$ FLOPs per inference
*   **Model Memory Traffic ($Q$):** 18 MB = $18 \times 10^6$ Bytes (Weights + Activations)

$$I = \frac{W}{Q} = \frac{45 \times 10^6 \text{ FLOPs}}{18 \times 10^6 \text{ Bytes}} = 2.5\text{ FLOP/Byte}$$

The Operational Ridge Point ($I_{\text{ridge}}$) of the Raspberry Pi 5 CPU hardware structure is calculated as follows:

$$I_{\text{ridge}} = \frac{\text{Peak Performance}}{\text{Memory Bandwidth}} = \frac{16\text{ GFLOP/s}}{12\text{ GB/s}} \approx 1.33\text{ FLOP/Byte}$$

### Roofline Classification

Since the model's localized Arithmetic Intensity ($I = 2.5\text{ FLOP/Byte}$) is strictly **greater** than the hardware ridge point ($I_{\text{ridge}} \approx 1.33\text{ FLOP/Byte}$), the deployment falls definitively within the **Compute-Bound** execution zone of the Roofline model.

### Optimization Implications
Since performance is throttled by execution unit throughput rather than memory access latency, standard unstructured weight pruning or model compression without structural alterations will yield minimal execution speedups on the CPU. To reduce latency and satisfy constraints, we must target:
1.  **Post-Training Quantization (PTQ INT8):** Converting operations to 8-bit integers to harness higher-throughput ARM NEON integer SIMD pipelines.
2.  **Structural Filter Pruning:** Physically stripping out hidden layers or neurons t