# Component C: Sensor Pipeline and MQTT Architecture

## 1. MQTT Topic Tree Topology

logibridge/
└── trucks/
└── {truck_id}/
├── sensors/
│   ├── temperature      [QoS 1]
│   ├── vibration        [QoS 0]
│   └── door             [QoS 1]
└── alerts               [QoS 2]


### Quality of Service (QoS) Strategy
*   **`logibridge/trucks/{truck_id}/sensors/temperature` (QoS 1 - At Least Once):** Guarantees that slow-moving but critical pharmaceutical thermal fluctuations are registered by the local broker without packet drops.
*   **`logibridge/trucks/{truck_id}/sensors/vibration` (QoS 0 - At Most Once):** Configured for high-frequency telemetric updates (0.5 Hz RMS calculation derived from 500 Hz raw stream). Dropping an occasional message is acceptable, whereas enforcing handshakes at this scale would risk network traffic saturation and broker buffering lag.
*   **`logibridge/trucks/{truck_id}/sensors/door` (QoS 1 - At Least Once):** Door open/close configurations represent discrete events that alter thermal containment dynamics. These events must be delivered reliably.
*   **`logibridge/trucks/{truck_id}/alerts` (QoS 2 - Exactly Once):** Reserved exclusively for System St