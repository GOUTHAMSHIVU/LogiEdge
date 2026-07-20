# monitoring/drift_monitor.py
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque

TRUCK_ID = "01"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_SUB_INF = f"logibridge/trucks/{TRUCK_ID}/inference"
REF_PATH = "monitoring/reference_dist.json"

CHECK_INTERVAL_SEC = 60
ROLLING_WINDOW = 100
PSI_ALERT_THRESHOLD = 0.25

with open(REF_PATH, "r") as f:
    ref = json.load(f)
BIN_EDGES = ref["bin_edges"]
EXPECTED_PCT = np.array(ref["bin_percentages"])

confidence_history = deque(maxlen=ROLLING_WINDOW)

def calculate_psi(expected_pct, actual_counts, total):
    actual_pct = actual_counts / total
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    probs = payload.get("probabilities")
    if probs:
        confidence_history.append(max(probs))

def main():
    print("[MONITOR] Drift analysis initialized. Subscribing to inference stream...")
    client = mqtt.Client(client_id=f"Drift_Monitor_{TRUCK_ID}")
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_SUB_INF)
    client.loop_start()

    last_check = time.time()
    try:
        while True:
            now = time.time()
            if now - last_check >= CHECK_INTERVAL_SEC:
                if len(confidence_history) >= 10:  # need a minimum sample size
                    actual_counts, _ = np.histogram(list(confidence_history), bins=BIN_EDGES)
                    psi = calculate_psi(EXPECTED_PCT, actual_counts, len(confidence_history))
                    print(f"[MLOPS DRIFT MONITOR] PSI={psi:.3f} | samples={len(confidence_history)}")
                    if psi > PSI_ALERT_THRESHOLD:
                        print(f"[LOGIBRIDGE DRIFT ALERT] PSI={psi:.3f}")
                else:
                    print("[MLOPS DRIFT MONITOR] Waiting for more inference samples...")
                last_check = now
            time.sleep(1)
    except KeyboardInterrupt:
        client.loop_stop()

if __name__ == "__main__":
    main()