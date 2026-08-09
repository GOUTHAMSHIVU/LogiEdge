# monitoring/drift_monitor.py
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
import os

TRUCK_ID = os.getenv("TRUCK_ID", "01")
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

TOPIC_SUB_INF = f"LogiEdge/trucks/{TRUCK_ID}/inference"
REF_PATH = "monitoring/reference_dist.json"

CHECK_INTERVAL_SEC = 60
MIN_SAMPLES = 100  # Analyzes drift when at least 100 samples arrive

with open(REF_PATH, "r") as f:
    ref = json.load(f)
BIN_EDGES = ref["bin_edges"]
EXPECTED_PCT = np.array(ref["bin_percentages"])

confidence_history = deque(maxlen=MIN_SAMPLES)
last_check_time = time.time()

def calculate_psi(expected_pct, actual_counts, total):
    actual_pct = actual_counts / total
    expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"[MONITOR] Connected to MQTT Broker (rc={rc}). Subscribing to {TOPIC_SUB_INF}...", flush=True)
    client.subscribe(TOPIC_SUB_INF)

def on_message(client, userdata, msg):
    global last_check_time
    try:
        payload = json.loads(msg.payload.decode())
        probs = payload.get("probabilities")
        conf = payload.get("confidence")
        cls_id = payload.get("class")
        
        # Explicit print of probabilities and class on every single packet
        print(f"[RAW INFERENCE RECV] Class: {cls_id} | Conf: {conf} | Probs: {probs}", flush=True)
        
        val = max(probs) if probs else (float(conf) if conf is not None else None)
        if val is not None:
            confidence_history.append(val)
            print(f"[MONITOR] Stored confidence sample ({val:.4f}) | Buffer: {len(confidence_history)}/{MIN_SAMPLES}", flush=True)

        # Evaluate drift inline whenever an inference packet arrives
        now = time.time()
        if now - last_check_time >= CHECK_INTERVAL_SEC:
            if len(confidence_history) >= MIN_SAMPLES:
                actual_counts, _ = np.histogram(list(confidence_history), bins=BIN_EDGES)
                psi = calculate_psi(EXPECTED_PCT, actual_counts, len(confidence_history))
                print(f"\n--- [MLOPS DRIFT EVALUATION] ---", flush=True)
                print(f"PSI Score: {psi:.4f} | Evaluated Samples: {len(confidence_history)}", flush=True)
                print(f"Confidence Queue: {list(confidence_history)}", flush=True)
                if psi > 0.25:
                    print(f"[LogiEdge DRIFT ALERT] Distribution Shift Detected! (PSI={psi:.4f} > 0.25)", flush=True)
                print("--------------------------------\n", flush=True)
            else:
                print(f"[MONITOR STATUS] Accumulating samples... ({len(confidence_history)}/{MIN_SAMPLES})", flush=True)
            last_check_time = now

    except Exception as e:
        print(f"[ERROR] Failed parsing payload: {e}", flush=True)

def main():
    print(f"[MONITOR] Starting listener targeting {MQTT_BROKER}:{MQTT_PORT}...", flush=True)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"Drift_Monitor_{TRUCK_ID}")
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception:
        client.connect("localhost", MQTT_PORT, 60)
        
    # loop_forever handles socket I/O synchronously on the main thread
    client.loop_forever()

if __name__ == "__main__":
    main()