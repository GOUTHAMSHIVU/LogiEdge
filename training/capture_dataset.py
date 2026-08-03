import os
import json
import time
import argparse
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from scipy.stats import kurtosis

TRUCK_ID = "01"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPIC_TEMP = f"LogiEdge/trucks/{TRUCK_ID}/sensors/temperature"
TOPIC_VIBE = f"LogiEdge/trucks/{TRUCK_ID}/sensors/vibration"

temp_buffer = deque(maxlen=30)
vibe_buffer = deque(maxlen=15)
temp_filter_buf = deque(maxlen=5)
vibe_filter_buf = deque(maxlen=5)

captured_windows = []

def moving_average(buf, val):
    buf.append(val)
    return sum(buf) / len(buf)

def extract_features():
    if len(temp_buffer) < 30 or len(vibe_buffer) < 15:
        return None
    t_data = np.array(temp_buffer)
    v_data = np.array(vibe_buffer)
    t_mean = np.mean(t_data)
    t_std = np.std(t_data)
    t_rate = ((t_data[-1] - t_data[0]) / 30.0) * 60.0
    v_rms = np.sqrt(np.mean(v_data ** 2))
    v_peak = np.max(np.abs(v_data))
    v_kurt = kurtosis(v_data)
    return np.array([t_mean, t_std, t_rate, v_rms, v_peak, v_kurt])

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    val = payload.get("value")
    if val is None:
        return
    if msg.topic == TOPIC_TEMP:
        temp_buffer.append(moving_average(temp_filter_buf, val))
    elif msg.topic == TOPIC_VIBE:
        vibe_buffer.append(moving_average(vibe_filter_buf, val))

def main():
    parser = argparse.ArgumentParser(description="Capture live windowed features for dataset labeling")
    parser.add_argument("--anomaly", required=True, choices=["none", "temp_drift", "vibration", "combined"])
    parser.add_argument("--duration", type=int, required=True, help="Capture duration in seconds")
    parser.add_argument("--step", type=int, default=10, help="Window step in seconds (default 10)")
    args = parser.parse_args()

    client = mqtt.Client(client_id="Dataset_Capture")
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_TEMP)
    client.subscribe(TOPIC_VIBE)
    client.loop_start()

    print(f"[CAPTURE] Starting capture for mode='{args.anomaly}', duration={args.duration}s")
    print(f"[CAPTURE] Make sure simulator.py --anomaly {args.anomaly} is running in another terminal.")

    start = time.time()
    last_window = start
    try:
        while time.time() - start < args.duration:
            now = time.time()
            if now - last_window >= args.step:
                feats = extract_features()
                if feats is not None:
                    captured_windows.append(feats)
                    print(f"[CAPTURE] Window {len(captured_windows)}: {np.round(feats, 3)}")
                last_window = now
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()

    out_path = f"training/windows_{args.anomaly}.npy"
    os.makedirs("training", exist_ok=True)
    np.save(out_path, np.array(captured_windows))
    print(f"[CAPTURE] Done. Saved {len(captured_windows)} windows to {out_path}")

if __name__ == "__main__":
    main()