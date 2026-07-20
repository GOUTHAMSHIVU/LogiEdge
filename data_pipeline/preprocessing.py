import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from scipy.stats import kurtosis

# Buffer management for sliding windows (30s window)[cite: 1]
temp_buffer = deque(maxlen=30)  # 30 seconds @ 1Hz
vibe_buffer = deque(maxlen=15)  # 30 seconds @ 0.5Hz

# Raw filtering buffers (5-sample moving averages)[cite: 1]
temp_filter_buf = deque(maxlen=5)
vibe_filter_buf = deque(maxlen=5)

def moving_average(buffer, new_value):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)

def extract_features():
    if len(temp_buffer) < 30 or len(vibe_buffer) < 15:
        return None  # Buffers must be completely full[cite: 1]
        
    t_data = np.array(temp_buffer)
    v_data = np.array(vibe_buffer)
    
    # Feature extraction formulas[cite: 1]
    t_mean = np.mean(t_data)
    t_std = np.std(t_data)
    t_rate = ((t_data[-1] - t_data[0]) / 30.0) * 60.0  # Normalized to °C/min[cite: 1]
    
    v_rms = np.sqrt(np.mean(v_data**2))
    v_peak = np.max(np.abs(v_data))
    v_kurt = kurtosis(v_data)
    
    # Joint 6-value feature-level fusion vector[cite: 1]
    return np.array([t_mean, t_std, t_rate, v_rms, v_peak, v_kurt])

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        if "value" not in payload:
            return
        val = payload["value"]
        
        if msg.topic == "logibridge/trucks/01/sensors/temperature":
            filtered_val = moving_average(temp_filter_buf, val)
            temp_buffer.append(filtered_val)
            
        elif msg.topic == "logibridge/trucks/01/sensors/vibration":
            filtered_val = moving_average(vibe_filter_buf, val)
            vibe_buffer.append(filtered_val)
    except Exception as e:
        print(f"[ERROR] Message parsing failed: {e}")

def main():
    client = mqtt.Client(client_id="Preprocessing_Engine")
    client.on_message = on_message
    client.connect("localhost", 1883, 60)
    
    client.subscribe("logibridge/trucks/01/sensors/temperature")
    client.subscribe("logibridge/trucks/01/sensors/vibration")
    
    client.loop_start()
    
    # Sliding window execution config: step = 10s[cite: 1]
    last_window_time = time.time()
    print("[PREPROCESSING] Ingestion pipeline armed. Awaiting window population...")
    
    try:
        while True:
            current_time = time.time()
            if current_time - last_window_time >= 10:
                features = extract_features()
                if features is not None:
                    print(f"[WINDOW EXTRACTED] Features: {np.round(features, 3)}")
                last_window_time = current_time
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("[PREPROCESSING] Halting engine...")
        client.loop_stop()

if __name__ == "__main__":
    main()