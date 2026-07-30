import os
import json
import time
import numpy as np
import paho.mqtt.client as mqtt
from collections import deque
from scipy.stats import kurtosis

# 1. Environment and Path Configurations
MODEL_PATH = os.getenv("MODEL_PATH", "inference/model.tflite")
TRUCK_ID = os.getenv("TRUCK_ID", "01")

MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC_SUB_TEMP = f"logibridge/trucks/{TRUCK_ID}/sensors/temperature"
TOPIC_SUB_VIBE = f"logibridge/trucks/{TRUCK_ID}/sensors/vibration"
TOPIC_PUB_INF  = f"logibridge/trucks/{TRUCK_ID}/inference"

# 2. Window Buffers
temp_buffer = deque(maxlen=30)
vibe_buffer = deque(maxlen=15)
temp_filter_buf = deque(maxlen=5)
vibe_filter_buf = deque(maxlen=5)

# Load normalization arrays
STATS_PATH = "data_pipeline/training_stats.npy"
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError("Missing calibration parameters! Run generate_dataset.py first.")
stats = np.load(STATS_PATH, allow_pickle=True).item()

# 3. Initialize TFLite Interpreter
print(f"[EDGE SERVICE] Initializing TFLite Engine utilizing asset: {MODEL_PATH}")
try:
    import tensorflow.lite as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
except ImportError:
    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def moving_average(buf, val):
    buf.append(val)
    return sum(buf) / len(buf)

"""def extract_features():
    if len(temp_buffer) < 30 or len(vibe_buffer) < 15:
        return None
    t_data = np.array(temp_buffer)
    v_data = np.array(vibe_buffer)
    
    t_mean = np.mean(t_data)
    t_std = np.std(t_data)
    # Ensure this matches dataset generator formula exactly
    t_rate = ((t_data[-1] - t_data[0]) / 30.0) * 60.0
    
    v_rms = np.sqrt(np.mean(v_data**2))
    v_peak = np.max(np.abs(v_data))
    v_kurt = float(kurtosis(v_data, fisher=True))
    
    raw_vector = np.array([t_mean, t_std, t_rate, v_rms, v_peak, v_kurt])

    # Apply Z-Score Normalization
    return (raw_vector - stats["mean"]) / (stats["std"] + 1e-8)"""
def extract_features():
    if len(temp_buffer) < 30 or len(vibe_buffer) < 15:
        return None

    t_data = np.array(temp_buffer)
    v_data = np.array(vibe_buffer)
    
    t_mean = np.mean(t_data)
    t_std = np.std(t_data)
    t_rate = ((t_data[-1] - t_data[0]) / 30.0) * 60.0

    v_rms = np.sqrt(np.mean(v_data**2))
    v_peak = np.max(np.abs(v_data))
    v_kurt = float(kurtosis(v_data, fisher=True))

    raw_vector = np.array([
        t_mean,
        t_std,
        t_rate,
        v_rms,
        v_peak,
        v_kurt
    ])

    normalized = (raw_vector - stats["mean"]) / (stats["std"] + 1e-8)

    return normalized

def execute_inference(scaled_features):
    input_scale, input_zero_point = input_details[0]['quantization']
    
    # 1. Input Tensor Preparation
    if input_scale == 0 or input_scale is None:
        # Float32 model
        input_data = np.expand_dims(scaled_features, axis=0).astype(np.float32)
    else:
        # INT8 Quantized model
        quantized_input = (scaled_features / input_scale) + input_zero_point
        quantized_input = np.clip(quantized_input, -128, 127).astype(np.int8)
        input_data = np.expand_dims(quantized_input, axis=0)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 2. Output Tensor Extraction
    raw_output = interpreter.get_tensor(output_details[0]['index'])[0]
    out_scale, out_zero_point = output_details[0]['quantization']

    # 3. Output Scale Decoding
    if out_scale == 0 or out_scale is None:
        # Model output is already float probabilities (Softmax)
        probs = raw_output.astype(np.float32)
    else:
        # Model output is INT8 quantized
        probs = (raw_output.astype(np.float32) - out_zero_point) * out_scale
        
    # Ensure probabilities sum to 1 if dequantization introduced slight noise
    if np.sum(probs) > 0:
        probs = probs / np.sum(probs)
        
    return probs

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    val = payload.get("value")
    if val is None: return
    
    if msg.topic == TOPIC_SUB_TEMP:
        # print(f"RAW MQTT TEMP: {val}")
        filtered_val = moving_average(temp_filter_buf, val)
        # print(f"FILTERED TEMP: {filtered_val}")
        temp_buffer.append(moving_average(temp_filter_buf, val))
    elif msg.topic == TOPIC_SUB_VIBE:
        vibe_buffer.append(moving_average(vibe_filter_buf, val))

def main():
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2, 
        client_id=f"Inference_Service_{TRUCK_ID}"
    )
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_SUB_TEMP)
    client.subscribe(TOPIC_SUB_VIBE)
    client.loop_start()
    
    last_window_time = time.time()
    
    try:
        while True:
            curr_time = time.time()
            if curr_time - last_window_time >= 10:
                features = extract_features()
                if features is not None:
                    print(f"\n[DEBUG DATA IN] Fused & Normalized Vector: {np.round(features, 4)}")
                    
                    probs = execute_inference(features)
                    predicted_class = int(np.argmax(probs))
                    
                    output_payload = {
                        "timestamp": time.time(),
                        "class": predicted_class,
                        "confidence": float(probs[predicted_class]),
                        "probabilities": probs.tolist()
                    }
                    client.publish(TOPIC_PUB_INF, json.dumps(output_payload), qos=1)
                    print(f"[INFERENCE] Classified State: {predicted_class} | Conf: {probs[predicted_class]:.3f}")
                
                last_window_time = curr_time
            time.sleep(0.2)
    except KeyboardInterrupt:
        client.loop_stop()

if __name__ == "__main__":
    main()