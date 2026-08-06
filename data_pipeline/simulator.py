import argparse
import time
import json
import random
import paho.mqtt.client as mqtt
import os

MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-broker")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

TOPIC_TEMP = "LogiEdge/trucks/01/sensors/temperature"
TOPIC_VIBE = "LogiEdge/trucks/01/sensors/vibration"
TOPIC_DOOR = "LogiEdge/trucks/01/sensors/door"

def parse_arguments():
    parser = argparse.ArgumentParser(description="LogiEdge Cold-Chain Sensor Simulator")
    parser.add_argument(
        "--anomaly", 
        choices=["none", "temp_drift", "vibration", "combined"], 
        default="none",
        help="Inject specific operational anomalies into the data stream"
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="Truck_01_Sim")
    
    print(f"Connecting to MQTT Broker at {MQTT_BROKER}:{MQTT_PORT}...", flush=True)
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()  #  Always start the loop on successful connection
        print("Connected successfully!", flush=True)
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}", flush=True)
        return

    print(f"[SIMULATOR] Started in mode: {args.anomaly}", flush=True)
    
    base_temp = 4.0
    step_count = 0
    
    try:
        while True:
            # 1. Temperature Stream (1 Hz)
            if args.anomaly in ["temp_drift", "combined"]:
                # Cap base_temp so normalized feature values stay within model bounds
                MAX_TEMP_CEILING = 31.0  # Max realistic anomaly baseline (e.g., 25.0 baseline + 6.0°C drift)
                base_temp = min(base_temp + 0.08, MAX_TEMP_CEILING)
                temp = random.normalvariate(base_temp, 0.3)
            else:
                temp = random.normalvariate(4.0, 0.3)
            print(f"[{args.anomaly}] Publishing Temp = {temp:.3f}")
            temp_payload = {"timestamp": time.time(), "value": round(temp, 3)}
            client.publish(TOPIC_TEMP, json.dumps(temp_payload), qos=1)
            
            # 2. Vibration Stream (0.5 Hz -> Send every 2 seconds)
            if step_count % 2 == 0:
                if args.anomaly in ["vibration", "combined"]:
                    vibe_rms = random.normalvariate(1.2, 0.15)  # Bearing wear: ~1.2g
                else:
                    vibe_rms = random.normalvariate(0.45, 0.05)
                    
                vibe_payload = {"timestamp": time.time(), "value": round(vibe_rms, 4)}
                client.publish(TOPIC_VIBE, json.dumps(vibe_payload), qos=0)
            
            # 3. Random Discrete Door Events
            if random.random() < 0.02:
                door_state = random.choice(["OPEN", "CLOSE"])
                door_payload = {"timestamp": time.time(), "event": door_state}
                client.publish(TOPIC_DOOR, json.dumps(door_payload), qos=1)
                
            step_count += 1
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("[SIMULATOR] Terminating stream...", flush=True)
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()