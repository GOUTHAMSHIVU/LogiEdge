import argparse
import time
import json
import random
import paho.mqtt.client as mqtt

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

TOPIC_TEMP = "logibridge/trucks/01/sensors/temperature"
TOPIC_VIBE = "logibridge/trucks/01/sensors/vibration"
TOPIC_DOOR = "logibridge/trucks/01/sensors/door"

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
    client = mqtt.Client(client_id="Truck_01_Sim")
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    
    print(f"[SIMULATOR] Started in mode: {args.anomaly}")
    
    base_temp = 4.0
    step_count = 0
    
    try:
        while True:
            # 1. Temperature Stream (1 Hz)
            if args.anomaly in ["temp_drift", "combined"]:
                base_temp += 0.08  # Linear drift: +0.08°C per reading
                temp = random.normalvariate(base_temp, 0.3)
            else:
                temp = random.normalvariate(4.0, 0.3)
                
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
        print("[SIMULATOR] Terminating stream...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()