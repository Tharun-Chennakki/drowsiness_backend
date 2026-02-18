import cv2
from ultralytics import YOLO
import time
import numpy as np
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import os


# --- SETUP FLASK APP ---
app = Flask(__name__)
CORS(app)

# --- LOAD YOUR TRAINED YOLO MODEL ---
try:
    model = YOLO(os.path.join(os.path.dirname(__file__), "best.pt"))
    print("✅ Model 'best.pt' loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- SETTINGS (Updated as per your request) ---
DROWSY_TIME_THRESHOLD = 1.5  # Seconds to confirm drowsy state
ALARM_INTERVAL = 3.0       # Seconds between alarm triggers

# --- STATE VARIABLES ---
drowsiness_state = {
    'drowsy_start_time': None,
    'is_drowsy_confirmed': False,
    'last_alarm_time': 0
}

# --- THE MAIN DETECTION API ENDPOINT ---
@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    # Step 1: Receive and prepare the image from the React frontend
    img_data = base64.b64decode(data['image'])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    now = time.time()
    main_box = None
    detected_class = "awake"
    box_confidence = 0.0
    trigger_alarm_now = False # This new flag tells the frontend to play the beeps

    # Step 2: YOLO Detection to find the largest face
    results = model(frame, verbose=False)
    max_area = 0
    if results and len(results[0].boxes) > 0:
        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls[0])].lower()
            if class_name in ['awake', 'drowsy']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                # This logic ensures we only process the largest face detected
                if area > max_area:
                    max_area = area
                    main_box = (x1, y1, x2, y2)
                    detected_class = class_name
                    box_confidence = float(box.conf)

    # -------------------------
    # Step 3: Drowsiness & Alarm Logic (Your exact logic, adapted for the server)
    # -------------------------
    if detected_class == "drowsy":
        if drowsiness_state['drowsy_start_time'] is None:
            drowsiness_state['drowsy_start_time'] = now
        
        elapsed_time = now - drowsiness_state['drowsy_start_time']
        
        if elapsed_time >= DROWSY_TIME_THRESHOLD:
            if not drowsiness_state['is_drowsy_confirmed']:
                 print("😴 Driver is drowsy! Starting alert.")
            drowsiness_state['is_drowsy_confirmed'] = True
            
            if now - drowsiness_state['last_alarm_time'] >= ALARM_INTERVAL:
                trigger_alarm_now = True
                drowsiness_state['last_alarm_time'] = now
    else: 
        if drowsiness_state['is_drowsy_confirmed']:
             print("😃 Driver woke up! Stopping alarm.")
        drowsiness_state['drowsy_start_time'] = None
        drowsiness_state['is_drowsy_confirmed'] = False
        drowsiness_state['last_alarm_time'] = 0

    # -------------------------
    # Step 4: Send the results back to the React Frontend
    # -------------------------
    detections_for_frontend = []
    if main_box:
        state_name = "Drowsy" if drowsiness_state['is_drowsy_confirmed'] else "Awake"
        detections_for_frontend.append({
            'box': main_box,
            'name': state_name,
            'conf': box_confidence
        })

    return jsonify({
        'detections': detections_for_frontend,
        'is_drowsy': drowsiness_state['is_drowsy_confirmed'], 
        'trigger_alarm': trigger_alarm_now
    })

# --- START THE SERVER ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)


