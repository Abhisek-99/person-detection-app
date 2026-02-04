from flask import Flask, Response, render_template, request, jsonify
import cv2
from ultralytics import YOLO
import pygame
import time
import os
from datetime import datetime

app = Flask(__name__)

model = YOLO("yolov8n.pt")

# Initialize pygame mixer for audio
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("alert.mp3")
    print("✅ Alert sound loaded successfully")
except:
    print("⚠️ Warning: alert.mp3 not found. Please add it to your project folder.")
    alert_sound = None

# Camera storage
active_cameras = {}
last_alert_time = {}
last_save_time = {}

SAVE_DIR = "detected_persons"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def gen_frames(camera_id):
    global last_alert_time, last_save_time
    
    if camera_id not in last_alert_time:
        last_alert_time[camera_id] = 0
        last_save_time[camera_id] = 0

    cap = active_cameras.get(camera_id)
    if not cap:
        return

    while True:
        success, frame = cap.read()
        if not success:
            print(f"Failed to grab frame from {camera_id}")
            time.sleep(0.1)
            continue

        results = model(frame, conf=0.6)
        person_detected = False

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    person_detected = True
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cv2.putText(
                        frame,
                        f"Person Detected! [{camera_id}]",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        2
                    )

        current_time = time.time()
        
        # Play MP3 alert sound
        if person_detected and current_time - last_alert_time[camera_id] > 3:
            if alert_sound:
                alert_sound.play()
            last_alert_time[camera_id] = current_time

        # Save image
        if person_detected and current_time - last_save_time[camera_id] > 5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/{camera_id}_person_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Image saved: {filename}")
            last_save_time[camera_id] = current_time

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video/<camera_id>')
def video(camera_id):
    return Response(
        gen_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.json
    camera_id = data.get('camera_id')
    camera_url = data.get('camera_url')
    
    if camera_id and camera_url:
        try:
            try:
                camera_url = int(camera_url)
            except ValueError:
                pass
            
            cap = cv2.VideoCapture(camera_url)
            if cap.isOpened():
                active_cameras[camera_id] = cap
                print(f"✅ Camera '{camera_id}' added successfully")
                return jsonify({'success': True, 'message': f'Camera {camera_id} added'})
            else:
                return jsonify({'success': False, 'message': 'Failed to open camera'})
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})
    
    return jsonify({'success': False, 'message': 'Invalid data'})


@app.route('/remove_camera/<camera_id>', methods=['POST'])
def remove_camera(camera_id):
    if camera_id in active_cameras:
        active_cameras[camera_id].release()
        del active_cameras[camera_id]
        print(f"❌ Camera '{camera_id}' removed")
        return jsonify({'success': True, 'message': f'Camera {camera_id} removed'})
    return jsonify({'success': False, 'message': 'Camera not found'})


@app.route('/list_cameras')
def list_cameras():
    cameras = [{'id': cam_id, 'active': True} for cam_id in active_cameras.keys()]
    return jsonify(cameras)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("📌 IMPORTANT: Start ngrok in a SEPARATE terminal:")
    print("   Run: ngrok http 5000")
    print("   Then copy the public URL and use it to access this app")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        active_cameras['laptop'] = cap
        print("✅ Laptop camera initialized")
    
    app.run(debug=False, host='0.0.0.0', port=5000)