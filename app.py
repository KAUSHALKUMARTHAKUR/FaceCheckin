from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import time
import uuid
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import cv2
from deepface import DeepFace
import mediapipe as mp
from typing import Optional, Dict, Tuple, Any
import tempfile
import atexit
import shutil

# --- Evaluation Metrics Counters (legacy, kept for compatibility display) ---
total_attempts = 0
correct_recognitions = 0
false_accepts = 0
false_rejects = 0
unauthorized_attempts = 0
inference_times = []

# ---------------------------------------------------
# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Create temporary directory for image processing
TEMP_DIR = tempfile.mkdtemp()

def cleanup_temp_dir():
    """Clean up temporary directory on exit"""
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"Error cleaning up temp directory: {e}")

# Register cleanup function
atexit.register(cleanup_temp_dir)

# MongoDB Connection
try:
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    client = MongoClient(mongo_uri)
    db = client['face_attendance_system']
    students_collection = db['students']
    teachers_collection = db['teachers']
    attendance_collection = db['attendance']
    metrics_events = db['metrics_events']

    # Indexes
    students_collection.create_index([("student_id", pymongo.ASCENDING)], unique=True)
    teachers_collection.create_index([("teacher_id", pymongo.ASCENDING)], unique=True)
    attendance_collection.create_index([
        ("student_id", pymongo.ASCENDING),
        ("date", pymongo.ASCENDING),
        ("subject", pymongo.ASCENDING)
    ])
    metrics_events.create_index([("ts", pymongo.DESCENDING)])
    metrics_events.create_index([("event", pymongo.ASCENDING)])
    metrics_events.create_index([("attempt_type", pymongo.ASCENDING)])
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection error: {e}")

# ---------------- Lightweight Model Implementations ----------------

# Initialize YuNet Face Detector with proper model download and fallback
face_detector = None
face_detector_type = None  # Track detector type for proper usage

try:
    model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_path):
        import urllib.request
        url = "https://github.com/opencv/opencv_zoo/raw/master/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        print(f"Downloading YuNet model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print("Downloaded YuNet model successfully")

    face_detector = cv2.FaceDetectorYN.create(
        model=model_path,
        config="",
        input_size=(640, 480)
    )
    face_detector_type = "yunet"
    print("YuNet face detector initialized successfully")
except Exception as e:
    print(f"Error initializing YuNet: {e}")
    # Fallback to Haar cascade
    try:
        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_detector_type = "haar"
        print("Fallback to Haar cascade face detector")
    except Exception as e2:
        print(f"Error initializing Haar cascade: {e2}")

# Initialize MediaPipe Face Mesh
face_mesh = None
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
    print("MediaPipe Face Mesh initialized successfully")
except Exception as e:
    print(f"Error initializing MediaPipe: {e}")

# Initialize Haar Cascade for simple liveness detection
eye_cascade = None
try:
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    print("Eye cascade classifier initialized successfully")
except Exception as e:
    print(f"Error initializing eye cascade: {e}")

def get_unique_temp_path(prefix="temp", suffix=".jpg"):
    """Generate unique temporary file path"""
    unique_id = str(uuid.uuid4())
    filename = f"{prefix}_{unique_id}_{int(time.time())}{suffix}"
    return os.path.join(TEMP_DIR, filename)

def detect_faces_yunet(image):
    """Detect faces using YuNet or fallback to Haar cascade"""
    if face_detector is None:
        return []

    try:
        height, width = image.shape[:2]

        # Use YuNet detector
        if face_detector_type == "yunet":
            face_detector.setInputSize((width, height))
            _, faces = face_detector.detect(image)
            if faces is None:
                return []

            detections = []
            for face in faces:
                x1, y1, w, h = face[:4].astype(int)
                x2, y2 = x1 + w, y1 + h
                confidence = face[14] if len(face) > 14 else 0.9
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": float(confidence)
                })
            return detections

        # Use Haar cascade detector
        elif face_detector_type == "haar":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": 0.9  # Default confidence for Haar
                })
            return detections

        else:
            return []

    except Exception as e:
        print(f"Error in face detection: {e}")
        # Final fallback to simple Haar cascade
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    "bbox": [x, y, x + w, y + h],
                    "score": 0.9
                })
            return detections
        except Exception as e2:
            print(f"Error in fallback Haar cascade detection: {e2}")
            return []

def recognize_face_deepface(image, user_id, user_type='student'):
    """Face recognition using DeepFace (lightweight alternative) - optimized for Render"""
    global total_attempts, correct_recognitions, unauthorized_attempts, inference_times
    
    temp_files = []  # Track temp files for cleanup
    
    try:
        start_time = time.time()
        
        # Save current image temporarily with unique filename
        temp_img_path = get_unique_temp_path(f"current_{user_id}")
        temp_files.append(temp_img_path)
        cv2.imwrite(temp_img_path, image)
        
        # Get user's reference image
        if user_type == 'student':
            user = students_collection.find_one({'student_id': user_id})
        else:
            user = teachers_collection.find_one({'teacher_id': user_id})
        
        if not user or 'face_image' not in user:
            unauthorized_attempts += 1
            return False, f"No reference face found for {user_type} ID {user_id}"
        
        # Save reference image temporarily with unique filename
        ref_image_bytes = user['face_image']
        ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
        ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
        temp_ref_path = get_unique_temp_path(f"ref_{user_id}")
        temp_files.append(temp_ref_path)
        cv2.imwrite(temp_ref_path, ref_image)
        
        try:
            # Use DeepFace for verification
            result = DeepFace.verify(
                img1_path=temp_img_path,
                img2_path=temp_ref_path,
                model_name="Facenet512",  # Lightweight model
                enforce_detection=False
            )
            
            is_verified = result["verified"]
            distance = result["distance"]
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            total_attempts += 1
            
            if is_verified:
                correct_recognitions += 1
                return True, f"Face recognized (distance={distance:.3f}, time={inference_time:.2f}s)"
            else:
                unauthorized_attempts += 1
                return False, f"Unauthorized attempt detected (distance={distance:.3f})"
                
        except Exception as e:
            return False, f"DeepFace verification error: {str(e)}"
        
    except Exception as e:
        return False, f"Error in face recognition: {str(e)}"
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"Error cleaning up temp file {temp_file}: {e}")

def get_face_landmarks_mediapipe(image):
    """Get face landmarks using MediaPipe"""
    if face_mesh is None:
        return None
    
    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None
    except Exception as e:
        print(f"Error in MediaPipe landmarks: {e}")
        return None

def simple_liveness_check(image):
    """Simple liveness detection using eye detection"""
    if eye_cascade is None:
        return 0.5  # Default moderate score if cascade not available
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Simple liveness scoring based on eye detection
        if len(eyes) >= 2:
            return 0.8  # High confidence if both eyes detected
        elif len(eyes) == 1:
            return 0.6  # Medium confidence if one eye detected
        else:
            return 0.3  # Low confidence if no eyes detected
    except Exception as e:
        print(f"Error in liveness check: {e}")
        return 0.5

def expand_and_clip_box(bbox_xyxy, scale: float, w: int, h: int):
    x1, y1, x2, y2 = bbox_xyxy
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    bw2 = bw * scale
    bh2 = bh * scale
    x1n = int(max(0, cx - bw2 / 2.0))
    y1n = int(max(0, cy - bh2 / 2.0))
    x2n = int(min(w - 1, cx + bw2 / 2.0))
    y2n = int(min(h - 1, cy + bh2 / 2.0))
    return x1n, y1n, x2n, y2n

def draw_live_overlay(img_bgr: np.ndarray, bbox, label: str, prob: float, color):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {prob:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y_top = max(0, y1 - th - 8)
    cv2.rectangle(img_bgr, (x1, y_top), (x1 + tw + 6, y_top + th + 6), color, -1)
    cv2.putText(img_bgr, text, (x1 + 3, y_top + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

def image_to_data_uri(img_bgr: np.ndarray) -> Optional[str]:
    success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def decode_image(base64_image):
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]
    image_bytes = base64.b64decode(base64_image)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

# Legacy function for backward compatibility
def get_face_features(image):
    """Legacy wrapper - now uses DeepFace internally"""
    return None  # DeepFace handles feature extraction internally

def recognize_face(image, user_id, user_type='student'):
    """Legacy wrapper for the new DeepFace recognition"""
    return recognize_face_deepface(image, user_id, user_type)

# ---------------------- Metrics helpers ----------------------
def log_metrics_event(event: dict):
    try:
        metrics_events.insert_one(event)
    except Exception as e:
        print("Failed to log metrics event:", e)

def log_metrics_event_normalized(
    *,
    event: str,
    attempt_type: str,
    claimed_id: Optional[str],
    recognized_id: Optional[str],
    liveness_pass: bool,
    distance: Optional[float],
    live_prob: Optional[float],
    latency_ms: Optional[float],
    client_ip: Optional[str],
    reason: Optional[str] = None
):
    if not liveness_pass:
        decision = "spoof_blocked"
    else:
        decision = "recognized" if event.startswith("accept") else "not_recognized"

    doc = {
        "ts": datetime.now(timezone.utc),
        "event": event,
        "attempt_type": attempt_type,
        "claimed_id": claimed_id,
        "recognized_id": recognized_id,
        "liveness_pass": bool(liveness_pass),
        "distance": distance,
        "live_prob": live_prob,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
        "reason": reason,
        "decision": decision,
    }
    log_metrics_event(doc)

def classify_event(ev: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Returns (event, attempt_type), robust to legacy documents."""
    if ev.get("event"):
        e = ev.get("event")
        at = ev.get("attempt_type")
        if not at:
            if e in ("accept_true", "reject_false"):
                at = "genuine"
            elif e in ("accept_false", "reject_true"):
                at = "impostor"
        return e, at

    decision = ev.get("decision")
    success = ev.get("success")
    reason = (ev.get("reason") or "") if isinstance(ev.get("reason"), str) else ev.get("reason")

    if decision == "recognized" and (success is True or success is None):
        return "accept_true", "genuine"

    if decision == "spoof_blocked":
        return "reject_true", "impostor"

    if decision == "not_recognized":
        if reason in ("false_reject",):
            return "reject_false", "genuine"
        if reason in ("unauthorized_attempt", "liveness_fail", "mismatch_claim", "no_face_detected", "failed_crop", "recognition_error"):
            return "reject_true", "impostor"
        return "reject_true", "impostor"

    return None, None

def compute_metrics(limit: int = 10000):
    """Robust metrics aggregation that tolerates legacy docs."""
    try:
        cursor = metrics_events.find({}, {"_id": 0}).sort("ts", -1).limit(limit)
        counts = {
            "trueAccepts": 0,
            "falseAccepts": 0,
            "trueRejects": 0,
            "falseRejects": 0,
            "genuineAttempts": 0,
            "impostorAttempts": 0,
            "unauthorizedRejected": 0,
            "unauthorizedAccepted": 0,
        }

        total_attempts_calc = 0

        for ev in cursor:
            e, at = classify_event(ev)
            if not e:
                continue
            total_attempts_calc += 1

            if e == "accept_true":
                counts["trueAccepts"] += 1
            elif e == "accept_false":
                counts["falseAccepts"] += 1
                counts["unauthorizedAccepted"] += 1
            elif e == "reject_true":
                counts["trueRejects"] += 1
                counts["unauthorizedRejected"] += 1
            elif e == "reject_false":
                counts["falseRejects"] += 1

            if at == "genuine":
                counts["genuineAttempts"] += 1
            elif at == "impostor":
                counts["impostorAttempts"] += 1

        genuine_attempts = max(counts["genuineAttempts"], 1)
        impostor_attempts = max(counts["impostorAttempts"], 1)
        total_attempts_final = max(total_attempts_calc, 1)

        FAR = counts["falseAccepts"] / impostor_attempts
        FRR = counts["falseRejects"] / genuine_attempts
        accuracy = (counts["trueAccepts"] + counts["trueRejects"]) / total_attempts_final

        return {
            "counts": counts,
            "rates": {
                "FAR": FAR,
                "FRR": FRR,
                "accuracy": accuracy
            },
            "totals": {
                "totalAttempts": total_attempts_calc
            }
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            "counts": {"trueAccepts": 0, "falseAccepts": 0, "trueRejects": 0, "falseRejects": 0,
                      "genuineAttempts": 0, "impostorAttempts": 0, "unauthorizedRejected": 0, "unauthorizedAccepted": 0},
            "rates": {"FAR": 0, "FRR": 0, "accuracy": 0},
            "totals": {"totalAttempts": 0}
        }

def compute_latency_avg(limit: int = 300) -> Optional[float]:
    try:
        cursor = metrics_events.find({"latency_ms": {"$exists": True}}, {"latency_ms": 1, "_id": 0}).sort("ts", -1).limit(limit)
        vals = [float(d["latency_ms"]) for d in cursor if isinstance(d.get("latency_ms"), (int, float))]
        if not vals:
            return None
        return sum(vals) / len(vals)
    except Exception as e:
        print(f"Error computing latency average: {e}")
        return None

# --------- STUDENT ROUTES ---------
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/register.html')
def register_page():
    return render_template('register.html')

@app.route('/metrics')
def metrics_dashboard():
    return render_template('metrics.html')

@app.route('/register', methods=['POST'])
def register():
    try:
        student_data = {
            'student_id': request.form.get('student_id'),
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'department': request.form.get('department'),
            'course': request.form.get('course'),
            'year': request.form.get('year'),
            'division': request.form.get('division'),
            'mobile': request.form.get('mobile'),
            'dob': request.form.get('dob'),
            'gender': request.form.get('gender'),
            'password': request.form.get('password'),
            'created_at': datetime.now()
        }
        face_image = request.form.get('face_image')
        if face_image and ',' in face_image:
            image_data = face_image.split(',')[1]
            student_data['face_image'] = Binary(base64.b64decode(image_data))
            student_data['face_image_type'] = face_image.split(',')[0].split(':')[1].split(';')[0]
        else:
            flash('Face image is required for registration.', 'danger')
            return redirect(url_for('register_page'))

        result = students_collection.insert_one(student_data)
        if result.inserted_id:
            flash('Registration successful! You can now login.', 'success')
            return redirect(url_for('login_page'))
        else:
            flash('Registration failed. Please try again.', 'danger')
            return redirect(url_for('register_page'))
    except pymongo.errors.DuplicateKeyError:
        flash('Student ID already exists. Please use a different ID.', 'danger')
        return redirect(url_for('register_page'))
    except Exception as e:
        flash(f'Registration failed: {str(e)}', 'danger')
        return redirect(url_for('register_page'))

@app.route('/login', methods=['POST'])
def login():
    student_id = request.form.get('student_id')
    password = request.form.get('password')
    student = students_collection.find_one({'student_id': student_id})

    if student and student['password'] == password:
        session['logged_in'] = True
        session['user_type'] = 'student'
        session['student_id'] = student_id
        session['name'] = student.get('name')
        flash('Login successful!', 'success')
        return redirect(url_for('dashboard'))
    else:
        flash('Invalid credentials. Please try again.', 'danger')
        return redirect(url_for('login_page'))

@app.route('/face-login', methods=['POST'])
def face_login():
    face_image = request.form.get('face_image')
    face_role = request.form.get('face_role')

    if not face_image or not face_role:
        flash('Face image and role are required for face login.', 'danger')
        return redirect(url_for('login_page'))

    image = decode_image(face_image)

    if face_role == 'student':
        collection = students_collection
        id_field = 'student_id'
        dashboard_route = 'dashboard'
    elif face_role == 'teacher':
        collection = teachers_collection
        id_field = 'teacher_id'
        dashboard_route = 'teacher_dashboard'
    else:
        flash('Invalid role selected for face login.', 'danger')
        return redirect(url_for('login_page'))

    users = collection.find({'face_image': {'$exists': True, '$ne': None}})
    
    # Use DeepFace for face matching with improved temp file handling
    temp_login_path = get_unique_temp_path("login_image")
    cv2.imwrite(temp_login_path, image)
    
    try:
        for user in users:
            ref_image_bytes = user['face_image']
            ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
            ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
            
            temp_ref_path = get_unique_temp_path(f"ref_{user[id_field]}")
            cv2.imwrite(temp_ref_path, ref_image)
            
            try:
                result = DeepFace.verify(
                    img1_path=temp_login_path,
                    img2_path=temp_ref_path,
                    model_name="Facenet512",
                    enforce_detection=False
                )
                
                if result["verified"]:
                    session['logged_in'] = True
                    session['user_type'] = face_role
                    session[id_field] = user[id_field]
                    session['name'] = user.get('name')
                    flash('Face login successful!', 'success')
                    
                    # Cleanup
                    for temp_file in [temp_ref_path, temp_login_path]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    return redirect(url_for(dashboard_route))
                
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
            except Exception as e:
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
                continue
        
        if os.path.exists(temp_login_path):
            os.remove(temp_login_path)
            
    except Exception as e:
        if os.path.exists(temp_login_path):
            os.remove(temp_login_path)

    flash('Face not recognized. Please try again or contact admin.', 'danger')
    return redirect(url_for('login_page'))

@app.route('/auto-face-login', methods=['POST'])
def auto_face_login():
    """Enhanced auto face login with role support"""
    try:
        data = request.json
        face_image = data.get('face_image')
        face_role = data.get('face_role', 'student')
        if not face_image:
            return jsonify({'success': False, 'message': 'No image received'})
        
        image = decode_image(face_image)
        
        if face_role == 'teacher':
            collection = teachers_collection
            id_field = 'teacher_id'
            dashboard_route = '/teacher_dashboard'
        else:
            collection = students_collection
            id_field = 'student_id'
            dashboard_route = '/dashboard'

        # Use DeepFace for recognition with improved temp file handling
        temp_auto_path = get_unique_temp_path("auto_login")
        cv2.imwrite(temp_auto_path, image)
        
        try:
            users = collection.find({'face_image': {'$exists': True, '$ne': None}})
            for user in users:
                try:
                    ref_image_array = np.frombuffer(user['face_image'], np.uint8)
                    ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
                    
                    temp_ref_path = get_unique_temp_path(f"auto_ref_{user[id_field]}")
                    cv2.imwrite(temp_ref_path, ref_image)
                    
                    result = DeepFace.verify(
                        img1_path=temp_auto_path,
                        img2_path=temp_ref_path,
                        model_name="Facenet512",
                        enforce_detection=False
                    )
                    
                    if result["verified"]:
                        session['logged_in'] = True
                        session['user_type'] = face_role
                        session[id_field] = user[id_field]
                        session['name'] = user.get('name')
                        
                        # Cleanup
                        for temp_file in [temp_ref_path, temp_auto_path]:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        
                        return jsonify({
                            'success': True,
                            'message': f'Welcome {user["name"]}! Redirecting...',
                            'redirect_url': dashboard_route,
                            'face_role': face_role
                        })
                    
                    if os.path.exists(temp_ref_path):
                        os.remove(temp_ref_path)
                except Exception as e:
                    continue
            
            if os.path.exists(temp_auto_path):
                os.remove(temp_auto_path)
                
        except Exception as e:
            if os.path.exists(temp_auto_path):
                os.remove(temp_auto_path)

        return jsonify({'success': False, 'message': f'Face not recognized in {face_role} database'})
    except Exception as e:
        print(f"Auto face login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed due to server error'})

@app.route('/attendance.html')
def attendance_page():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return redirect(url_for('login_page'))
    student_id = session.get('student_id')
    student = students_collection.find_one({'student_id': student_id})
    return render_template('attendance.html', student=student)

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return redirect(url_for('login_page'))
    student_id = session.get('student_id')
    student = students_collection.find_one({'student_id': student_id})
    if student and 'face_image' in student and student['face_image']:
        face_image_base64 = base64.b64encode(student['face_image']).decode('utf-8')
        mime_type = student.get('face_image_type', 'image/jpeg')
        student['face_image_url'] = f"data:{mime_type};base64,{face_image_base64}"
    attendance_records = list(attendance_collection.find({'student_id': student_id}).sort('date', -1))
    return render_template('dashboard.html', student=student, attendance_records=attendance_records)

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return jsonify({'success': False, 'message': 'Not logged in'})

    data = request.json
    student_id = session.get('student_id') or data.get('student_id')
    program = data.get('program')
    semester = data.get('semester')
    course = data.get('course')
    face_image = data.get('face_image')

    if not all([student_id, program, semester, course, face_image]):
        return jsonify({'success': False, 'message': 'Missing required data'})

    client_ip = request.remote_addr
    t0 = time.time()

    # Decode image
    image = decode_image(face_image)
    if image is None or image.size == 0:
        return jsonify({'success': False, 'message': 'Invalid image data'})

    h, w = image.shape[:2]
    vis = image.copy()

    # 1) YuNet face detection (lightweight)
    detections = detect_faces_yunet(image)
    if not detections:
        overlay = image_to_data_uri(vis)
        log_metrics_event_normalized(
            event="reject_true",
            attempt_type="impostor",
            claimed_id=student_id,
            recognized_id=None,
            liveness_pass=False,
            distance=None,
            live_prob=None,
            latency_ms=round((time.time() - t0) * 1000.0, 2),
            client_ip=client_ip,
            reason="no_face_detected"
        )
        return jsonify({'success': False, 'message': 'No face detected for liveness', 'overlay': overlay})

    # Pick highest-score detection
    best = max(detections, key=lambda d: d["score"])
    x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
    x1e, y1e, x2e, y2e = expand_and_clip_box((x1, y1, x2, y2), scale=1.2, w=w, h=h)
    face_crop = image[y1e:y2e, x1e:x2e]
    if face_crop.size == 0:
        overlay = image_to_data_uri(vis)
        log_metrics_event_normalized(
            event="reject_true",
            attempt_type="impostor",
            claimed_id=student_id,
            recognized_id=None,
            liveness_pass=False,
            distance=None,
            live_prob=None,
            latency_ms=round((time.time() - t0) * 1000.0, 2),
            client_ip=client_ip,
            reason="failed_crop"
        )
        return jsonify({'success': False, 'message': 'Failed to crop face for liveness', 'overlay': overlay})

    # 2) Simple liveness check (lightweight)
    live_prob = simple_liveness_check(face_crop)
    is_live = live_prob >= 0.7
    label = "LIVE" if is_live else "SPOOF"
    color = (0, 200, 0) if is_live else (0, 0, 255)
    draw_live_overlay(vis, (x1e, y1e, x2e, y2e), label, live_prob, color)
    overlay_data = image_to_data_uri(vis)

    if not is_live:
        log_metrics_event_normalized(
            event="reject_true",
            attempt_type="impostor",
            claimed_id=student_id,
            recognized_id=None,
            liveness_pass=False,
            distance=None,
            live_prob=float(live_prob),
            latency_ms=round((time.time() - t0) * 1000.0, 2),
            client_ip=client_ip,
            reason="liveness_fail"
        )
        return jsonify({'success': False, 'message': f'Spoof detected or face not live (p={live_prob:.2f}).', 'overlay': overlay_data})

    # 3) Face recognition using DeepFace
    success, message = recognize_face_deepface(image, student_id, user_type='student')
    total_latency_ms = round((time.time() - t0) * 1000.0, 2)

    # Parse distance from message if available
    distance_val = None
    try:
        if "distance=" in message:
            part = message.split("distance=")[1]
            distance_val = float(part.split(",")[0].strip(") "))
    except Exception:
        pass

    # Derive reason string
    reason = None
    if not success:
        if message.startswith("Unauthorized attempt"):
            reason = "unauthorized_attempt"
        elif message.startswith("No face detected"):
            reason = "no_face_detected"
        elif message.startswith("False reject"):
            reason = "false_reject"
        elif message.startswith("Error in face recognition") or message.startswith("DeepFace"):
            reason = "recognition_error"
        else:
            reason = "not_recognized"

    # Log event
    if success:
        log_metrics_event_normalized(
            event="accept_true",
            attempt_type="genuine",
            claimed_id=student_id,
            recognized_id=student_id,
            liveness_pass=True,
            distance=distance_val,
            live_prob=float(live_prob),
            latency_ms=total_latency_ms,
            client_ip=client_ip,
            reason=None
        )
        
        # Save attendance
        attendance_data = {
            'student_id': student_id,
            'program': program,
            'semester': semester,
            'subject': course,
            'date': datetime.now().date().isoformat(),
            'time': datetime.now().time().strftime('%H:%M:%S'),
            'status': 'present',
            'created_at': datetime.now()
        }
        try:
            existing_attendance = attendance_collection.find_one({
                'student_id': student_id,
                'subject': course,
                'date': datetime.now().date().isoformat()
            })
            if existing_attendance:
                return jsonify({'success': False, 'message': 'Attendance already marked for this course today', 'overlay': overlay_data})
            attendance_collection.insert_one(attendance_data)
            return jsonify({'success': True, 'message': 'Attendance marked successfully', 'overlay': overlay_data})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Database error: {str(e)}', 'overlay': overlay_data})
    else:
        if reason == "false_reject":
            log_metrics_event_normalized(
                event="reject_false",
                attempt_type="genuine",
                claimed_id=student_id,
                recognized_id=student_id,
                liveness_pass=True,
                distance=distance_val,
                live_prob=float(live_prob),
                latency_ms=total_latency_ms,
                client_ip=client_ip,
                reason=reason
            )
        else:
            log_metrics_event_normalized(
                event="reject_true",
                attempt_type="impostor",
                claimed_id=student_id,
                recognized_id=None,
                liveness_pass=True,
                distance=distance_val,
                live_prob=float(live_prob),
                latency_ms=total_latency_ms,
                client_ip=client_ip,
                reason=reason
            )
        return jsonify({'success': False, 'message': message, 'overlay': overlay_data})

@app.route('/liveness-preview', methods=['POST'])
def liveness_preview():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return jsonify({'success': False, 'message': 'Not logged in'})
    try:
        data = request.json or {}
        face_image = data.get('face_image')
        if not face_image:
            return jsonify({'success': False, 'message': 'No image received'})
        
        image = decode_image(face_image)
        if image is None or image.size == 0:
            return jsonify({'success': False, 'message': 'Invalid image data'})
        
        h, w = image.shape[:2]
        vis = image.copy()
        detections = detect_faces_yunet(image)
        
        if not detections:
            overlay_data = image_to_data_uri(vis)
            return jsonify({
                'success': True,
                'live': False,
                'live_prob': 0.0,
                'message': 'No face detected',
                'overlay': overlay_data
            })
        
        best = max(detections, key=lambda d: d["score"])
        x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
        x1e, y1e, x2e, y2e = expand_and_clip_box((x1, y1, x2, y2), scale=1.2, w=w, h=h)
        face_crop = image[y1e:y2e, x1e:x2e]
        
        if face_crop.size == 0:
            overlay_data = image_to_data_uri(vis)
            return jsonify({
                'success': True,
                'live': False,
                'live_prob': 0.0,
                'message': 'Failed to crop face',
                'overlay': overlay_data
            })
        
        live_prob = simple_liveness_check(face_crop)
        threshold = 0.7
        label = "LIVE" if live_prob >= threshold else "SPOOF"
        color = (0, 200, 0) if label == "LIVE" else (0, 0, 255)
        
        draw_live_overlay(vis, (x1e, y1e, x2e, y2e), label, live_prob, color)
        overlay_data = image_to_data_uri(vis)
        
        return jsonify({
            'success': True,
            'live': bool(live_prob >= threshold),
            'live_prob': float(live_prob),
            'overlay': overlay_data
        })
    except Exception as e:
        print("liveness_preview error:", e)
        return jsonify({'success': False, 'message': 'Server error during preview'})

# --------- TEACHER ROUTES ---------
@app.route('/teacher_register.html')
def teacher_register_page():
    return render_template('teacher_register.html')

@app.route('/teacher_login.html')
def teacher_login_page():
    return render_template('teacher_login.html')

@app.route('/teacher_register', methods=['POST'])
def teacher_register():
    try:
        teacher_data = {
            'teacher_id': request.form.get('teacher_id'),
            'name': request.form.get('name'),
            'email': request.form.get('email'),
            'department': request.form.get('department'),
            'designation': request.form.get('designation'),
            'mobile': request.form.get('mobile'),
            'dob': request.form.get('dob'),
            'gender': request.form.get('gender'),
            'password': request.form.get('password'),
            'created_at': datetime.now()
        }
        face_image = request.form.get('face_image')
        if face_image and ',' in face_image:
            image_data = face_image.split(',')[1]
            teacher_data['face_image'] = Binary(base64.b64decode(image_data))
            teacher_data['face_image_type'] = face_image.split(',')[0].split(':')[1].split(';')[0]
        else:
            flash('Face image is required for registration.', 'danger')
            return redirect(url_for('teacher_register_page'))
        result = teachers_collection.insert_one(teacher_data)
        if result.inserted_id:
            flash('Registration successful! You can now login.', 'success')
            return redirect(url_for('teacher_login_page'))
        else:
            flash('Registration failed. Please try again.', 'danger')
            return redirect(url_for('teacher_register_page'))
    except pymongo.errors.DuplicateKeyError:
        flash('Teacher ID already exists. Please use a different ID.', 'danger')
        return redirect(url_for('teacher_register_page'))
    except Exception as e:
        flash(f'Registration failed: {str(e)}', 'danger')
        return redirect(url_for('teacher_register_page'))

@app.route('/teacher_login', methods=['POST'])
def teacher_login():
    teacher_id = request.form.get('teacher_id')
    password = request.form.get('password')
    teacher = teachers_collection.find_one({'teacher_id': teacher_id})
    if teacher and teacher['password'] == password:
        session['logged_in'] = True
        session['user_type'] = 'teacher'
        session['teacher_id'] = teacher_id
        session['name'] = teacher.get('name')
        flash('Login successful!', 'success')
        return redirect(url_for('teacher_dashboard'))
    else:
        flash('Invalid credentials. Please try again.', 'danger')
        return redirect(url_for('teacher_login_page'))

@app.route('/teacher_dashboard')
def teacher_dashboard():
    if 'logged_in' not in session or session.get('user_type') != 'teacher':
        return redirect(url_for('teacher_login_page'))
    teacher_id = session.get('teacher_id')
    teacher = teachers_collection.find_one({'teacher_id': teacher_id})
    if teacher and 'face_image' in teacher and teacher['face_image']:
        face_image_base64 = base64.b64encode(teacher['face_image']).decode('utf-8')
        mime_type = teacher.get('face_image_type', 'image/jpeg')
        teacher['face_image_url'] = f"data:{mime_type};base64,{face_image_base64}"
    return render_template('teacher_dashboard.html', teacher=teacher)

@app.route('/teacher_logout')
def teacher_logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('teacher_login_page'))

# --------- COMMON LOGOUT ---------
@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login_page'))

# --------- METRICS JSON ENDPOINTS ---------
@app.route('/metrics-data', methods=['GET'])
def metrics_data():
    data = compute_metrics()
    try:
        recent = list(metrics_events.find({}, {"_id": 0}).sort("ts", -1).limit(200))
        normalized_recent = []
        for r in recent:
            if isinstance(r.get("ts"), datetime):
                r["ts"] = r["ts"].isoformat()
            event, attempt_type = classify_event(r)
            if event and not r.get("event"):
                r["event"] = event
            if attempt_type and not r.get("attempt_type"):
                r["attempt_type"] = attempt_type
            if "liveness_pass" not in r:
                if r.get("decision") == "spoof_blocked":
                    r["liveness_pass"] = False
                elif isinstance(r.get("live_prob"), (int, float)):
                    r["liveness_pass"] = bool(r["live_prob"] >= 0.7)
                else:
                    r["liveness_pass"] = None
            normalized_recent.append(r)

        data["recent"] = normalized_recent
    except Exception as e:
        print(f"Error getting recent metrics: {e}")
        data["recent"] = []
    
    data["avg_latency_ms"] = compute_latency_avg()
    return jsonify(data)

@app.route('/metrics-json')
def metrics_json():
    m = compute_metrics()
    counts = m["counts"]
    rates = m["rates"]
    totals = m["totals"]
    avg_latency = compute_latency_avg()
    accuracy_pct = rates["accuracy"] * 100.0
    far_pct = rates["FAR"] * 100.0
    frr_pct = rates["FRR"] * 100.0

    return jsonify({
        'Accuracy': f"{accuracy_pct:.2f}%" if totals["totalAttempts"] > 0 else "N/A",
        'False Accepts (FAR)': f"{far_pct:.2f}%" if counts["impostorAttempts"] > 0 else "N/A",
        'False Rejects (FRR)': f"{frr_pct:.2f}%" if counts["genuineAttempts"] > 0 else "N/A",
        'Average Inference Time (s)': f"{(avg_latency/1000.0):.2f}" if isinstance(avg_latency, (int, float)) else "N/A",
        'Correct Recognitions': counts["trueAccepts"],
        'Total Attempts': totals["totalAttempts"],
        'Unauthorized Attempts': counts["unauthorizedRejected"],
        'enhanced': {
            'totals': {
                'attempts': totals["totalAttempts"],
                'trueAccepts': counts["trueAccepts"],
                'falseAccepts': counts["falseAccepts"],
                'trueRejects': counts["trueRejects"],
                'falseRejects': counts["falseRejects"],
                'genuineAttempts': counts["genuineAttempts"],
                'impostorAttempts': counts["impostorAttempts"],
                'unauthorizedRejected': counts["unauthorizedRejected"],
                'unauthorizedAccepted': counts["unauthorizedAccepted"],
            },
            'accuracy_pct': round(accuracy_pct, 2),
            'avg_latency_ms': round(avg_latency, 2) if isinstance(avg_latency, (int, float)) else None
        }
    })

@app.route('/metrics-events')
def metrics_events_api():
    limit = int(request.args.get("limit", 200))
    try:
        cursor = metrics_events.find({}, {"_id": 0}).sort("ts", -1).limit(limit)
        events = list(cursor)
        for ev in events:
            if isinstance(ev.get("ts"), datetime):
                ev["ts"] = ev["ts"].isoformat()
        return jsonify(events)
    except Exception as e:
        print(f"Error getting metrics events: {e}")
        return jsonify([])

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
