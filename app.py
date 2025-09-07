from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import os
import time
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import cv2
import requests
from typing import Optional, Dict, Tuple, Any
from deepface import DeepFace
import tempfile
import logging
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Evaluation Metrics Counters ---
total_attempts = 0
correct_recognitions = 0
false_accepts = 0
false_rejects = 0
unauthorized_attempts = 0
inference_times = []

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24))

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

# OpenCV DNN Face Detector
class OpenCVFaceDetector:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        # Load OpenCV DNN face detection model
        try:
            # Download models if they don't exist
            self.prototxt_path = "models/deploy.prototxt"
            self.model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Download prototxt file
            if not os.path.exists(self.prototxt_path):
                print("Downloading face detection prototxt...")
                prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
                try:
                    response = requests.get(prototxt_url, timeout=30)
                    response.raise_for_status()
                    with open(self.prototxt_path, 'w') as f:
                        f.write(response.text)
                    print("Prototxt downloaded successfully")
                except Exception as e:
                    print(f"Error downloading prototxt: {e}")
            
            # Download caffemodel file
            if not os.path.exists(self.model_path):
                print("Downloading face detection model...")
                model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
                try:
                    response = requests.get(model_url, timeout=60)
                    response.raise_for_status()
                    with open(self.model_path, 'wb') as f:
                        f.write(response.content)
                    print("Model downloaded successfully")
                except Exception as e:
                    print(f"Error downloading model: {e}")
            
            if os.path.exists(self.prototxt_path) and os.path.exists(self.model_path):
                self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)
                print("OpenCV face detection model loaded successfully")
            else:
                self.net = None
                print("Could not load face detection model")
                
        except Exception as e:
            print(f"Error loading face detection model: {e}")
            self.net = None

    def detect_faces(self, image):
        if self.net is None:
            return []
        
        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                       (300, 300), (104.0, 177.0, 123.0))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x1, y1 = box.astype("int")
                    # Ensure coordinates are within image bounds
                    x = max(0, x)
                    y = max(0, y)
                    x1 = min(w, x1)
                    y1 = min(h, y1)
                    faces.append({
                        'bbox': [x, y, x1, y1],
                        'confidence': float(confidence)
                    })
            return faces
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []

# Simple anti-spoofing using image quality metrics
class SimpleAntiSpoof:
    def __init__(self, blur_threshold=100, brightness_threshold=(50, 200)):
        self.blur_threshold = blur_threshold
        self.brightness_threshold = brightness_threshold

    def is_live(self, face_image):
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Check blur using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Check brightness
            brightness = np.mean(gray)
            
            # Simple liveness check
            is_live = (blur_score > self.blur_threshold and 
                      self.brightness_threshold[0] < brightness < self.brightness_threshold[1])
            
            confidence = min(blur_score / 200.0, 1.0)  # Normalize to 0-1
            
            return is_live, confidence
        except Exception as e:
            print(f"Error in liveness detection: {e}")
            return False, 0.0

# Initialize models
face_detector = OpenCVFaceDetector()
anti_spoof = SimpleAntiSpoof()

def decode_image(base64_image):
    try:
        if ',' in base64_image:
            base64_image = base64_image.split(',')[1]
        image_bytes = base64.b64decode(base64_image)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def get_face_embedding(image):
    """Extract face embedding using DeepFace"""
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            cv2.imwrite(tmp_file.name, image)
            
            # Get embedding using DeepFace
            embedding = DeepFace.represent(img_path=tmp_file.name, 
                                         model_name='Facenet', 
                                         enforce_detection=True,
                                         detector_backend='opencv')
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            if embedding and len(embedding) > 0:
                return np.array(embedding[0]['embedding'])
            return None
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        # Clean up temp file if it exists
        try:
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
        except:
            pass
        return None

def recognize_face(image, user_id, user_type='student'):
    """Face recognition using DeepFace"""
    global total_attempts, correct_recognitions, false_accepts, false_rejects, inference_times, unauthorized_attempts
    
    try:
        start_time = time.time()
        
        # Get face embedding from input image
        test_embedding = get_face_embedding(image)
        if test_embedding is None:
            return False, "No face detected"

        # Get user from database
        if user_type == 'student':
            user = students_collection.find_one({'student_id': user_id})
        else:
            user = teachers_collection.find_one({'teacher_id': user_id})

        if not user or 'face_image' not in user:
            unauthorized_attempts += 1
            return False, f"No reference face found for {user_type} ID {user_id}"

        # Decode reference image
        ref_image_bytes = user['face_image']
        ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
        ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
        
        # Get reference embedding
        ref_embedding = get_face_embedding(ref_image)
        if ref_embedding is None:
            return False, "No face detected in reference image"

        # Calculate cosine similarity
        similarity = cosine_similarity([test_embedding], [ref_embedding])[0][0]
        
        # Threshold for recognition (adjust as needed)
        threshold = 0.6
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        total_attempts += 1

        if similarity > threshold:
            correct_recognitions += 1
            return True, f"Face recognized (similarity={similarity:.3f}, time={inference_time:.2f}s)"
        else:
            unauthorized_attempts += 1
            return False, f"Unauthorized attempt detected (similarity={similarity:.3f})"
            
    except Exception as e:
        return False, f"Error in face recognition: {str(e)}"

# Helper functions for metrics
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
    """Returns (event, attempt_type)"""
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
    """Compute evaluation metrics"""
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

def compute_latency_avg(limit: int = 300) -> Optional[float]:
    cursor = metrics_events.find({"latency_ms": {"$exists": True}}, {"latency_ms": 1, "_id": 0}).sort("ts", -1).limit(limit)
    vals = [float(d["latency_ms"]) for d in cursor if isinstance(d.get("latency_ms"), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)

def image_to_data_uri(img_bgr: np.ndarray) -> Optional[str]:
    success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not success:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# --------- ROUTES ---------
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
    if image is None:
        flash('Invalid image data.', 'danger')
        return redirect(url_for('login_page'))

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
    test_embedding = get_face_embedding(image)
    
    if test_embedding is None:
        flash('No face detected. Please try again.', 'danger')
        return redirect(url_for('login_page'))

    for user in users:
        try:
            ref_image_bytes = user['face_image']
            ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
            ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
            ref_embedding = get_face_embedding(ref_image)
            
            if ref_embedding is None:
                continue
                
            similarity = cosine_similarity([test_embedding], [ref_embedding])[0][0]
            
            if similarity > 0.6:
                session['logged_in'] = True
                session['user_type'] = face_role
                session[id_field] = user[id_field]
                session['name'] = user.get('name')
                flash('Face login successful!', 'success')
                return redirect(url_for(dashboard_route))
        except Exception as e:
            print(f"Error processing user {user.get(id_field)}: {e}")
            continue

    flash('Face not recognized. Please try again or contact admin.', 'danger')
    return redirect(url_for('login_page'))

@app.route('/auto-face-login', methods=['POST'])
def auto_face_login():
    try:
        data = request.json
        face_image = data.get('face_image')
        face_role = data.get('face_role', 'student')
        
        if not face_image:
            return jsonify({'success': False, 'message': 'No image received'})
            
        image = decode_image(face_image)
        if image is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})
            
        test_embedding = get_face_embedding(image)
        if test_embedding is None:
            return jsonify({'success': False, 'message': 'No face detected'})

        if face_role == 'teacher':
            collection = teachers_collection
            id_field = 'teacher_id'
            dashboard_route = '/teacher_dashboard'
        else:
            collection = students_collection
            id_field = 'student_id'
            dashboard_route = '/dashboard'

        users = collection.find({'face_image': {'$exists': True, '$ne': None}})
        
        for user in users:
            try:
                ref_image_array = np.frombuffer(user['face_image'], np.uint8)
                ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
                ref_embedding = get_face_embedding(ref_image)
                
                if ref_embedding is None:
                    continue
                    
                similarity = cosine_similarity([test_embedding], [ref_embedding])[0][0]
                
                if similarity > 0.6:
                    session['logged_in'] = True
                    session['user_type'] = face_role
                    session[id_field] = user[id_field]
                    session['name'] = user.get('name')
                    return jsonify({
                        'success': True,
                        'message': f'Welcome {user["name"]}! Redirecting...',
                        'redirect_url': dashboard_route,
                        'face_role': face_role
                    })
            except Exception as e:
                print(f"Error processing user {user.get(id_field)}: {e}")
                continue

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

    t0 = time.time()
    image = decode_image(face_image)
    
    if image is None or image.size == 0:
        return jsonify({'success': False, 'message': 'Invalid image data'})

    # Face detection
    faces = face_detector.detect_faces(image)
    if not faces:
        return jsonify({'success': False, 'message': 'No face detected'})

    # Get the best face detection
    best_face = max(faces, key=lambda f: f['confidence'])
    x, y, x1, y1 = best_face['bbox']
    
    # Extract face region with some padding
    padding = 20
    h, w = image.shape[:2]
    y_start = max(0, y - padding)
    y_end = min(h, y1 + padding)
    x_start = max(0, x - padding)
    x_end = min(w, x1 + padding)
    
    face_crop = image[y_start:y_end, x_start:x_end]
    
    if face_crop.size == 0:
        return jsonify({'success': False, 'message': 'Failed to crop face'})

    # Simple liveness check
    is_live, live_confidence = anti_spoof.is_live(face_crop)
    
    if not is_live:
        log_metrics_event_normalized(
            event="reject_true",
            attempt_type="impostor",
            claimed_id=student_id,
            recognized_id=None,
            liveness_pass=False,
            distance=None,
            live_prob=float(live_confidence),
            latency_ms=round((time.time() - t0) * 1000.0, 2),
            client_ip=request.remote_addr,
            reason="liveness_fail"
        )
        return jsonify({'success': False, 'message': f'Liveness check failed (confidence={live_confidence:.2f})'})

    # Face recognition
    success, message = recognize_face(image, student_id, user_type='student')
    total_latency_ms = round((time.time() - t0) * 1000.0, 2)

    # Parse similarity from message if available
    similarity_val = None
    try:
        if "similarity=" in message:
            part = message.split("similarity=")[1]
            similarity_val = float(part.split(",")[0].strip(") "))
    except Exception:
        pass

    if success:
        # Log successful recognition
        log_metrics_event_normalized(
            event="accept_true",
            attempt_type="genuine",
            claimed_id=student_id,
            recognized_id=student_id,
            liveness_pass=True,
            distance=similarity_val,
            live_prob=float(live_confidence),
            latency_ms=total_latency_ms,
            client_ip=request.remote_addr,
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
                return jsonify({'success': False, 'message': 'Attendance already marked for this course today'})
            
            attendance_collection.insert_one(attendance_data)
            return jsonify({'success': True, 'message': 'Attendance marked successfully'})
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Database error: {str(e)}'})
    else:
        # Log failed recognition
        reason = "unauthorized_attempt" if "Unauthorized attempt" in message else "not_recognized"
        
        log_metrics_event_normalized(
            event="reject_true",
            attempt_type="impostor",
            claimed_id=student_id,
            recognized_id=None,
            liveness_pass=True,
            distance=similarity_val,
            live_prob=float(live_confidence),
            latency_ms=total_latency_ms,
            client_ip=request.remote_addr,
            reason=reason
        )
        
        return jsonify({'success': False, 'message': message})

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
        
        # Face detection
        faces = face_detector.detect_faces(image)
        if not faces:
            return jsonify({
                'success': True,
                'live': False,
                'live_prob': 0.0,
                'message': 'No face detected'
            })
        
        # Get best face and crop
        best_face = max(faces, key=lambda f: f['confidence'])
        x, y, x1, y1 = best_face['bbox']
        
        padding = 20
        h, w = image.shape[:2]
        y_start = max(0, y - padding)
        y_end = min(h, y1 + padding)
        x_start = max(0, x - padding)
        x_end = min(w, x1 + padding)
        
        face_crop = image[y_start:y_end, x_start:x_end]
        
        if face_crop.size == 0:
            return jsonify({
                'success': True,
                'live': False,
                'live_prob': 0.0,
                'message': 'Failed to crop face'
            })
        
        # Liveness check
        is_live, live_confidence = anti_spoof.is_live(face_crop)
        
        # Draw rectangle on original image for preview
        vis = image.copy()
        color = (0, 255, 0) if is_live else (0, 0, 255)
        cv2.rectangle(vis, (x, y), (x1, y1), color, 2)
        
        label = "LIVE" if is_live else "NOT LIVE"
        cv2.putText(vis, f"{label} ({live_confidence:.2f})", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        overlay_data = image_to_data_uri(vis)
        
        return jsonify({
            'success': True,
            'live': is_live,
            'live_prob': float(live_confidence),
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

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login_page'))

# --------- METRICS ENDPOINTS ---------
@app.route('/metrics-data', methods=['GET'])
def metrics_data():
    data = compute_metrics()
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
    cursor = metrics_events.find({}, {"_id": 0}).sort("ts", -1).limit(limit)
    events = list(cursor)
    
    for ev in events:
        if isinstance(ev.get("ts"), datetime):
            ev["ts"] = ev["ts"].isoformat()
    
    return jsonify(events)

# Health check route for Coolify
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('home.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)
