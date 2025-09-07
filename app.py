from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from imutils.face_utils import FaceAligner, rect_to_bb  # noqa: F401
import imutils  # noqa: F401
import os
import time
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
import base64
from datetime import datetime, timezone
from dotenv import load_dotenv
import numpy as np
import dlib
import cv2
import bz2
import requests
import logging
import sys
from typing import Optional, Dict, Tuple, Any

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

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

# Initialize Flask app with production configuration
app = Flask(__name__, static_folder='app/static', template_folder='app/templates')

# Production-ready secret key configuration
SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY:
    if os.getenv('FLASK_ENV') == 'production':
        logger.error("SECRET_KEY environment variable is required in production!")
        sys.exit(1)
    else:
        SECRET_KEY = os.urandom(24)
        logger.warning("Using random secret key for development")

app.secret_key = SECRET_KEY

# Production configuration
app.config.update(
    SESSION_COOKIE_SECURE=True if os.getenv('FLASK_ENV') == 'production' else False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=3600,  # 1 hour
)

# MongoDB Connection with retry logic
def connect_mongodb(max_retries=5, retry_delay=5):
    """Connect to MongoDB with retry logic for production"""
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    
    for attempt in range(max_retries):
        try:
            client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                maxPoolSize=50,
                retryWrites=True
            )
            # Test connection
            client.admin.command('ping')
            logger.info("MongoDB connection successful")
            return client
        except Exception as e:
            logger.error(f"MongoDB connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("All MongoDB connection attempts failed")
                if os.getenv('FLASK_ENV') == 'production':
                    sys.exit(1)
                return None

# Initialize MongoDB
client = connect_mongodb()
if client:
    db = client['face_attendance_system']
    students_collection = db['students']
    teachers_collection = db['teachers']
    attendance_collection = db['attendance']
    metrics_events = db['metrics_events']

    try:
        # Create indexes
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
        logger.info("MongoDB indexes created successfully")
    except Exception as e:
        logger.error(f"Error creating MongoDB indexes: {e}")
else:
    logger.warning("Running without MongoDB connection")

# ---------------- OpenCV DNN Implementation (Replaces ONNX Runtime) ----------------

def _letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    shape = image.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        r = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, r, (left, top)

def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

class YoloV5FaceDetector:
    """YOLOv5 Face Detection using OpenCV DNN (replaces ONNX Runtime)"""
    
    def __init__(self, model_path: str, input_size: int = 640, conf_threshold: float = 0.3, iou_threshold: float = 0.45):
        if not os.path.exists(model_path):
            logger.error(f"YOLO model not found at: {model_path}")
            raise FileNotFoundError(f"YOLO model not found at: {model_path}")
        
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        
        # Load ONNX model with OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(model_path)
        
        # Set backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        logger.info(f"YOLO Face Detector loaded with OpenCV DNN from {model_path}")

    @staticmethod
    def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def detect(self, image_bgr: np.ndarray, max_det: int = 20):
        """Detect faces in image using OpenCV DNN"""
        try:
            h0, w0 = image_bgr.shape[:2]
            img, ratio, dwdh = _letterbox(image_bgr, new_shape=(self.input_size, self.input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            # Create blob
            blob = cv2.dnn.blobFromImage(img, 1.0, (self.input_size, self.input_size), swapRB=False)
            
            # Run inference
            self.net.setInput(blob)
            preds = self.net.forward()[0]
            
            if preds.ndim == 3 and preds.shape[0] == 1:
                preds = preds[0]
            if preds.ndim != 2:
                raise RuntimeError(f"Unexpected YOLO output shape: {preds.shape}")
            
            num_attrs = preds.shape[1]
            has_landmarks = num_attrs >= 15
            boxes_xywh = preds[:, 0:4]
            
            if has_landmarks:
                scores = preds[:, 4]
            else:
                obj = preds[:, 4:5]
                cls_scores = preds[:, 5:]
                if cls_scores.size == 0:
                    scores = obj.squeeze(-1)
                else:
                    class_conf = cls_scores.max(axis=1, keepdims=True)
                    scores = (obj * class_conf).squeeze(-1)
            
            keep = scores > self.conf_threshold
            boxes_xywh = boxes_xywh[keep]
            scores = scores[keep]
            
            if boxes_xywh.shape[0] == 0:
                return []
            
            boxes_xyxy = self._xywh2xyxy(boxes_xywh)
            boxes_xyxy[:, [0, 2]] -= dwdh[0]
            boxes_xyxy[:, [1, 3]] -= dwdh[1]
            boxes_xyxy /= ratio
            boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w0 - 1)
            boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h0 - 1)
            boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w0 - 1)
            boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h0 - 1)
            
            keep_inds = _nms(boxes_xyxy, scores, self.iou_threshold)
            if len(keep_inds) > max_det:
                keep_inds = keep_inds[:max_det]
            
            dets = []
            for i in keep_inds:
                dets.append({"bbox": boxes_xyxy[i].tolist(), "score": float(scores[i])})
            return dets
            
        except Exception as e:
            logger.error(f"Error in YOLO detection: {e}")
            return []

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

class AntiSpoofBinary:
    """Anti-spoofing using OpenCV DNN (replaces ONNX Runtime)"""
    
    def __init__(self, model_path: str, input_size: int = 128, rgb: bool = True, normalize: bool = True,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), live_index: int = 1):
        if not os.path.exists(model_path):
            logger.error(f"Anti-spoof model not found at: {model_path}")
            raise FileNotFoundError(f"Anti-spoof model not found at: {model_path}")
            
        self.input_size = int(input_size)
        self.rgb = bool(rgb)
        self.normalize = bool(normalize)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.live_index = int(live_index)
        
        # Load ONNX model with OpenCV DNN
        self.net = cv2.dnn.readNetFromONNX(model_path)
        
        # Set backend and target
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        logger.info(f"Anti-spoof model loaded with OpenCV DNN from {model_path}")

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if self.normalize:
            img = (img - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img, 1.0, (self.input_size, self.input_size), swapRB=False)
        return blob

    def predict_live_prob(self, face_bgr: np.ndarray) -> float:
        """Predict liveness probability using OpenCV DNN"""
        try:
            inp = self._preprocess(face_bgr)
            
            # Run inference
            self.net.setInput(inp)
            out = self.net.forward()[0]
            
            if out.ndim > 1:
                out = np.squeeze(out)
            
            if out.size == 2:
                # Softmax for 2-class output
                vec = out.astype(np.float32)
                probs = np.exp(vec - np.max(vec))
                probs = probs / (np.sum(probs) + 1e-9)
                live_prob = float(probs[self.live_index])
            else:
                # Sigmoid for single output
                live_prob = float(_sigmoid(out.astype(np.float32)))
            
            return max(0.0, min(1.0, live_prob))
            
        except Exception as e:
            logger.error(f"Error in anti-spoof prediction: {e}")
            return 0.0  # Default to not live if error occurs

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

# Model initialization with error handling
def initialize_models():
    """Initialize all ML models with proper error handling"""
    global yolo_face, anti_spoof_bin, detector, shape_predictor, face_recognition_model
    
    # Model paths
    YOLO_FACE_MODEL_PATH = "models/yolov5s-face.onnx"
    ANTI_SPOOF_BIN_MODEL_PATH = "models/anti_spoofing/AntiSpoofing_bin_1.5_128.onnx"
    DLIB_MODELS_DIR = 'models'
    SHAPE_PREDICTOR_PATH = os.path.join(DLIB_MODELS_DIR, 'shape_predictor_68_face_landmarks.dat')
    FACE_RECOGNITION_MODEL_PATH = os.path.join(DLIB_MODELS_DIR, 'dlib_face_recognition_resnet_model_v1.dat')
    
    try:
        # Initialize YOLO face detector with OpenCV DNN
        if os.path.exists(YOLO_FACE_MODEL_PATH):
            yolo_face = YoloV5FaceDetector(YOLO_FACE_MODEL_PATH, input_size=640, conf_threshold=0.3, iou_threshold=0.45)
            logger.info("YOLO Face Detector loaded successfully with OpenCV DNN")
        else:
            logger.warning(f"YOLO model not found at {YOLO_FACE_MODEL_PATH}, using dlib only")
            yolo_face = None
        
        # Initialize anti-spoof model with OpenCV DNN
        if os.path.exists(ANTI_SPOOF_BIN_MODEL_PATH):
            anti_spoof_bin = AntiSpoofBinary(ANTI_SPOOF_BIN_MODEL_PATH, input_size=128, rgb=True, normalize=True, live_index=1)
            logger.info("Anti-spoof model loaded successfully with OpenCV DNN")
        else:
            logger.warning(f"Anti-spoof model not found at {ANTI_SPOOF_BIN_MODEL_PATH}, liveness detection disabled")
            anti_spoof_bin = None
        
        # Ensure dlib models directory exists
        os.makedirs(DLIB_MODELS_DIR, exist_ok=True)
        
        # Download dlib face recognition model if missing
        download_and_extract_model(FACE_RECOGNITION_MODEL_PATH)
        
        # Initialize dlib models
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        face_recognition_model = dlib.face_recognition_model_v1(FACE_RECOGNITION_MODEL_PATH)
        logger.info("Dlib models loaded successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        if os.getenv('FLASK_ENV') == 'production':
            sys.exit(1)
        return False

def download_and_extract_model(model_path: str):
    """Download dlib face recognition model if not present"""
    if not os.path.exists(model_path):
        try:
            logger.info("Downloading dlib_face_recognition_resnet_model_v1.dat.bz2...")
            url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
            response = requests.get(url, timeout=300)
            response.raise_for_status()
            
            compressed_path = model_path + ".bz2"
            with open(compressed_path, 'wb') as f:
                f.write(response.content)
            
            with bz2.BZ2File(compressed_path) as f_in:
                with open(model_path, 'wb') as f_out:
                    f_out.write(f_in.read())
            
            os.remove(compressed_path)
            logger.info("Dlib face recognition model downloaded and extracted successfully")
        except Exception as e:
            logger.error(f"Error downloading dlib model: {e}")
            raise

# Initialize models
models_loaded = initialize_models()

def decode_image(base64_image):
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]
    image_bytes = base64.b64decode(base64_image)
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

def align_face(image, shape):
    """Align the face using eye landmarks"""
    left_eye = (shape.part(36).x, shape.part(36).y)
    right_eye = (shape.part(45).x, shape.part(45).y)
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_image

def get_face_features(image):
    """Extract aligned face features using ResNet model"""
    if not models_loaded:
        logger.error("Models not loaded, cannot extract face features")
        return None
        
    try:
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = detector(rgb_img, 1)
        if len(dets) == 0:
            return None
        face = max(dets, key=lambda rect: rect.width() * rect.height())
        shape = shape_predictor(rgb_img, face)
        aligned_img = align_face(rgb_img, shape)
        dets_aligned = detector(aligned_img, 1)
        if len(dets_aligned) == 0:
            return None
        face_aligned = max(dets_aligned, key=lambda rect: rect.width() * rect.height())
        shape_aligned = shape_predictor(aligned_img, face_aligned)
        face_descriptor = face_recognition_model.compute_face_descriptor(aligned_img, shape_aligned)
        return np.array(face_descriptor)
    except Exception as e:
        logger.error(f"Error extracting face features: {e}")
        return None

def recognize_face(image, user_id, user_type='student'):
    """Face recognition with error handling"""
    global total_attempts, correct_recognitions, false_accepts, false_rejects, inference_times, unauthorized_attempts
    
    if not models_loaded:
        return False, "Models not loaded"
        
    if not client:
        return False, "Database connection unavailable"
    
    try:
        start_time = time.time()
        features = get_face_features(image)
        if features is None:
            return False, "No face detected"

        if user_type == 'student':
            user = students_collection.find_one({'student_id': user_id})
        else:
            user = teachers_collection.find_one({'teacher_id': user_id})

        if not user or 'face_image' not in user:
            unauthorized_attempts += 1
            return False, f"No reference face found for {user_type} ID {user_id}"

        ref_image_bytes = user['face_image']
        ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
        ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
        ref_features = get_face_features(ref_image)
        if ref_features is None:
            return False, "No face detected in reference image"

        dist = np.linalg.norm(features - ref_features)
        threshold = 0.6
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        total_attempts += 1

        if dist < threshold:
            correct_recognitions += 1
            return True, f"Face recognized (distance={dist:.3f}, time={inference_time:.2f}s)"
        else:
            unauthorized_attempts += 1
            return False, f"Unauthorized attempt detected (distance={dist:.3f})"
    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        return False, f"Error in face recognition: {str(e)}"

# Metrics helpers (keeping your existing implementation)
def log_metrics_event(event: dict):
    if not client:
        logger.warning("Cannot log metrics event - no database connection")
        return
    try:
        metrics_events.insert_one(event)
    except Exception as e:
        logger.error(f"Failed to log metrics event: {e}")

def log_metrics_event_normalized(*, event: str, attempt_type: str, claimed_id: Optional[str],
    recognized_id: Optional[str], liveness_pass: bool, distance: Optional[float],
    live_prob: Optional[float], latency_ms: Optional[float], client_ip: Optional[str],
    reason: Optional[str] = None):
    
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
    if not client:
        return {"counts": {}, "rates": {}, "totals": {"totalAttempts": 0}}
        
    cursor = metrics_events.find({}, {"_id": 0}).sort("ts", -1).limit(limit)
    counts = {
        "trueAccepts": 0, "falseAccepts": 0, "trueRejects": 0, "falseRejects": 0,
        "genuineAttempts": 0, "impostorAttempts": 0, "unauthorizedRejected": 0, "unauthorizedAccepted": 0,
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
        "rates": {"FAR": FAR, "FRR": FRR, "accuracy": accuracy},
        "totals": {"totalAttempts": total_attempts_calc}
    }

def compute_latency_avg(limit: int = 300) -> Optional[float]:
    if not client:
        return None
    cursor = metrics_events.find({"latency_ms": {"$exists": True}}, {"latency_ms": 1, "_id": 0}).sort("ts", -1).limit(limit)
    vals = [float(d["latency_ms"]) for d in cursor if isinstance(d.get("latency_ms"), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)

# Health check endpoint for Coolify
@app.route('/health')
def health_check():
    """Health check endpoint for load balancers and monitoring"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'models_loaded': models_loaded,
        'database_connected': client is not None,
        'version': '1.0.0'
    }
    
    if models_loaded:
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            if yolo_face:
                yolo_face.detect(test_img, max_det=1)
            health_status['models_status'] = 'operational'
        except Exception as e:
            health_status['models_status'] = f'error: {str(e)}'
            health_status['status'] = 'degraded'
    else:
        health_status['models_status'] = 'not_loaded'
        health_status['status'] = 'degraded'
    
    if client:
        try:
            client.admin.command('ping')
            health_status['database_status'] = 'connected'
        except Exception as e:
            health_status['database_status'] = f'error: {str(e)}'
            health_status['status'] = 'degraded'
    else:
        health_status['database_status'] = 'disconnected'
        health_status['status'] = 'degraded'
    
    status_code = 200 if health_status['status'] in ['healthy', 'degraded'] else 503
    return jsonify(health_status), status_code

# Simple error handlers (no template dependency)
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

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
    if not client:
        flash('Service temporarily unavailable. Please try again later.', 'danger')
        return redirect(url_for('register_page'))
        
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
        logger.error(f"Registration error: {e}")
        flash(f'Registration failed: {str(e)}', 'danger')
        return redirect(url_for('register_page'))

@app.route('/login', methods=['POST'])
def login():
    if not client:
        flash('Service temporarily unavailable. Please try again later.', 'danger')
        return redirect(url_for('login_page'))
        
    try:
        student_id = request.form.get('student_id')
        password = request.form.get('password')
        
        if not student_id or not password:
            flash('Please provide both student ID and password.', 'danger')
            return redirect(url_for('login_page'))
        
        student = students_collection.find_one({'student_id': student_id})

        if student and student.get('password') == password:
            session['logged_in'] = True
            session['user_type'] = 'student'
            session['student_id'] = student_id
            session['name'] = student.get('name')
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
            return redirect(url_for('login_page'))
    except Exception as e:
        logger.error(f"Login error: {e}")
        flash('Login failed. Please try again.', 'danger')
        return redirect(url_for('login_page'))

@app.route('/face-login', methods=['POST'])
def face_login():
    if not client or not models_loaded:
        flash('Service temporarily unavailable. Please try again later.', 'danger')
        return redirect(url_for('login_page'))

    face_image = request.form.get('face_image')
    face_role = request.form.get('face_role', 'student')

    if not face_image or not face_role:
        flash('Face image and role are required for face login.', 'danger')
        return redirect(url_for('login_page'))

    try:
        image = decode_image(face_image)

        # Select collection and ID/key field based on role
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
        test_features = get_face_features(image)
        if test_features is None:
            flash('No face detected. Please try again.', 'danger')
            return redirect(url_for('login_page'))

        for user in users:
            try:
                ref_image_bytes = user['face_image']
                ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
                ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
                ref_features = get_face_features(ref_image)
                if ref_features is None:
                    continue
                dist = np.linalg.norm(test_features - ref_features)
                if dist < 0.6:
                    session['logged_in'] = True
                    session['user_type'] = face_role
                    session[id_field] = user[id_field]
                    session['name'] = user.get('name')
                    flash('Face login successful!', 'success')
                    return redirect(url_for(dashboard_route))
            except Exception as e:
                logger.error(f"Error processing user {user.get(id_field)}: {e}")
                continue

        flash('Face not recognized. Please try again or contact admin.', 'danger')
        return redirect(url_for('login_page'))
    except Exception as e:
        logger.error(f"Face login error: {e}")
        flash('Face login failed. Please try again.', 'danger')
        return redirect(url_for('login_page'))

@app.route('/auto-face-login', methods=['POST'])
def auto_face_login():
    """Enhanced auto face login with role support"""
    if not client or not models_loaded:
        return jsonify({'success': False, 'message': 'Service temporarily unavailable'})
        
    try:
        data = request.json
        face_image = data.get('face_image')
        face_role = data.get('face_role', 'student')
        if not face_image:
            return jsonify({'success': False, 'message': 'No image received'})
            
        image = decode_image(face_image)
        test_features = get_face_features(image)
        if test_features is None:
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
                ref_features = get_face_features(ref_image)
                if ref_features is None:
                    continue
                dist = np.linalg.norm(test_features - ref_features)
                if dist < 0.6:  # Face recognized
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
                logger.error(f"Error processing user {user.get(id_field)}: {e}")
                continue

        return jsonify({'success': False, 'message': f'Face not recognized in {face_role} database'})
    except Exception as e:
        logger.error(f"Auto face login error: {e}")
        return jsonify({'success': False, 'message': 'Login failed due to server error'})

@app.route('/attendance.html')
def attendance_page():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return redirect(url_for('login_page'))
    if not client:
        flash('Service temporarily unavailable.', 'danger')
        return redirect(url_for('login_page'))
    try:
        student_id = session.get('student_id')
        student = students_collection.find_one({'student_id': student_id})
        return render_template('attendance.html', student=student)
    except Exception as e:
        logger.error(f"Attendance page error: {e}")
        flash('Error loading attendance page.', 'danger')
        return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return redirect(url_for('login_page'))
    
    if not client:
        flash('Service temporarily unavailable.', 'danger')
        return redirect(url_for('login_page'))
        
    try:
        student_id = session.get('student_id')
        student = students_collection.find_one({'student_id': student_id})
        if student and 'face_image' in student and student['face_image']:
            face_image_base64 = base64.b64encode(student['face_image']).decode('utf-8')
            mime_type = student.get('face_image_type', 'image/jpeg')
            student['face_image_url'] = f"data:{mime_type};base64,{face_image_base64}"
        attendance_records = list(attendance_collection.find({'student_id': student_id}).sort('date', -1))
        return render_template('dashboard.html', student=student, attendance_records=attendance_records)
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        flash('Error loading dashboard.', 'danger')
        return redirect(url_for('login_page'))

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return jsonify({'success': False, 'message': 'Not logged in'})

    if not client:
        return jsonify({'success': False, 'message': 'Service temporarily unavailable'})

    if not models_loaded:
        return jsonify({'success': False, 'message': 'AI models not available'})

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

    try:
        # Decode image
        image = decode_image(face_image)
        if image is None or image.size == 0:
            return jsonify({'success': False, 'message': 'Invalid image data'})

        h, w = image.shape[:2]
        vis = image.copy()

        # 1) YOLOv5-face detection (if available)
        if yolo_face:
            detections = yolo_face.detect(image, max_det=20)
            if not detections:
                overlay = image_to_data_uri(vis)
                log_metrics_event_normalized(
                    event="reject_true", attempt_type="impostor", claimed_id=student_id,
                    recognized_id=None, liveness_pass=False, distance=None, live_prob=None,
                    latency_ms=round((time.time() - t0) * 1000.0, 2), client_ip=client_ip,
                    reason="no_face_detected"
                )
                return jsonify({'success': False, 'message': 'No face detected for liveness', 'overlay': overlay})

            # pick highest-score detection
            best = max(detections, key=lambda d: d["score"])
            x1, y1, x2, y2 = [int(v) for v in best["bbox"]]
            x1e, y1e, x2e, y2e = expand_and_clip_box((x1, y1, x2, y2), scale=1.2, w=w, h=h)
            face_crop = image[y1e:y2e, x1e:x2e]
            if face_crop.size == 0:
                overlay = image_to_data_uri(vis)
                log_metrics_event_normalized(
                    event="reject_true", attempt_type="impostor", claimed_id=student_id,
                    recognized_id=None, liveness_pass=False, distance=None, live_prob=None,
                    latency_ms=round((time.time() - t0) * 1000.0, 2), client_ip=client_ip,
                    reason="failed_crop"
                )
                return jsonify({'success': False, 'message': 'Failed to crop face for liveness', 'overlay': overlay})

            # 2) Binary Anti-Spoof (if available)
            if anti_spoof_bin:
                live_prob = anti_spoof_bin.predict_live_prob(face_crop)
                is_live = live_prob >= 0.7
                label = "LIVE" if is_live else "SPOOF"
                color = (0, 200, 0) if is_live else (0, 0, 255)
                draw_live_overlay(vis, (x1e, y1e, x2e, y2e), label, live_prob, color)
                overlay_data = image_to_data_uri(vis)

                if not is_live:
                    log_metrics_event_normalized(
                        event="reject_true", attempt_type="impostor", claimed_id=student_id,
                        recognized_id=None, liveness_pass=False, distance=None, live_prob=float(live_prob),
                        latency_ms=round((time.time() - t0) * 1000.0, 2), client_ip=client_ip,
                        reason="liveness_fail"
                    )
                    return jsonify({'success': False, 'message': f'Spoof detected or face not live (p={live_prob:.2f}).', 'overlay': overlay_data})
            else:
                live_prob = 1.0  # Assume live if no anti-spoof model
                overlay_data = image_to_data_uri(vis)
        else:
            # Use dlib detection fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if not faces:
                overlay = image_to_data_uri(vis)
                return jsonify({'success': False, 'message': 'No face detected', 'overlay': overlay})
            live_prob = 1.0  # Assume live if no YOLO/anti-spoof
            overlay_data = image_to_data_uri(vis)

        # 3) Face recognition
        success, message = recognize_face(image, student_id, user_type='student')
        total_latency_ms = round((time.time() - t0) * 1000.0, 2)

        # Parse distance from message if available
        distance_val = None
        try:
            if "distance=" in message:
                part = message.split("distance=")[1]
                distance_val = float(part.split(",")[0].strip(") "))
        except Exception:
            pass

        # Determine reason
        reason = None
        if not success:
            if message.startswith("Unauthorized attempt"):
                reason = "unauthorized_attempt"
            elif message.startswith("No face detected"):
                reason = "no_face_detected"
            elif message.startswith("False reject"):
                reason = "false_reject"
            elif message.startswith("Error in face recognition"):
                reason = "recognition_error"
            else:
                reason = "not_recognized"

        # Log event
        if success:
            log_metrics_event_normalized(
                event="accept_true", attempt_type="genuine", claimed_id=student_id,
                recognized_id=student_id, liveness_pass=True, distance=distance_val,
                live_prob=float(live_prob), latency_ms=total_latency_ms, client_ip=client_ip,
                reason=None
            )
            
            # Mark attendance
            attendance_data = {
                'student_id': student_id, 'program': program, 'semester': semester,
                'subject': course, 'date': datetime.now().date().isoformat(),
                'time': datetime.now().time().strftime('%H:%M:%S'), 'status': 'present',
                'created_at': datetime.now()
            }
            
            existing_attendance = attendance_collection.find_one({
                'student_id': student_id, 'subject': course,
                'date': datetime.now().date().isoformat()
            })
            
            if existing_attendance:
                return jsonify({'success': False, 'message': 'Attendance already marked for this course today', 'overlay': overlay_data})
            
            attendance_collection.insert_one(attendance_data)
            return jsonify({'success': True, 'message': 'Attendance marked successfully', 'overlay': overlay_data})
        else:
            # Log rejection
            if reason == "false_reject":
                log_metrics_event_normalized(
                    event="reject_false", attempt_type="genuine", claimed_id=student_id,
                    recognized_id=student_id, liveness_pass=True, distance=distance_val,
                    live_prob=float(live_prob), latency_ms=total_latency_ms, client_ip=client_ip,
                    reason=reason
                )
            else:
                log_metrics_event_normalized(
                    event="reject_true", attempt_type="impostor", claimed_id=student_id,
                    recognized_id=None, liveness_pass=True, distance=distance_val,
                    live_prob=float(live_prob), latency_ms=total_latency_ms, client_ip=client_ip,
                    reason=reason
                )
            return jsonify({'success': False, 'message': message, 'overlay': overlay_data})

    except Exception as e:
        logger.error(f"Mark attendance error: {e}")
        return jsonify({'success': False, 'message': 'Server error occurred'})

@app.route('/liveness-preview', methods=['POST'])
def liveness_preview():
    if 'logged_in' not in session or session.get('user_type') != 'student':
        return jsonify({'success': False, 'message': 'Not logged in'})
    
    if not models_loaded:
        return jsonify({'success': False, 'message': 'AI models not available'})
        
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
        
        if yolo_face:
            detections = yolo_face.detect(image, max_det=10)
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
                
            if anti_spoof_bin:
                live_prob = anti_spoof_bin.predict_live_prob(face_crop)
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
            else:
                # No anti-spoof model, just show face detection
                cv2.rectangle(vis, (x1e, y1e), (x2e, y2e), (0, 255, 0), 2)
                overlay_data = image_to_data_uri(vis)
                return jsonify({
                    'success': True,
                    'live': True,
                    'live_prob': 1.0,
                    'message': 'Face detected (liveness check disabled)',
                    'overlay': overlay_data
                })
        else:
            # Fallback to dlib detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if faces:
                # Draw rectangle around largest face
                face = max(faces, key=lambda rect: rect.width() * rect.height())
                cv2.rectangle(vis, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                overlay_data = image_to_data_uri(vis)
                return jsonify({
                    'success': True,
                    'live': True,
                    'live_prob': 1.0,
                    'message': 'Face detected (YOLO disabled)',
                    'overlay': overlay_data
                })
            else:
                overlay_data = image_to_data_uri(vis)
                return jsonify({
                    'success': True,
                    'live': False,
                    'live_prob': 0.0,
                    'message': 'No face detected',
                    'overlay': overlay_data
                })
