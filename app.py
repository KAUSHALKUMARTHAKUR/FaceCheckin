from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
import onnxruntime as ort
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
import gdown
from typing import Optional, Dict, Tuple, Any
from deepface import DeepFace
from deepface.commons import functions
import tensorflow as tf

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
app.secret_key = os.urandom(24)

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

# ---------------- Model Download and Loading Functions ----------------

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive using direct requests approach"""
    try:
        if not os.path.exists(destination):
            print(f"Downloading {destination}...")
            # Use direct download URL format
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            session = requests.Session()
            response = session.get(url, stream=True)
            
            # Handle Google Drive's virus scan warning for large files
            if 'download_warning' in response.text:
                for line in response.text.split('\n'):
                    if 'confirm=' in line:
                        confirm_token = line.split('confirm=')[1].split('&')[0]
                        url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                        response = session.get(url, stream=True)
                        break
            
            if response.status_code == 200:
                with open(destination, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {destination}")
                return True
            else:
                print(f"Failed to download {destination}: HTTP {response.status_code}")
                return False
        else:
            print(f"{destination} already exists")
            return True
    except Exception as e:
        print(f"Error downloading {destination}: {e}")
        return False

def download_from_url(url, destination):
    """Download file from direct URL"""
    try:
        if not os.path.exists(destination):
            print(f"Downloading {destination} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {destination}")
            return True
        else:
            print(f"{destination} already exists")
            return True
    except Exception as e:
        print(f"Error downloading {destination}: {e}")
        return False

def setup_models():
    """Download and setup all required models"""
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/anti-spoofing', exist_ok=True)
    
    # Model configurations with working URLs
    models_config = {
        'yolov5s-face.onnx': {
            'drive_id': '1sybYq9GGriXN6sY8YV1-RXMeVqYzhDrV',  # Your YOLO model Google Drive ID
            'path': 'models/yolov5s-face.onnx',
            'required': True
        },
        'AntiSpoofing_bin_1.5_128.onnx': {
            'drive_id': '1nH5G7dAHFE2KlW_H65txc8GDKSB7Zpy4',  # Your Anti-spoof model Google Drive ID
            'path': 'models/anti-spoofing/AntiSpoofing_bin_1.5_128.onnx',
            'required': True
        }
    }
    
    # Alternative working URLs for YOLO (try these first)
    yolo_working_urls = [
        'https://github.com/deepcam-cn/yolov5-face/releases/download/v6.0/yolov5s-face.onnx',
        'https://huggingface.co/arnabdhar/YOLOv5-Face/resolve/main/yolov5s-face.onnx',
        'https://github.com/deepcam-cn/yolov5-face/raw/master/weights/yolov5s-face.onnx'
    ]
    
    # Try downloading YOLO from working URLs first
    yolo_downloaded = False
    print("Attempting to download YOLO Face model...")
    
    for i, url in enumerate(yolo_working_urls):
        print(f"Trying YOLO URL {i+1}/{len(yolo_working_urls)}: {url}")
        try:
            if download_from_url(url, models_config['yolov5s-face.onnx']['path']):
                yolo_downloaded = True
                print(f"✅ YOLO model downloaded successfully from URL {i+1}")
                break
        except Exception as e:
            print(f"❌ Failed URL {i+1}: {e}")
            continue
    
    # If URL download failed, try Google Drive
    if not yolo_downloaded:
        print("All URLs failed. Trying YOLO download from Google Drive...")
        try:
            yolo_downloaded = download_file_from_google_drive(
                models_config['yolov5s-face.onnx']['drive_id'],
                models_config['yolov5s-face.onnx']['path']
            )
            if yolo_downloaded:
                print("✅ YOLO model downloaded successfully from Google Drive")
        except Exception as e:
            print(f"❌ Google Drive download also failed: {e}")
    
    # Download anti-spoofing model from Google Drive
    print("Downloading Anti-spoofing model from Google Drive...")
    antispoof_downloaded = False
    try:
        antispoof_downloaded = download_file_from_google_drive(
            models_config['AntiSpoofing_bin_1.5_128.onnx']['drive_id'],
            models_config['AntiSpoofing_bin_1.5_128.onnx']['path']
        )
        if antispoof_downloaded:
            print("✅ Anti-spoofing model downloaded successfully")
    except Exception as e:
        print(f"❌ Anti-spoofing model download failed: {e}")
    
    # Print final status
    print("\n" + "="*50)
    print("MODEL DOWNLOAD STATUS:")
    print(f"YOLO Face Model: {'✅ Available' if yolo_downloaded else '❌ Failed'}")
    print(f"Anti-Spoof Model: {'✅ Available' if antispoof_downloaded else '❌ Failed'}")
    print("="*50 + "\n")
    
    return yolo_downloaded, antispoof_downloaded


# Initialize models on startup
print("Setting up models...")
yolo_available, antispoof_available = setup_models()

# ---------------- YOLOv5s-face Detection ----------------

def _get_providers():
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

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
    def __init__(self, model_path: str, input_size: int = 640, conf_threshold: float = 0.3, iou_threshold: float = 0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.input_size = int(input_size)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.session = ort.InferenceSession(model_path, providers=_get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        shape = self.session.get_inputs()[0].shape
        if isinstance(shape[2], int):
            self.input_size = int(shape[2])

    @staticmethod
    def _xywh2xyxy(x: np.ndarray) -> np.ndarray:
        y = np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def detect(self, image_bgr: np.ndarray, max_det: int = 20):
        h0, w0 = image_bgr.shape[:2]
        img, ratio, dwdh = _letterbox(image_bgr, new_shape=(self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        preds = self.session.run(self.output_names, {self.input_name: img})[0]
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

# ---------------- Anti-Spoof Model ----------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

class AntiSpoofBinary:
    def __init__(self, model_path: str, input_size: int = 128, rgb: bool = True, normalize: bool = True,
                 mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), live_index: int = 1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.input_size = int(input_size)
        self.rgb = bool(rgb)
        self.normalize = bool(normalize)
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self.live_index = int(live_index)
        self.session = ort.InferenceSession(model_path, providers=_get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(face_bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        if self.normalize:
            img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        return img

    def predict_live_prob(self, face_bgr: np.ndarray) -> float:
        inp = self._preprocess(face_bgr)
        outs = self.session.run(self.output_names, {self.input_name: inp})
        out = outs[0]
        if out.ndim > 1:
            out = np.squeeze(out, axis=0)
        if out.size == 2:
            vec = out.astype(np.float32)
            probs = np.exp(vec - np.max(vec))
            probs = probs / (np.sum(probs) + 1e-9)
            live_prob = float(probs[self.live_index])
        else:
            live_prob = float(_sigmoid(out.astype(np.float32)))
        return max(0.0, min(1.0, live_prob))

# ---------------- Helper Functions ----------------

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

# Model paths
YOLO_FACE_MODEL_PATH = "models/yolov5s-face.onnx"
ANTI_SPOOF_BIN_MODEL_PATH = "models/anti-spoofing/AntiSpoofing_bin_1.5_128.onnx"

# Initialize models (with error handling)
yolo_face = None
anti_spoof_bin = None

try:
    if yolo_available:
        yolo_face = YoloV5FaceDetector(YOLO_FACE_MODEL_PATH, input_size=640, conf_threshold=0.3, iou_threshold=0.45)
        print("YOLO Face model loaded successfully")
    else:
        print("Warning: YOLO Face model not available")
except Exception as e:
    print(f"Error loading YOLO model: {e}")

try:
    if antispoof_available:
        anti_spoof_bin = AntiSpoofBinary(ANTI_SPOOF_BIN_MODEL_PATH, input_size=128, rgb=True, normalize=True, live_index=1)
        print("Anti-spoofing model loaded successfully")
    else:
        print("Warning: Anti-spoofing model not available")
except Exception as e:
    print(f"Error loading anti-spoofing model: {e}")

# ------------------------------------------------------------------------------------------------
# ----------------------------- DeepFace-based Recognition Pipeline -----------------------------

def get_face_features_deepface(image):
    """Extract face features using DeepFace with VGG-Face model"""
    try:
        # Convert BGR to RGB for DeepFace
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face embedding using DeepFace
        embedding = DeepFace.represent(
            img_path=rgb_image, 
            model_name='VGG-Face',
            detector_backend='opencv',
            enforce_detection=False
        )
        
        if isinstance(embedding, list) and len(embedding) > 0:
            return np.array(embedding[0]['embedding'])
        else:
            return np.array(embedding['embedding']) if 'embedding' in embedding else None
            
    except Exception as e:
        print(f"Error in DeepFace feature extraction: {e}")
        return None

def recognize_face_deepface(image, user_id, user_type='student'):
    """Face recognition using DeepFace instead of dlib"""
    global total_attempts, correct_recognitions, false_accepts, false_rejects, inference_times, unauthorized_attempts
    
    try:
        start_time = time.time()
        features = get_face_features_deepface(image)
        
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
        ref_features = get_face_features_deepface(ref_image)
        
        if ref_features is None:
            return False, "No face detected in reference image"

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarity = cosine_similarity([features], [ref_features])[0][0]
        distance = 1 - similarity
        threshold = 0.4
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        total_attempts += 1

        if distance < threshold:
            correct_recognitions += 1
            return True, f"Face recognized (distance={distance:.3f}, similarity={similarity:.3f}, time={inference_time:.2f}s)"
        else:
            unauthorized_attempts += 1
            return False, f"Unauthorized attempt detected (distance={distance:.3f}, similarity={similarity:.3f})"
            
    except Exception as e:
        return False, f"Error in face recognition: {str(e)}"

def recognize_face(image, user_id, user_type='student'):
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

# --------- ALL ROUTES (keeping your existing routes with enhanced error handling) ---------

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
    test_features = get_face_features_deepface(image)
    if test_features is None:
        flash('No face detected. Please try again.', 'danger')
        return redirect(url_for('login_page'))

    for user in users:
        ref_image_bytes = user['face_image']
        ref_image_array = np.frombuffer(ref_image_bytes, np.uint8)
        ref_image = cv2.imdecode(ref_image_array, cv2.IMREAD_COLOR)
        ref_features = get_face_features_deepface(ref_image)
        if ref_features is None:
            continue
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([test_features], [ref_features])[0][0]
        distance = 1 - similarity
        
        if distance < 0.4:
            session['logged_in'] = True
            session['user_type'] = face_role
            session[id_field] = user[id_field]
            session['name'] = user.get('name')
            flash('Face login successful!', 'success')
            return redirect(url_for(dashboard_route))

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
        test_features = get_face_features_deepface(image)
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
                ref_features = get_face_features_deepface(ref_image)
                if ref_features is None:
                    continue
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([test_features], [ref_features])[0][0]
                distance = 1 - similarity
                
                if distance < 0.4:
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

    # Check if required models are available
    if not yolo_face:
        return jsonify({'success': False, 'message': 'Face detection model not available. Please contact admin.'})

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

    image = decode_image(face_image)
    if image is None or image.size == 0:
        return jsonify({'success': False, 'message': 'Invalid image data'})

    h, w = image.shape[:2]
    vis = image.copy()

    detections = yolo_face.detect(image, max_det=20)
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

    # Anti-spoofing (only if model is available)
    live_prob = 1.0  # Default to live if no anti-spoof model
    is_live = True
    
    if anti_spoof_bin:
        live_prob = anti_spoof_bin.predict_live_prob(face_crop)
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

    success, message = recognize_face(image, student_id, user_type='student')
    total_latency_ms = round((time.time() - t0) * 1000.0, 2)

    distance_val = None
    try:
        if "distance=" in message:
            part = message.split("distance=")[1]
            distance_val = float(part.split(",")[0].strip(") "))
    except Exception:
        pass

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
    
    if not yolo_face:
        return jsonify({'success': False, 'message': 'Face detection model not available'})
        
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
        
        # Anti-spoofing (only if model available)
        live_prob = 1.0  # Default to live
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

# --------- METRICS JSON ENDPOINTS ---------
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

# FIXED: Proper port binding for Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
