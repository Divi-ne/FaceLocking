# src/face_lock.py
"""
Face Locking assignment feature
Locks onto one enrolled identity and tracks it across frames.
Detects: move left/right, blink, smile, laugh
Records actions to data/history/<name>_history_<timestamp>.txt

Run: python -m src.face_lock

Keys:
  q     quit
  r     reload database
  + / - adjust threshold
  d     toggle debug overlay
  l     force lock to current best match (testing)
"""

import datetime
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Reuse your alignment function (relative import)
from .haar_5pt import align_face_5pt

# ────────────────────────────────────────────────
# ArcFace Embedder class (this was missing!)
# ────────────────────────────────────────────────

class ArcFaceEmbedderONNX:
    def __init__(self, model_path: str = "models/embedder_arcface.onnx", input_size: Tuple[int, int] = (112, 112)):
        self.in_w, self.in_h = int(input_size[0]), int(input_size[1])
        try:
            self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"\nERROR: Cannot load ArcFace model from {model_path}")
            print(f" → {e}")
            print("Make sure the file exists and is not corrupted (~174 MB).")
            raise
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[ArcFace] Successfully loaded: {model_path}")
        print(f"  Input shape:  {self.sess.get_inputs()[0].shape}")
        print(f"  Output shape: {self.sess.get_outputs()[0].shape}")

    def _preprocess(self, aligned_bgr: np.ndarray) -> np.ndarray:
        img = aligned_bgr
        if img.shape[:2] != (self.in_h, self.in_w):
            img = cv2.resize(img, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        x = np.transpose(rgb, (2, 0, 1))[None, ...]
        return x.astype(np.float32)

    @staticmethod
    def _l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = v.astype(np.float32).ravel()
        norm = np.linalg.norm(v) + eps
        return (v / norm).astype(np.float32)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(aligned_bgr)
        y = self.sess.run([self.out_name], {self.in_name: x})[0]
        emb = y.ravel().astype(np.float32)
        return self._l2_normalize(emb)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

TARGET_NAME = "Divine"  # Change this to match an enrolled name

# Tuning thresholds (you can adjust these later)
BLINK_EAR_THRESHOLD      = 0.23
SMILE_WIDTH_RATIO        = 1.45
LAUGH_OPEN_RATIO         = 0.18
MOVE_CENTER_DELTA_PX     = 18
UNLOCK_AFTER_N_MISSES    = 12
IOU_MIN_FOR_TRACK        = 0.35
EMB_SIM_CONFIRM          = 0.62

HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ────────────────────────────────────────────────
# Data classes (unchanged)
# ────────────────────────────────────────────────

@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray
    left_eye: np.ndarray
    right_eye: np.ndarray
    mouth: np.ndarray

@dataclass
class LockedFace:
    name: str
    prev_center_x: float
    prev_ear: float
    prev_mouth_ratio: float
    prev_mouth_open: float
    prev_box: Tuple[int,int,int,int]
    embedding: np.ndarray
    missing_count: int = 0
    history_file: Optional[Path] = None

@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    similarity: float
    accepted: bool

# ────────────────────────────────────────────────
# Math helpers (unchanged)
# ────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.ravel(), b.ravel()))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)

def box_iou(b1: Tuple[int,int,int,int], b2: Tuple[int,int,int,int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    v1 = np.linalg.norm(eye_pts[1] - eye_pts[5])
    v2 = np.linalg.norm(eye_pts[2] - eye_pts[4])
    h  = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def mouth_metrics(mouth_pts: np.ndarray, eye_dist: float) -> Tuple[float, float]:
    width  = np.linalg.norm(mouth_pts[0] - mouth_pts[1])
    height = np.linalg.norm(mouth_pts[2] - mouth_pts[3])
    return width / (eye_dist + 1e-6), height / width

# ────────────────────────────────────────────────
# Database & Matcher (unchanged)
# ────────────────────────────────────────────────

def load_db(db_path: Path = Path("data/db/face_db.npz")) -> Dict[str, np.ndarray]:
    if not db_path.exists():
        return {}
    data = np.load(str(db_path), allow_pickle=True)
    return {k: data[k].astype(np.float32).ravel() for k in data.files}

class FaceMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.34):
        self.db = db
        self.dist_thresh = dist_thresh
        self.names = sorted(db.keys())
        self.embeddings = np.stack([db[n] for n in self.names], axis=0) if self.names else None

    def reload(self):
        self.db = load_db()
        self.names = sorted(self.db.keys())
        self.embeddings = np.stack([self.db[n] for n in self.names], axis=0) if self.names else None

    def match(self, emb: np.ndarray) -> MatchResult:
        if self.embeddings is None:
            return MatchResult(None, 1.0, 0.0, False)
        sims = np.dot(self.embeddings, emb)
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_dist = 1.0 - best_sim
        ok = best_dist <= self.dist_thresh
        return MatchResult(self.names[best_idx] if ok else None, best_dist, best_sim, ok)

# ────────────────────────────────────────────────
# Face Detector (unchanged)
# ────────────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.idx_5pt = [33, 263, 1, 61, 291]
        self.idx_left_eye  = [362, 385, 387, 263, 373, 380]
        self.idx_right_eye = [33, 160, 158, 133, 153, 144]
        self.idx_mouth     = [61, 291, 13, 14]

    def _get_landmarks(self, lm, indices, W, H):
        return np.array([[lm[i].x * W, lm[i].y * H] for i in indices], dtype=np.float32)

    def detect(self, frame: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(70,70))
        if len(faces) == 0:
            return []

        areas = faces[:,2] * faces[:,3]
        idx = np.argsort(areas)[::-1]
        faces = faces[idx][:max_faces]

        results = []
        for (x,y,w,h) in faces:
            rx1 = max(0, x - int(0.25*w))
            ry1 = max(0, y - int(0.35*h))
            rx2 = min(W, x + w + int(0.25*w))
            ry2 = min(H, y + h + int(0.35*h))
            roi = frame[ry1:ry2, rx1:rx2]
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            res = self.mesh.process(rgb)
            if not res.multi_face_landmarks:
                continue
            lm = res.multi_face_landmarks[0].landmark
            rw, rh = roi.shape[1], roi.shape[0]

            kps5   = self._get_landmarks(lm, self.idx_5pt, rw, rh);     kps5[:,0] += rx1; kps5[:,1] += ry1
            leye   = self._get_landmarks(lm, self.idx_left_eye, rw, rh); leye[:,0] += rx1; leye[:,1] += ry1
            reye   = self._get_landmarks(lm, self.idx_right_eye, rw, rh); reye[:,0] += rx1; reye[:,1] += ry1
            mouth  = self._get_landmarks(lm, self.idx_mouth, rw, rh);   mouth[:,0] += rx1; mouth[:,1] += ry1

            xmin, ymin = kps5.min(axis=0)
            xmax, ymax = kps5.max(axis=0)
            bw, bh = xmax-xmin, ymax-ymin
            x1 = int(max(0, xmin - 0.55*bw))
            y1 = int(max(0, ymin - 0.85*bh))
            x2 = int(min(W-1, xmax + 0.55*bw))
            y2 = int(min(H-1, ymax + 1.15*bh))

            results.append(FaceDet(x1,y1,x2,y2, kps5, leye, reye, mouth))

        return results

# ────────────────────────────────────────────────
# Action logging (unchanged)
# ────────────────────────────────────────────────

def log_action(locked: LockedFace, action: str, desc: str = ""):
    if locked.history_file is None:
        ts = int(time.time() * 100)
        locked.history_file = HISTORY_DIR / f"{locked.name.lower()}_history_{ts}.txt"
        with open(locked.history_file, "w", encoding="utf-8") as f:
            f.write(f"Face locking history for {locked.name}\n")
            f.write("="*40 + "\n")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{now}  {action:<12}  {desc}\n"
    with open(locked.history_file, "a", encoding="utf-8") as f:
        f.write(line)
    print(line.strip())

# ────────────────────────────────────────────────
# Main function (unchanged except for using the class)
# ────────────────────────────────────────────────

def main():
    if TARGET_NAME == "":
        target = input("Enter target name to lock (must be enrolled): ").strip()
    else:
        target = TARGET_NAME

    db_path = Path("data/db/face_db.npz")
    if not db_path.exists():
        print("No database found. Run enroll.py first.")
        return

    detector = FaceDetector()
    embedder = ArcFaceEmbedderONNX()           # ← now defined!
    db = load_db(db_path)
    matcher = FaceMatcher(db, dist_thresh=0.34)

    if target not in db:
        print(f"Target '{target}' not found in database.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print(f"Face Locking for '{target}'. Controls: q=quit, r=reload, +/- thresh, d=debug, l=force lock")

    locked: Optional[LockedFace] = None
    show_debug = False
    t0 = time.time()
    frame_count = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        vis = frame.copy()
        faces = detector.detect(frame)

        frame_count += 1
        dt = time.time() - t0
        if dt > 1.0:
            fps = frame_count / dt
            frame_count = 0
            t0 = time.time()

        current_face = None

        if locked is None:
            for f in faces:
                aligned, _ = align_face_5pt(frame, f.kps, (112,112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                if mr.name == target:
                    eye_dist = np.linalg.norm(f.kps[0] - f.kps[1])
                    _, open_ratio = mouth_metrics(f.mouth, eye_dist)
                    width_ratio, _ = mouth_metrics(f.mouth, eye_dist)
                    avg_ear = (eye_aspect_ratio(f.left_eye) + eye_aspect_ratio(f.right_eye)) / 2

                    locked = LockedFace(
                        target,
                        (f.x1 + f.x2)/2,
                        avg_ear,
                        width_ratio,
                        open_ratio,
                        (f.x1,f.y1,f.x2,f.y2),
                        emb
                    )
                    log_action(locked, "LOCKED", "Target identity recognized")
                    current_face = f
                    break
        else:
            best_iou = 0.0
            best_face = None
            for f in faces:
                iou = box_iou(locked.prev_box, (f.x1,f.y1,f.x2,f.y2))
                if iou > best_iou:
                    best_iou = iou
                    best_face = f

            if best_face and best_iou > IOU_MIN_FOR_TRACK:
                aligned, _ = align_face_5pt(frame, best_face.kps, (112,112))
                emb = embedder.embed(aligned)
                sim = cosine_similarity(locked.embedding, emb)
                if sim > EMB_SIM_CONFIRM:
                    current_face = best_face
                    locked.missing_count = 0
                    locked.embedding = 0.7 * locked.embedding + 0.3 * emb
                else:
                    locked.missing_count += 1
            else:
                locked.missing_count += 1

            if locked.missing_count >= UNLOCK_AFTER_N_MISSES:
                log_action(locked, "UNLOCKED", f"Lost for {locked.missing_count} frames")
                if locked.history_file:
                    print(f"History saved → {locked.history_file}")
                locked = None

        if locked and current_face:
            cx = (current_face.x1 + current_face.x2) / 2.0
            delta_x = cx - locked.prev_center_x

            if abs(delta_x) > MOVE_CENTER_DELTA_PX:
                dir_str = "RIGHT" if delta_x > 0 else "LEFT"
                log_action(locked, f"MOVE_{dir_str}", f"Δx = {delta_x:.1f}px")
            locked.prev_center_x = cx

            ear_l = eye_aspect_ratio(current_face.left_eye)
            ear_r = eye_aspect_ratio(current_face.right_eye)
            avg_ear = (ear_l + ear_r) / 2
            if avg_ear < BLINK_EAR_THRESHOLD and locked.prev_ear >= BLINK_EAR_THRESHOLD:
                log_action(locked, "BLINK")
            locked.prev_ear = avg_ear

            eye_dist = np.linalg.norm(current_face.kps[0] - current_face.kps[1])
            w_ratio, open_ratio = mouth_metrics(current_face.mouth, eye_dist)
            if w_ratio > SMILE_WIDTH_RATIO and locked.prev_mouth_ratio <= SMILE_WIDTH_RATIO:
                if open_ratio > LAUGH_OPEN_RATIO:
                    log_action(locked, "LAUGH")
                else:
                    log_action(locked, "SMILE")
            locked.prev_mouth_ratio = w_ratio
            locked.prev_mouth_open = open_ratio

            locked.prev_box = (current_face.x1, current_face.y1, current_face.x2, current_face.y2)

            cv2.rectangle(vis, (current_face.x1, current_face.y1),
                          (current_face.x2, current_face.y2), (255,0,0), 3)
            cv2.putText(vis, f"LOCKED: {locked.name}", (current_face.x1, current_face.y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        for f in faces:
            if f is not current_face:
                cv2.rectangle(vis, (f.x1,f.y1), (f.x2,f.y2), (0,255,0), 1)

        status = f"Target: {target}   Locked: {'YES' if locked else 'NO'}   Thr: {matcher.dist_thresh:.2f}"
        if fps > 0:
            status += f"   FPS: {fps:.1f}"
        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Face Lock", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            matcher.reload()
            print("Database reloaded")
        elif key in (ord('+'), ord('=')):
            matcher.dist_thresh = min(1.2, matcher.dist_thresh + 0.01)
            print(f"Threshold → {matcher.dist_thresh:.3f}")
        elif key == ord('-'):
            matcher.dist_thresh = max(0.05, matcher.dist_thresh - 0.01)
            print(f"Threshold → {matcher.dist_thresh:.3f}")
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('l') and not locked and faces:
            f = faces[0]
            aligned, _ = align_face_5pt(frame, f.kps, (112,112))
            emb = embedder.embed(aligned)
            mr = matcher.match(emb)
            if mr.accepted and mr.name == target:
                eye_dist = np.linalg.norm(f.kps[0] - f.kps[1])
                w_ratio, o_ratio = mouth_metrics(f.mouth, eye_dist)
                avg_ear = (eye_aspect_ratio(f.left_eye) + eye_aspect_ratio(f.right_eye)) / 2
                locked = LockedFace(
                    target,
                    (f.x1 + f.x2)/2,
                    avg_ear,
                    w_ratio,
                    o_ratio,
                    (f.x1,f.y1,f.x2,f.y2),
                    emb
                )
                log_action(locked, "LOCKED", "Forced lock (key 'l')")

    if locked and locked.history_file:
        print(f"\nFinal history saved to:\n{locked.history_file}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()