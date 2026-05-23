"""
person_detector.py — HOG + SVM Person Detector
================================================
Pipeline:
  Frame → Resize 640×480
        → MOG2 background subtraction (có chuyển động?)
        → HOG + LinearSVC sliding window  (cv2.detectMultiScale)
        → NMS (gộp box trùng)
        → Người → 🔴 Bounding box đỏ + Cảnh báo
        → Không  → ✅ Bình thường

Yêu cầu:
  model/hog_svm_model.pkl  (train bằng colab/train_hog_svm.py)
"""

import cv2
import numpy as np
import joblib
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


COLOR_PERSON = (0, 0, 255)    # BGR đỏ


# ─── Data classes (giữ nguyên interface với app.py) ───────────────────────────

@dataclass
class Detection:
    x1: int; y1: int; x2: int; y2: int
    class_id: int = 0
    class_name: str = 'person'
    confidence: float = 0.0
    is_person: bool = True

    @property
    def width(self):  return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1


@dataclass
class FrameResult:
    frame_idx: int;        timestamp: float
    has_motion: bool;      has_person: bool
    person_count: int;     object_count: int
    detections: List[Detection] = field(default_factory=list)
    alert: bool = False
    annotated_frame: Optional[np.ndarray] = None
    fps: float = 0.0
    processing_ms: float = 0.0


# ─── PersonDetector ───────────────────────────────────────────────────────────

class PersonDetector:
    """
    Phát hiện người bằng HOG features + LinearSVC.
    Dùng cv2.HOGDescriptor.setSVMDetector() để chạy nhanh (C++ backend).
    """

    # HOG window — phải khớp với config khi train
    WIN_W = 64
    WIN_H = 128

    def __init__(self, model_path: str, conf_threshold: float = 0.0):
        """
        Args:
            model_path     : đường dẫn tới hog_svm_model.pkl
            conf_threshold : ngưỡng SVM decision score (0 = đường biên quyết định)
                             tăng lên để giảm false positive
        """
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.frame_count    = 0
        self._times: list   = []
        self._model_info    = {}

        # Background subtractor MOG2
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # HOG descriptor (tham số khớp với lúc train)
        self.hog = cv2.HOGDescriptor(
            (self.WIN_W, self.WIN_H),   # winSize
            (16, 16),                   # blockSize
            (8,  8),                    # blockStride
            (8,  8),                    # cellSize
            9                           # nbins  → feature dim = 3780
        )

        self._load_model()

    # ── Load model ────────────────────────────────────────────────────────────

    def _load_model(self):
        path = Path(self.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"\n{'='*55}\n"
                f"❌ Không tìm thấy model: {path}\n\n"
                f"   Hướng dẫn:\n"
                f"   1. Mở Google Colab\n"
                f"   2. Upload & chạy colab/train_hog_svm.py\n"
                f"   3. Download model từ Drive về máy\n"
                f"   4. Đặt vào: model/hog_svm_model.pkl\n"
                f"{'='*55}\n"
            )

        data = joblib.load(str(path))

        if not isinstance(data, dict) or 'svm_weights' not in data:
            raise ValueError(
                "❌ File model không đúng format.\n"
                "   Hãy re-train bằng colab/train_hog_svm.py mới nhất."
            )

        # Kiểm tra feature dimension
        expected_size = int(self.hog.getDescriptorSize()) + 1   # 3781
        weights = np.array(data['svm_weights'], dtype=np.float64)
        if len(weights) != expected_size:
            raise ValueError(
                f"❌ SVM weights size {len(weights)} ≠ expected {expected_size}.\n"
                f"   HOG params không khớp giữa train và detect."
            )

        # Cài SVM weights vào HOG descriptor (C++ backend → nhanh)
        self.hog.setSVMDetector(weights)

        # Lưu thông tin để hiển thị
        info = data.get('training_info', {})
        self._model_info = {
            'test_acc'       : info.get('test_acc_pct', 'N/A'),
            'feature_dim'    : info.get('feature_dim', 3780),
            'fps_benchmark'  : info.get('fps_benchmark', 'N/A'),
            'flops_per_frame': info.get('flops_per_frame', 0),
            'dataset'        : info.get('dataset', 'INRIA'),
            'n_train_total'  : info.get('total_train', 0),
        }

        print(
            f"✅ HOG+SVM model loaded\n"
            f"   Dataset   : {self._model_info['dataset']}\n"
            f"   Test acc  : {self._model_info['test_acc']}\n"
            f"   FPS bench : {self._model_info['fps_benchmark']} FPS\n"
            f"   Features  : {self._model_info['feature_dim']}\n"
        )

    # ── Motion detection (MOG2) ───────────────────────────────────────────────

    def _detect_motion(self, frame) -> tuple:
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        mask    = self.bg_sub.apply(blurred, learningRate=0.01)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask    = cv2.erode(mask,  self.kernel, iterations=1)
        mask    = cv2.dilate(mask, self.kernel, iterations=3)
        has_motion = cv2.countNonZero(mask) > 1500
        return has_motion, mask

    # ── HOG + SVM detection ───────────────────────────────────────────────────

    def _detect_persons(self, frame) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detectMultiScale dùng C++ backend → nhanh hơn Python loop nhiều lần
        result = self.hog.detectMultiScale(
            gray,
            hitThreshold   = self.conf_threshold,  # ngưỡng SVM score
            winStride      = (8, 8),
            padding        = (4, 4),
            scale          = 1.05,
            groupThreshold = 2,                     # gộp box: ≥2 hit mới giữ
        )

        # OpenCV trả về (rects, weights) hoặc empty tuple khi không có gì
        if not isinstance(result, tuple) or len(result) != 2:
            return []
        boxes_raw, scores = result

        if not isinstance(boxes_raw, np.ndarray) or len(boxes_raw) == 0:
            return []

        detections = []
        for i in range(len(boxes_raw)):
            x, y, w, h = boxes_raw[i]
            score = float(scores[i]) if (
                isinstance(scores, np.ndarray) and len(scores) > i
            ) else 0.0
            detections.append(Detection(
                x1=int(x), y1=int(y), x2=int(x+w), y2=int(y+h),
                class_name='person',
                confidence=round(score, 3),
                is_person=True
            ))
        return detections

    # ── Annotate frame ────────────────────────────────────────────────────────

    def _draw(self, frame, detections, has_motion, alert, fps) -> np.ndarray:
        out  = frame.copy()
        h, w = out.shape[:2]

        # Vẽ bounding boxes
        for det in detections:
            cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), COLOR_PERSON, 3)
            lbl = f"NGUOI  {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(out,
                (det.x1, det.y1 - th - 10),
                (det.x1 + tw + 6, det.y1),
                COLOR_PERSON, -1)
            cv2.putText(out, lbl,
                (det.x1 + 3, det.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        # Header bar
        bar_color = (0, 0, 160) if alert else (18, 18, 18)
        cv2.rectangle(out, (0, 0), (w, 44), bar_color, -1)

        if alert:
            n_p = len(detections)
            txt = f"CANH BAO: {n_p} NGUOI phat hien"
            tc  = (0, 220, 255)
            # Viền đỏ toàn frame
            cv2.rectangle(out, (2, 2), (w-2, h-2), COLOR_PERSON, 4)
        elif has_motion:
            txt = "Chuyen dong: khong co nguoi"
            tc  = (0, 210, 110)
        else:
            txt = "Khong co chuyen dong"
            tc  = (140, 140, 140)

        cv2.putText(out, txt,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, tc, 2)
        cv2.putText(out, f"HOG+SVM  FPS:{fps:.0f}",
                    (w - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 200, 80), 2)

        return out

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(self, frame, timestamp: float = 0.0,
                skip_if_no_motion: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        self.frame_count += 1

        # Resize về 640×480 để ổn định tốc độ
        frame_rs = cv2.resize(frame, (640, 480))

        # 1) Motion detection
        has_motion, _ = self._detect_motion(frame_rs)

        # 2) HOG+SVM chỉ chạy khi có chuyển động (hoặc bắt buộc)
        detections: List[Detection] = []
        if has_motion or not skip_if_no_motion:
            detections = self._detect_persons(frame_rs)

        alert = len(detections) > 0

        # 3) FPS rolling average (30 frames)
        elapsed = time.perf_counter() - t0
        self._times.append(elapsed)
        if len(self._times) > 30:
            self._times.pop(0)
        fps = 1.0 / (sum(self._times) / len(self._times))

        # 4) Annotate
        annotated = self._draw(frame_rs, detections, has_motion, alert, fps)

        return FrameResult(
            frame_idx      = self.frame_count,
            timestamp      = timestamp,
            has_motion     = has_motion,
            has_person     = alert,
            person_count   = len(detections),
            object_count   = 0,          # HOG+SVM chỉ detect người
            detections     = detections,
            alert          = alert,
            annotated_frame= annotated,
            fps            = round(fps, 1),
            processing_ms  = round(elapsed * 1000, 1),
        )

    # ── Reset (dùng khi bắt đầu video/webcam mới) ────────────────────────────

    def reset(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)
        self.frame_count = 0
        self._times      = []

    @property
    def model_info(self) -> dict:
        return self._model_info
