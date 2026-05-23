"""
app.py — MotionGuard AI Backend (HOG + SVM version)
====================================================
Model: HOG features + LinearSVC  (train bằng colab/train_hog_svm.py)

Endpoints:
  POST /api/upload          - Upload video để phân tích
  POST /api/webcam/start    - Bật webcam realtime
  POST /api/webcam/stop     - Tắt webcam
  GET  /video_feed          - MJPEG stream
  GET  /api/alerts          - Lịch sử cảnh báo
  POST /api/alerts/clear    - Xóa log
  GET  /api/settings        - Đọc cài đặt
  POST /api/settings        - Cập nhật cài đặt
  GET  /api/model/status    - Trạng thái model
  GET  /api/health          - Health check
"""

import os, cv2, time, base64, threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

import sys
sys.path.insert(0, str(Path(__file__).parent))

from person_detector import PersonDetector, FrameResult

# ─── Flask ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
app        = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")
CORS(app)

UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR  = BASE_DIR / "model"
UPLOAD_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─── Global state ─────────────────────────────────────────────────────────────
state_lock    = threading.Lock()
detector      = None
alert_history = deque(maxlen=200)
webcam_active = False
webcam_cap    = None
latest_frame  = None

settings = {
    "conf_threshold"    : 0.0,      # SVM decision score threshold
    "skip_if_no_motion" : True,     # bỏ qua HOG nếu không có chuyển động
    "model_path"        : "model/hog_svm_model.pkl",
}


# ─── Load model ───────────────────────────────────────────────────────────────
def init_detector():
    global detector
    model_path = BASE_DIR / settings["model_path"]
    if not model_path.exists():
        app.logger.error(
            f"❌ Không tìm thấy model: {model_path}\n"
            "   → Chạy colab/train_hog_svm.py trên Google Colab,\n"
            "     download hog_svm_model.pkl và đặt vào thư mục model/"
        )
        return
    try:
        detector = PersonDetector(
            model_path     = str(model_path),
            conf_threshold = settings["conf_threshold"],
        )
        app.logger.info("✅ PersonDetector (HOG+SVM) sẵn sàng")
    except Exception as e:
        app.logger.error(f"❌ Lỗi load model: {e}")


# ─── Webcam thread ────────────────────────────────────────────────────────────
def webcam_thread():
    global webcam_active, webcam_cap, latest_frame
    webcam_cap = cv2.VideoCapture(0)
    if not webcam_cap.isOpened():
        app.logger.error("❌ Không mở được webcam")
        webcam_active = False
        return

    frame_idx = 0
    app.logger.info("📷 Webcam started")

    while webcam_active:
        ret, frame = webcam_cap.read()
        if not ret:
            break

        result = detector.process(
            frame,
            timestamp         = frame_idx / 30.0,
            skip_if_no_motion = settings["skip_if_no_motion"],
        )
        frame_idx += 1

        if result.alert:
            with state_lock:
                alert_history.append({
                    "timestamp"    : datetime.now().isoformat(),
                    "person_count" : result.person_count,
                    "object_count" : 0,
                    "frame_idx"    : result.frame_idx,
                    "fps"          : result.fps,
                    "processing_ms": result.processing_ms,
                })

        _, buf = cv2.imencode('.jpg', result.annotated_frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 85])
        with state_lock:
            latest_frame = buf.tobytes()

        time.sleep(0.033)   # ~30 FPS cap

    webcam_cap.release()
    app.logger.info("📷 Webcam stopped")


def generate_frames():
    while webcam_active:
        with state_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + frame + b'\r\n')
        time.sleep(0.033)


# ═══ ROUTES ═══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


# ─── Upload video ─────────────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload_video():
    if detector is None:
        return jsonify({"error": "Model chưa load — đặt hog_svm_model.pkl vào thư mục model/"}), 500
    if "video" not in request.files:
        return jsonify({"error": "Không có file video"}), 400

    file      = request.files["video"]
    save_path = UPLOAD_DIR / file.filename
    file.save(str(save_path))

    cap = cv2.VideoCapture(str(save_path))
    if not cap.isOpened():
        return jsonify({"error": "Không đọc được video"}), 500

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dùng detector mới (reset MOG2) cho video
    temp_det = PersonDetector(
        model_path     = str(BASE_DIR / settings["model_path"]),
        conf_threshold = settings["conf_threshold"],
    )

    results         = []
    frame_idx       = 0
    alert_cnt       = 0
    sample_interval = max(1, int(fps * 0.5))   # lấy mẫu mỗi 0.5 giây

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            t      = frame_idx / fps
            result = temp_det.process(
                frame,
                timestamp         = t,
                skip_if_no_motion = settings["skip_if_no_motion"],
            )

            if result.alert:
                alert_cnt += 1

            _, buf  = cv2.imencode('.jpg', result.annotated_frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, 72])
            img_b64 = base64.b64encode(buf).decode()

            results.append({
                "frame"        : frame_idx,
                "timestamp"    : round(t, 2),
                "has_motion"   : result.has_motion,
                "has_person"   : result.has_person,
                "person_count" : result.person_count,
                "object_count" : 0,
                "alert"        : result.alert,
                "fps"          : result.fps,
                "processing_ms": result.processing_ms,
                "detections"   : [
                    {
                        "x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2,
                        "class_name": d.class_name,
                        "confidence": round(d.confidence, 3),
                        "is_person" : True,
                    }
                    for d in result.detections
                ],
                "image": f"data:image/jpeg;base64,{img_b64}",
            })

        frame_idx += 1

    cap.release()
    try:
        os.remove(str(save_path))
    except Exception:
        pass

    return jsonify({
        "success"     : True,
        "total_frames": total_f,
        "fps"         : fps,
        "analyzed"    : len(results),
        "alert_count" : alert_cnt,
        "results"     : results,
    })


# ─── Webcam ───────────────────────────────────────────────────────────────────
@app.route("/api/webcam/start", methods=["POST"])
def start_webcam():
    global webcam_active
    if webcam_active:
        return jsonify({"status": "already_running"})
    if detector is None:
        return jsonify({"error": "Model chưa load"}), 500
    detector.reset()
    webcam_active = True
    threading.Thread(target=webcam_thread, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/webcam/stop", methods=["POST"])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ─── Alerts ───────────────────────────────────────────────────────────────────
@app.route("/api/alerts")
def get_alerts():
    with state_lock:
        alerts = list(alert_history)
    return jsonify({"alerts": alerts, "count": len(alerts)})


@app.route("/api/alerts/clear", methods=["POST"])
def clear_alerts():
    with state_lock:
        alert_history.clear()
    return jsonify({"status": "cleared"})


# ─── Settings ─────────────────────────────────────────────────────────────────
@app.route("/api/settings", methods=["GET", "POST"])
def manage_settings():
    global settings, detector
    if request.method == "POST":
        data = request.get_json() or {}
        settings.update(data)
        # Reload nếu đổi conf hoặc model_path
        if "conf_threshold" in data or "model_path" in data:
            init_detector()
        return jsonify({"status": "updated", "settings": settings})
    return jsonify(settings)


# ─── Model status ─────────────────────────────────────────────────────────────
@app.route("/api/model/status")
def model_status():
    info = detector.model_info if detector else {}
    return jsonify({
        "model_loaded"   : detector is not None,
        "model_type"     : "HOG+SVM",
        "classifier"     : "LinearSVC",
        "feature_dim"    : info.get("feature_dim", 3780),
        "conf"           : settings["conf_threshold"],
        "test_acc"       : info.get("test_acc", "N/A"),
        "fps_benchmark"  : info.get("fps_benchmark", "N/A"),
        "flops_per_frame": info.get("flops_per_frame", 0),
        "dataset"        : info.get("dataset", "INRIA"),
        "classes"        : 2,    # person / background
    })


# ─── Health ───────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({
        "status"      : "ok",
        "model_loaded": detector is not None,
        "webcam_on"   : webcam_active,
        "time"        : datetime.now().isoformat(),
    })


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_detector()
    model_ok = "✅" if detector else "❌ (cần train model trước)"

    print("\n" + "="*60)
    print("🚀 MotionGuard AI — HOG + SVM Person Detection")
    print("="*60)
    print(f"  URL      : http://localhost:5000")
    print(f"  Stream   : http://localhost:5000/video_feed")
    print(f"  Model    : {model_ok}")
    if detector:
        info = detector.model_info
        print(f"  Accuracy : {info.get('test_acc','N/A')}")
        print(f"  FPS bench: {info.get('fps_benchmark','N/A')}")
        print(f"  FLOPs/fr : {info.get('flops_per_frame',0)/1e6:.0f} MFLOPs")
    else:
        print(f"\n  ⚠️  Chưa có model! Hướng dẫn:")
        print(f"     1. Chạy colab/train_hog_svm.py trên Google Colab")
        print(f"     2. Download hog_svm_model.pkl từ Drive")
        print(f"     3. Đặt vào thư mục model/")
        print(f"     4. Chạy lại python app.py")
    print("="*60 + "\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
