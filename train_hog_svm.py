# ============================================================
# MotionGuard AI — HOG + SVM Training Script
# Dataset : INRIA Person Dataset (via Kaggle)
# Chạy trên Google Colab — mỗi block là 1 cell riêng
# ============================================================

# ===========================================================
# CELL 1 — MOUNT GOOGLE DRIVE + SETUP
# ===========================================================
from google.colab import drive
drive.mount('/content/drive')

import os
MODEL_SAVE_DIR = '/content/drive/MyDrive/MotionGuardAI/models'
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
print(f"✅ Drive mounted\n📁 Model sẽ lưu tại: {MODEL_SAVE_DIR}")


# ===========================================================
# CELL 2 — CÀI PACKAGES
# ===========================================================
# !pip install kagglehub opencv-python-headless scikit-learn numpy joblib tqdm matplotlib seaborn -q

import cv2, os, time, warnings
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
warnings.filterwarnings('ignore')

print("✅ Packages OK — OpenCV", cv2.__version__)


# ===========================================================
# CELL 3 — KAGGLE AUTHENTICATION
# ===========================================================
# Bạn cần API key từ https://www.kaggle.com/settings → API → Create New Token
# File tải về tên là kaggle.json
#
# CÁCH 1 (đơn giản): Upload kaggle.json thủ công
# --------------------------------------------------
from google.colab import files
print("📎 Hãy upload file kaggle.json của bạn:")
uploaded = files.upload()   # chọn kaggle.json

import os, shutil
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
shutil.copy("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
print("✅ Kaggle API key đã được cài đặt")

# CÁCH 2 (nếu dùng Colab Secrets — bỏ comment nếu muốn dùng):
# --------------------------------------------------
# from google.colab import userdata
# import os
# os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
# os.environ['KAGGLE_KEY']      = userdata.get('KAGGLE_KEY')
# print("✅ Kaggle credentials từ Colab Secrets")


# ===========================================================
# CELL 4 — TẢI INRIA DATASET TỪ KAGGLE
# ===========================================================
import kagglehub

print("📥 Đang tải INRIA Person Dataset từ Kaggle...")
DATASET_PATH = kagglehub.dataset_download("jcoral02/inriaperson")
print(f"✅ Dataset tải về tại: {DATASET_PATH}")

# Xem cấu trúc thư mục sau khi tải
print("\n📂 Cấu trúc thư mục:")
for root, dirs, files_list in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = '  ' * level
    folder_name = os.path.basename(root)
    n_files = len(files_list)
    if level <= 3:
        print(f"{indent}📁 {folder_name}/  ({n_files} files)")


# ===========================================================
# CELL 5 — TÌM ĐƯỜNG DẪN ẢNH TRONG DATASET
# ===========================================================
def find_image_paths(base_dir):
    """
    Tìm ảnh positive (người) và negative (nền) trong INRIA dataset.
    Hỗ trợ nhiều cấu trúc thư mục khác nhau từ các nguồn Kaggle.
    """
    exts = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    base = Path(base_dir)

    def search(keyword):
        found = []
        for ext in exts:
            found += list(base.rglob(f'*{keyword}*/{ext}'))
            found += list(base.rglob(f'*{keyword.lower()}*/{ext}'))
            found += list(base.rglob(f'*{keyword.upper()}*/{ext}'))
        return list(set(str(p) for p in found))

    # Tìm theo pattern thư mục pos/neg
    pos_all = search('pos')
    neg_all = search('neg')

    # Tách train / test theo đường dẫn
    pos_train = [p for p in pos_all if any(k in p.lower() for k in ['train', 'training'])]
    pos_test  = [p for p in pos_all if any(k in p.lower() for k in ['test'])]
    neg_train = [p for p in neg_all if any(k in p.lower() for k in ['train', 'training'])]
    neg_test  = [p for p in neg_all if any(k in p.lower() for k in ['test'])]

    # Nếu không có train/test split rõ → dùng toàn bộ rồi tự split
    if not pos_train and pos_all:
        print("⚠️  Không thấy thư mục train/test riêng → sẽ tự split 80/20")
        pos_train, pos_test = pos_all[:int(len(pos_all)*0.8)], pos_all[int(len(pos_all)*0.8):]
    if not neg_train and neg_all:
        neg_train, neg_test = neg_all[:int(len(neg_all)*0.8)], neg_all[int(len(neg_all)*0.8):]

    return pos_train, neg_train, pos_test, neg_test


pos_train_paths, neg_train_paths, pos_test_paths, neg_test_paths = find_image_paths(DATASET_PATH)

print("📊 Dataset Summary:")
print(f"  Train → Positive (người) : {len(pos_train_paths):>5} ảnh")
print(f"  Train → Negative (nền)   : {len(neg_train_paths):>5} ảnh")
print(f"  Test  → Positive         : {len(pos_test_paths):>5} ảnh")
print(f"  Test  → Negative         : {len(neg_test_paths):>5} ảnh")

# Kiểm tra có đủ ảnh không
assert len(pos_train_paths) > 0, "❌ Không tìm thấy ảnh positive! Kiểm tra lại cấu trúc dataset."
assert len(neg_train_paths) > 0, "❌ Không tìm thấy ảnh negative!"

# Hiển thị vài ảnh mẫu để kiểm tra
fig, axes = plt.subplots(2, 5, figsize=(13, 5))
for i, ax in enumerate(axes[0]):
    img = cv2.cvtColor(cv2.imread(pos_train_paths[i]), cv2.COLOR_BGR2RGB)
    ax.imshow(img); ax.set_title(f'Person {i+1}', fontsize=9); ax.axis('off')
for i, ax in enumerate(axes[1]):
    img = cv2.imread(neg_train_paths[i])
    h, w = img.shape[:2]
    crop = img[:min(160,h), :min(96,w)]
    ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Background {i+1}', fontsize=9); ax.axis('off')
plt.suptitle('Dataset Samples — INRIA Person Dataset (Kaggle: jcoral02/inriaperson)',
             fontsize=11, fontweight='bold')
plt.tight_layout(); plt.show()


# ===========================================================
# CELL 6 — CẤU HÌNH HOG + TÍNH FEATURE DIMENSION
# ===========================================================
# Tham số HOG chuẩn theo Dalal & Triggs (CVPR 2005)
WIN_W,   WIN_H   = 64, 128   # Detection window (width × height)
CELL_W,  CELL_H  = 8,  8    # Pixels per cell
BLOCK_W, BLOCK_H = 16, 16   # Pixels per block  (2×2 cells)
BSTRIDE_W        = 8        # Block stride (50% overlap)
BSTRIDE_H        = 8
N_BINS           = 9        # Orientation histogram bins (0°–180°)

hog = cv2.HOGDescriptor(
    _winSize     = (WIN_W, WIN_H),
    _blockSize   = (BLOCK_W, BLOCK_H),
    _blockStride = (BSTRIDE_W, BSTRIDE_H),
    _cellSize    = (CELL_W, CELL_H),
    _nbins       = N_BINS
)
FEATURE_DIM = int(hog.getDescriptorSize())   # = 3780

# Tính thủ công để kiểm chứng
n_cells_x       = WIN_W // CELL_W                              # 8
n_cells_y       = WIN_H // CELL_H                              # 16
n_blocks_x      = (WIN_W  - BLOCK_W) // BSTRIDE_W + 1         # 7
n_blocks_y      = (WIN_H  - BLOCK_H) // BSTRIDE_H + 1         # 15
n_blocks        = n_blocks_x * n_blocks_y                      # 105
cells_per_block = (BLOCK_W // CELL_W) * (BLOCK_H // CELL_H)   # 4

print("═"*55)
print("⚙️  CẤU HÌNH HOG DESCRIPTOR")
print("═"*55)
print(f"  Detection window  : {WIN_W} × {WIN_H} px")
print(f"  Cell size         : {CELL_W} × {CELL_H} px")
print(f"  Block size        : {BLOCK_W} × {BLOCK_H} px  ({BLOCK_W//CELL_W}×{BLOCK_H//CELL_H} cells)")
print(f"  Block stride      : {BSTRIDE_W} × {BSTRIDE_H} px  (overlap 50%)")
print(f"  Orientation bins  : {N_BINS}")
print()
print(f"  Cells in window   : {n_cells_x} × {n_cells_y} = {n_cells_x*n_cells_y}")
print(f"  Blocks in window  : {n_blocks_x} × {n_blocks_y} = {n_blocks}")
print(f"  Feats per block   : {cells_per_block} cells × {N_BINS} bins = {cells_per_block*N_BINS}")
print(f"  ✅ TOTAL FEATURES  : {n_blocks} × {cells_per_block*N_BINS} = {FEATURE_DIM}")
assert FEATURE_DIM == 3780, f"Expected 3780, got {FEATURE_DIM}"


# ===========================================================
# CELL 7 — TRÍCH XUẤT HOG FEATURES
# ===========================================================
def extract_hog(img_or_path):
    """Đọc ảnh (hoặc dùng array), resize về WIN_W×WIN_H, trả về HOG features."""
    img = cv2.imread(img_or_path) if isinstance(img_or_path, str) else img_or_path
    if img is None:
        return None
    resized = cv2.resize(img, (WIN_W, WIN_H))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    return hog.compute(gray).flatten().astype(np.float32)


def extract_positive_features(paths, desc="Positive"):
    feats = []
    for p in tqdm(paths, desc=f"  HOG {desc}"):
        f = extract_hog(p)
        if f is not None:
            feats.append(f)
    return np.array(feats, dtype=np.float32)


def extract_negative_features(paths, patches_per_img=12, seed=42, desc="Negative"):
    """Lấy ngẫu nhiên nhiều patches từ mỗi ảnh nền."""
    rng   = np.random.default_rng(seed)
    feats = []
    for p in tqdm(paths, desc=f"  HOG {desc}"):
        img = cv2.imread(str(p))
        if img is None: continue
        h, w = img.shape[:2]
        if h < WIN_H or w < WIN_W: continue
        for _ in range(patches_per_img):
            y = rng.integers(0, h - WIN_H)
            x = rng.integers(0, w - WIN_W)
            f = extract_hog(img[y:y+WIN_H, x:x+WIN_W])
            if f is not None:
                feats.append(f)
    return np.array(feats, dtype=np.float32)


print("🔄 Trích xuất HOG features từ dataset...")
X_pos_tr = extract_positive_features(pos_train_paths, "Train-Positive")
X_neg_tr = extract_negative_features(neg_train_paths, patches_per_img=10, desc="Train-Negative")
X_pos_te = extract_positive_features(pos_test_paths,  "Test-Positive")
X_neg_te = extract_negative_features(neg_test_paths,  patches_per_img=5,  desc="Test-Negative")

X_train = np.vstack([X_pos_tr, X_neg_tr])
y_train = np.hstack([np.ones(len(X_pos_tr)), np.zeros(len(X_neg_tr))]).astype(int)
X_test  = np.vstack([X_pos_te, X_neg_te])
y_test  = np.hstack([np.ones(len(X_pos_te)), np.zeros(len(X_neg_te))]).astype(int)

print(f"\n📊 Sau feature extraction:")
print(f"  Train : {len(X_train):,}  ({len(X_pos_tr)} người  +  {len(X_neg_tr)} nền)")
print(f"  Test  : {len(X_test):,}  ({len(X_pos_te)} người  +  {len(X_neg_te)} nền)")
print(f"  Mỗi sample : {FEATURE_DIM} features")
print(f"  RAM dùng   : {X_train.nbytes/1e6:.1f} MB (train) + {X_test.nbytes/1e6:.1f} MB (test)")


# ===========================================================
# CELL 8 — TRAIN LinearSVC (ROUND 1)
# ===========================================================
print("🏋️  Training LinearSVC (round 1 / 2)...")
t0  = time.time()
clf = LinearSVC(C=0.01, max_iter=5000, random_state=42, dual=True)
clf.fit(X_train, y_train)
print(f"⏱  Xong trong {time.time()-t0:.1f}s")

y_pred_r1 = clf.predict(X_test)
print(f"\n  Train accuracy : {clf.score(X_train, y_train):.4f}")
print(f"  Test accuracy  : {accuracy_score(y_test, y_pred_r1):.4f}")
print()
print(classification_report(y_test, y_pred_r1,
                             target_names=['Nền (Background)', 'Người (Person)']))


# ===========================================================
# CELL 9 — HARD NEGATIVE MINING + RETRAIN (ROUND 2)
# ===========================================================
# Tìm ảnh nền bị dự đoán nhầm là người → thêm vào train
# để model cứng hơn với false positive.

def hard_negative_mining(clf, neg_paths, patches=60, max_fp_per_img=5, seed=99):
    rng    = np.random.default_rng(seed)
    h_negs = []
    for p in tqdm(neg_paths, desc="  Mining hard negatives"):
        img = cv2.imread(str(p))
        if img is None: continue
        h, w = img.shape[:2]
        if h < WIN_H or w < WIN_W: continue
        fp_found = 0
        for _ in range(patches):
            y = rng.integers(0, h - WIN_H)
            x = rng.integers(0, w - WIN_W)
            feat = extract_hog(img[y:y+WIN_H, x:x+WIN_W])
            if feat is not None and clf.predict([feat])[0] == 1:
                h_negs.append(feat)
                fp_found += 1
                if fp_found >= max_fp_per_img:
                    break
    return np.array(h_negs, np.float32) if h_negs else np.zeros((0, FEATURE_DIM), np.float32)


print("⛏️  Hard Negative Mining...")
X_hard = hard_negative_mining(clf, neg_train_paths)
print(f"  → Tìm thấy {len(X_hard)} hard negatives")

if len(X_hard) > 0:
    X_tr2 = np.vstack([X_train, X_hard])
    y_tr2 = np.hstack([y_train, np.zeros(len(X_hard), int)])
    print(f"\n🏋️  Retrain với hard negatives (round 2 / 2)...")
    print(f"  Tổng samples: {len(X_tr2):,}")
    t0        = time.time()
    clf_final = LinearSVC(C=0.01, max_iter=5000, random_state=42, dual=True)
    clf_final.fit(X_tr2, y_tr2)
    print(f"⏱  Xong trong {time.time()-t0:.1f}s")
else:
    clf_final = clf
    X_tr2     = X_train
    print("  Không có hard neg → giữ nguyên round 1")

y_pred = clf_final.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n📊 KẾT QUẢ CUỐI (sau hard negative mining):")
print(f"  Test accuracy : {acc:.4f}  ({acc:.2%})")
print()
print(classification_report(y_test, y_pred, target_names=['Nền', 'Người']))


# ===========================================================
# CELL 10 — VISUALIZATION
# ===========================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', linewidths=.5,
            xticklabels=['Nền', 'Người'],
            yticklabels=['Nền', 'Người'], ax=axes[0])
axes[0].set_title('Confusion Matrix\n(sau Hard Negative Mining)', fontweight='bold')
axes[0].set_ylabel('Thực tế'); axes[0].set_xlabel('Dự đoán')

# SVM decision score distribution
s_pos = clf_final.decision_function(X_pos_te)
s_neg = clf_final.decision_function(X_neg_te)
axes[1].hist(s_neg, bins=60, alpha=0.65, label='Nền (Background)', color='steelblue')
axes[1].hist(s_pos, bins=60, alpha=0.65, label='Người (Person)',   color='crimson')
axes[1].axvline(0, color='black', lw=2, ls='--', label='Decision boundary = 0')
axes[1].set_xlabel('SVM Decision Score'); axes[1].set_ylabel('Count')
axes[1].set_title('Phân phối SVM Score', fontweight='bold')
axes[1].legend()

plt.suptitle('HOG + LinearSVC — Kết quả Training', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{MODEL_SAVE_DIR}/training_results.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"✅ Lưu biểu đồ → {MODEL_SAVE_DIR}/training_results.png")


# ===========================================================
# CELL 11 — PHÂN TÍCH ĐỘ PHỨC TẠP TÍNH TOÁN
# ===========================================================
FRAME_W      = 640
FRAME_H      = 480
STRIDE       = 8
SCALE_FACTOR = 1.05
PIXEL_COUNT  = WIN_W * WIN_H   # 8192

print("═"*62)
print("📐 PHÂN TÍCH ĐỘ PHỨC TẠP TÍNH TOÁN HOG + SVM")
print("═"*62)

# ── [A] HOG FLOPs trên 1 window ──────────────────────────────
grad_flops  = PIXEL_COUNT * 8
hist_flops  = PIXEL_COUNT * 6
norm_flops  = n_blocks * (cells_per_block * N_BINS * 3 + 5)
hog_flops   = grad_flops + hist_flops + norm_flops
svm_flops   = FEATURE_DIM * 2 + 1
total_per_w = hog_flops + svm_flops

print(f"\n[A] HOG — 1 window ({WIN_W}×{WIN_H} px = {PIXEL_COUNT:,} pixels)")
print(f"    Gradient (Sobel)      : {grad_flops:>10,}  FLOPs")
print(f"    Histogram voting      : {hist_flops:>10,}  FLOPs")
print(f"    Block normalization   : {norm_flops:>10,}  FLOPs  ({n_blocks} blocks)")
print(f"    HOG subtotal          : {hog_flops:>10,}  FLOPs")
print(f"\n[B] SVM — 1 window")
print(f"    Dot product (dim=3780): {svm_flops:>10,}  FLOPs")
print(f"\n    → Tổng 1 window       : {total_per_w:>10,}  FLOPs  ≈ {total_per_w/1e3:.1f}K")

# ── [C] Sliding window (multi-scale) ─────────────────────────
print(f"\n[C] Sliding Window — frame {FRAME_W}×{FRAME_H}, stride={STRIDE}")
total_windows = 0
scale = 1.0
for i in range(30):
    fw = int(FRAME_W / scale)
    fh = int(FRAME_H / scale)
    if fw < WIN_W or fh < WIN_H: break
    nx = (fw - WIN_W) // STRIDE + 1
    ny = (fh - WIN_H) // STRIDE + 1
    wc = nx * ny
    total_windows += wc
    print(f"    Scale {scale:.2f}  {fw:>4}×{fh:<4}  →  {nx}×{ny:>2}  = {wc:>6,} windows")
    scale *= SCALE_FACTOR
print(f"    ──────────────────────────────────────────────────")
print(f"    Tổng windows/frame    : {total_windows:>10,}")

# ── [D] Tổng 1 frame ─────────────────────────────────────────
total_flops = total_windows * total_per_w
print(f"\n[D] TỔNG 1 FRAME")
print(f"    Windows               : {total_windows:>10,}")
print(f"    FLOPs/window          : {total_per_w:>10,}")
print(f"    ──────────────────────────────────────────────────")
print(f"    TỔNG FLOPs/frame      : {total_flops:>10,}")
print(f"    ≈ {total_flops/1e6:.1f} MFLOPs/frame")
print(f"    ≈ {total_flops/1e9:.4f} GFLOPs/frame")


# ===========================================================
# CELL 12 — BENCHMARK FPS (THỰC NGHIỆM)
# ===========================================================
print("\n[E] BENCHMARK FPS")
print(f"    Đang chạy 40 frames để đo...")

# Build detector tạm thời với SVM weights
_w  = np.append(clf_final.coef_[0], -clf_final.intercept_[0]).astype(np.float64)
_h  = cv2.HOGDescriptor((WIN_W,WIN_H),(16,16),(8,8),(8,8),9)
_h.setSVMDetector(_w)
_bg = cv2.createBackgroundSubtractorMOG2(500, 50, True)
_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
rng = np.random.default_rng(7)
base_frame = rng.integers(60, 200, (FRAME_H, FRAME_W, 3), dtype=np.uint8)

WARMUP, BENCH = 5, 40
t_mog_l, t_hog_l, t_tot_l = [], [], []

for i in range(WARMUP + BENCH):
    frame       = base_frame.copy()
    rx = rng.integers(0, FRAME_W-80); ry = rng.integers(0, FRAME_H-80)
    frame[ry:ry+80, rx:rx+80] = rng.integers(0, 255, (80,80,3), dtype=np.uint8)

    t0    = time.perf_counter()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (21,21), 0)

    tm0   = time.perf_counter()
    mask  = _bg.apply(blur, learningRate=0.01)
    _,mask= cv2.threshold(mask,200,255,cv2.THRESH_BINARY)
    mask  = cv2.erode(mask,_k,iterations=1)
    mask  = cv2.dilate(mask,_k,iterations=3)
    tm1   = time.perf_counter()

    th0   = time.perf_counter()
    _h.detectMultiScale(gray, hitThreshold=0.0, winStride=(8,8), padding=(4,4), scale=1.05)
    th1   = time.perf_counter()
    t1    = time.perf_counter()

    if i >= WARMUP:
        t_mog_l.append((tm1-tm0)*1000)
        t_hog_l.append((th1-th0)*1000)
        t_tot_l.append((t1-t0)*1000)

mog_ms = np.mean(t_mog_l); hog_ms = np.mean(t_hog_l); tot_ms = np.mean(t_tot_l)
fps    = 1000 / tot_ms

print(f"\n    MOG2 (background sub)  : {mog_ms:6.1f} ± {np.std(t_mog_l):.1f} ms")
print(f"    HOG+SVM (detection)    : {hog_ms:6.1f} ± {np.std(t_hog_l):.1f} ms")
print(f"    Tổng pipeline/frame    : {tot_ms:6.1f} ± {np.std(t_tot_l):.1f} ms")
print(f"\n    ★ FPS detection only   : {1000/hog_ms:6.1f} FPS")
print(f"    ★ FPS đầy đủ pipeline  : {fps:6.1f} FPS")
print(f"    ★ FPS thực tế app      : ~{fps*0.75:.0f}–{fps*0.88:.0f} FPS (có I/O + JPEG)")

# Bảng tổng kết
print(f"\n{'═'*62}")
print(f"{'📋 BẢNG TỔNG KẾT — copy vào báo cáo':^62}")
print(f"{'═'*62}")
rows = [
    ("Dataset",                       "INRIA Person Dataset"),
    ("Phương pháp",                   "HOG + LinearSVC"),
    ("HOG Window",                    f"{WIN_W}×{WIN_H} px"),
    ("Feature Vector",                f"{FEATURE_DIM} chiều"),
    ("Cells trong window",            f"{n_cells_x}×{n_cells_y} = {n_cells_x*n_cells_y} cells"),
    ("Blocks trong window",           f"{n_blocks_x}×{n_blocks_y} = {n_blocks} blocks"),
    ("Số windows/frame (640×480)",    f"{total_windows:,}"),
    ("FLOPs trên 1 window",           f"{total_per_w/1e3:.0f}K FLOPs"),
    ("FLOPs trên 1 frame",            f"{total_flops/1e6:.0f} MFLOPs"),
    ("Thời gian MOG2/frame",          f"{mog_ms:.1f} ms"),
    ("Thời gian HOG+SVM/frame",       f"{hog_ms:.1f} ms"),
    ("Tổng thời gian/frame",          f"{tot_ms:.1f} ms"),
    ("FPS (benchmark Colab)",         f"{fps:.1f} FPS"),
    ("Test Accuracy",                 f"{acc:.2%}"),
    ("SVM C",                         "0.01"),
    ("Hard Negative Mining",          f"✅  ({len(X_hard)} samples"),
]
for k, v in rows:
    print(f"  {k:<38} {v}")
print(f"{'═'*62}")


# ===========================================================
# CELL 13 — LƯU MODEL VỀ GOOGLE DRIVE
# ===========================================================
# Format cv2.HOGDescriptor.setSVMDetector():
# w[-1] = -intercept   →   score = dot(w[:-1], x) - w[-1]  ≡  dot(coef, x) + intercept

svm_coef        = clf_final.coef_[0].astype(np.float64)
svm_intercept   = float(clf_final.intercept_[0])
svm_weights_cv2 = np.append(svm_coef, -svm_intercept).astype(np.float64)
assert len(svm_weights_cv2) == FEATURE_DIM + 1

model_bundle = {
    # ─ dùng trong app ─────────────────────────────────────
    'svm_weights'   : svm_weights_cv2,

    # ─ backup / debug ──────────────────────────────────────
    'svm_coef'      : svm_coef,
    'svm_intercept' : svm_intercept,

    # ─ HOG config (phải khớp với person_detector.py) ───────
    'hog_params': {
        'win_w'        : WIN_W,
        'win_h'        : WIN_H,
        'block_size'   : (BLOCK_W, BLOCK_H),
        'block_stride' : (BSTRIDE_W, BSTRIDE_H),
        'cell_size'    : (CELL_W, CELL_H),
        'n_bins'       : N_BINS,
        'feature_dim'  : FEATURE_DIM,
    },

    # ─ thông tin training (dùng cho báo cáo) ───────────────
    'training_info': {
        'dataset'           : 'INRIA Person Dataset (Kaggle: jcoral02/inriaperson)',
        'n_train_pos'       : len(X_pos_tr),
        'n_train_neg'       : len(X_neg_tr),
        'n_hard_neg'        : int(len(X_hard)),
        'total_train'       : len(X_tr2),
        'n_test_pos'        : len(X_pos_te),
        'n_test_neg'        : len(X_neg_te),
        'test_acc'          : round(float(acc), 4),
        'test_acc_pct'      : f"{acc:.2%}",
        'feature_dim'       : FEATURE_DIM,
        'fps_benchmark'     : round(float(fps), 1),
        'fps_hog_only'      : round(float(1000/hog_ms), 1),
        'flops_per_frame'   : int(total_flops),
        'flops_per_window'  : int(total_per_w),
        'windows_per_frame' : int(total_windows),
        'time_per_frame_ms' : round(float(tot_ms), 1),
        'svm_C'             : 0.01,
        'hard_neg_mining'   : True,
    }
}

save_path = f'{MODEL_SAVE_DIR}/hog_svm_model.pkl'
joblib.dump(model_bundle, save_path, compress=3)
size_kb = os.path.getsize(save_path) / 1024

print(f"✅ Model saved  →  {save_path}")
print(f"   File size    :  {size_kb:.0f} KB")
print(f"""
╔══════════════════════════════════════════════════════╗
║           📌 BƯỚC TIẾP THEO SAU KHI TRAIN           ║
╠══════════════════════════════════════════════════════╣
║  1. Vào Google Drive:                                ║
║     MyDrive → MotionGuardAI → models                 ║
║     Download file: hog_svm_model.pkl                 ║
║                                                      ║
║  2. Đặt vào project VS Code:                         ║
║     Proj_TNTT_Ver2/model/hog_svm_model.pkl           ║
║                                                      ║
║  3. Chạy app:                                        ║
║     pip install -r requirements.txt                  ║
║     python app.py                                    ║
║     → http://localhost:5000                          ║
╠══════════════════════════════════════════════════════╣
║  Test Accuracy  : {acc:.2%}                               ║
║  FPS (Colab)    : {fps:.1f} FPS                              ║
║  FLOPs/frame    : {total_flops/1e6:.0f} MFLOPs                       ║
╚══════════════════════════════════════════════════════╝
""")
