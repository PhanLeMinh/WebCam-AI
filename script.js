/**
 * script.js — MotionGuard AI Frontend (YOLOv8 version)
 */

const API = 'http://localhost:5000/api';

// ─── State ────────────────────────────────────────────────────────────────────
let results      = [];
let currentFrame = 0;
let webcamActive = false;
let sessionStats = { frames: 0, alerts: 0, persons: 0, objects: 0 };

const $ = id => document.getElementById(id);

// ─── Utils ────────────────────────────────────────────────────────────────────
function toast(msg, dur = 2800) {
    const el = $('toast');
    el.textContent = msg;
    el.classList.add('show');
    setTimeout(() => el.classList.remove('show'), dur);
}

// ─── Tabs ────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        $(`tab-${tab.dataset.tab}`).classList.add('active');
    });
});

// ─── File upload ──────────────────────────────────────────────────────────────
const uploadZone = $('upload-zone');
const fileInput  = $('file-input');
let selectedFile = null;

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault(); uploadZone.classList.remove('drag');
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) setFile(f);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
    selectedFile = file;
    $('file-name').textContent = `📁 ${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
    $('upload-info').style.display = 'block';
    $('btn-analyze').disabled = false;
    toast(`✅ Đã chọn: ${file.name}`);
}

// ─── Analyze video ────────────────────────────────────────────────────────────
$('btn-analyze').addEventListener('click', async () => {
    if (!selectedFile) return;
    $('btn-analyze').disabled = true;
    $('btn-analyze').textContent = '⏳ Đang phân tích...';
    $('progress-fill').style.width = '15%';

    const fd = new FormData();
    fd.append('video', selectedFile);

    try {
        $('progress-fill').style.width = '45%';
        const res  = await fetch(`${API}/upload`, { method: 'POST', body: fd });
        $('progress-fill').style.width = '90%';
        const data = await res.json();
        $('progress-fill').style.width = '100%';

        if (data.success) {
            results      = data.results;
            currentFrame = 0;

            // Stats
            const totalPersonFrames = data.results.filter(r => r.has_person).length;
            const totalPersons = data.results.reduce((s, r) => s + r.person_count, 0);
            const totalObjects = data.results.reduce((s, r) => s + r.object_count, 0);

            sessionStats.frames  += data.analyzed;
            sessionStats.alerts  += data.alert_count;
            sessionStats.persons += totalPersons;
            sessionStats.objects += totalObjects;
            updateStats();

            showFrame(0);
            buildFrameGrid(data.results);

            $('btn-prev').disabled = false;
            $('btn-next').disabled = false;
            $('frame-info').textContent =
                `📹 ${data.total_frames} frames • ${data.fps.toFixed(0)}fps • 🚨 ${data.alert_count} cảnh báo người`;

            $('webcam-stream').style.display = 'none';
            $('video-placeholder').style.display = 'none';
            $('video-display').style.display = 'block';

            toast(`✅ Xong: ${data.alert_count} lần phát hiện người`);
        } else {
            toast(`❌ Lỗi: ${data.error}`);
        }
    } catch (err) {
        toast(`❌ Lỗi kết nối backend: ${err.message}`);
    } finally {
        $('btn-analyze').disabled = false;
        $('btn-analyze').textContent = '▶ Phân Tích Video';
        setTimeout(() => { $('progress-fill').style.width = '0'; }, 1000);
    }
});

// ─── Show frame ──────────────────────────────────────────────────────────────
function showFrame(idx) {
    if (!results.length) return;
    idx = Math.max(0, Math.min(idx, results.length - 1));
    currentFrame = idx;

    const fr = results[idx];
    $('video-display').src = fr.image;

    // Alert banner
    if (fr.has_person) {
        $('alert-banner').classList.add('show');
        $('alert-detail').textContent =
            `${fr.person_count} người phát hiện • ${fr.object_count} vật thể khác`;
        $('alert-status').style.display = 'flex';
        addAlertToFeed(fr);
    } else if (fr.has_motion) {
        $('alert-banner').classList.remove('show');
        // Hiển thị thông báo vật thể (không cảnh báo)
        $('frame-info').textContent =
            `Frame ${fr.frame} • t=${fr.timestamp}s • Chuyển động: ${fr.object_count} vật thể (không có người)`;
        return;
    } else {
        $('alert-banner').classList.remove('show');
    }

    $('frame-info').textContent =
        `Frame ${fr.frame} • t=${fr.timestamp}s • ` +
        (fr.has_person
            ? `🚨 ${fr.person_count} NGƯỜI`
            : fr.has_motion
                ? `🟡 Chuyển động (không có người)`
                : `✅ Không có chuyển động`) +
        (fr.fps          ? ` • ${fr.fps} FPS` : '') +
        (fr.processing_ms? ` • ${fr.processing_ms}ms` : '');
}

$('btn-prev').addEventListener('click', () => showFrame(currentFrame - 1));
$('btn-next').addEventListener('click', () => showFrame(currentFrame + 1));
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft')  showFrame(currentFrame - 1);
    if (e.key === 'ArrowRight') showFrame(currentFrame + 1);
});

// ─── Frame grid ──────────────────────────────────────────────────────────────
function buildFrameGrid(frames) {
    const grid = $('frame-grid');
    grid.innerHTML = '';
    frames.slice(0, 24).forEach((fr, i) => {
        const div = document.createElement('div');
        div.className = 'frame-thumb';

        let badgeClass = 'ok';
        let badgeTxt   = `✓ ${fr.timestamp}s`;
        if (fr.has_person) {
            badgeClass = 'alert';
            badgeTxt   = `👤 ${fr.person_count} người`;
        } else if (fr.has_motion) {
            badgeClass = 'motion';
            badgeTxt   = `⬤ ${fr.timestamp}s`;
        }

        div.innerHTML = `
            <img src="${fr.image}" loading="lazy" />
            <span class="frame-badge ${badgeClass}">${badgeTxt}</span>
        `;
        div.addEventListener('click', () => showFrame(i));
        grid.appendChild(div);
    });
}

// ─── Alert feed ──────────────────────────────────────────────────────────────
function addAlertToFeed(fr) {
    const feed = $('alert-feed');
    if (feed.querySelector('[data-placeholder]')) feed.innerHTML = '';

    const item = document.createElement('div');
    item.className = 'alert-item';
    item.innerHTML = `
        <div class="alert-time">🕐 t = ${fr.timestamp}s • Frame ${fr.frame}</div>
        <div class="alert-msg">👤 Phát hiện ${fr.person_count} người trong khung hình</div>
        <div class="alert-conf">
            Vật thể khác: ${fr.object_count} •
            ${fr.detections.filter(d=>d.is_person).map(d=>`conf ${(d.confidence*100).toFixed(0)}%`).join(', ')}
        </div>
    `;
    feed.prepend(item);
}

// ─── Stats ───────────────────────────────────────────────────────────────────
function updateStats() {
    $('s-frames').textContent  = sessionStats.frames;
    $('s-alerts').textContent  = sessionStats.alerts;
    $('s-persons').textContent = sessionStats.persons;
    $('s-objects').textContent = sessionStats.objects;
}

// ─── Webcam ───────────────────────────────────────────────────────────────────
$('btn-webcam-start').addEventListener('click', async () => {
    try {
        const res = await fetch(`${API}/webcam/start`, { method: 'POST' });
        const d   = await res.json();
        if (d.error) { toast(`❌ ${d.error}`); return; }

        webcamActive = true;
        $('webcam-stream').src = 'http://localhost:5000/video_feed';
        $('webcam-stream').style.display = 'block';
        $('video-display').style.display = 'none';
        $('video-placeholder').style.display = 'none';

        $('webcam-dot').classList.add('active');
        $('webcam-status-txt').textContent = 'Webcam ON';
        $('btn-webcam-start').disabled = true;
        $('btn-webcam-stop').disabled  = false;
        toast('📷 Webcam đang chạy — HOG+SVM đang theo dõi người');
        startAlertPolling();
    } catch (err) {
        toast(`❌ ${err.message}`);
    }
});

$('btn-webcam-stop').addEventListener('click', async () => {
    await fetch(`${API}/webcam/stop`, { method: 'POST' });
    webcamActive = false;
    $('webcam-stream').src = '';
    $('webcam-stream').style.display = 'none';
    $('video-placeholder').style.display = 'flex';
    $('webcam-dot').classList.remove('active');
    $('webcam-status-txt').textContent = 'Webcam off';
    $('btn-webcam-start').disabled = false;
    $('btn-webcam-stop').disabled  = true;
    toast('📷 Webcam đã tắt');
});

let alertPoll = null;
function startAlertPolling() {
    if (alertPoll) clearInterval(alertPoll);
    alertPoll = setInterval(async () => {
        if (!webcamActive) { clearInterval(alertPoll); return; }
        try {
            const res  = await fetch(`${API}/alerts`);
            const data = await res.json();
            if (data.count > 0) {
                const latest = data.alerts[data.alerts.length - 1];
                $('alert-banner').classList.add('show');
                $('alert-detail').textContent =
                    `${latest.person_count} người phát hiện`;
                $('alert-status').style.display = 'flex';
                sessionStats.alerts = data.count;
                updateStats();

                // Thêm vào feed
                const feed = $('alert-feed');
                if (feed.querySelector('[data-placeholder]')) feed.innerHTML = '';
                const item = document.createElement('div');
                item.className = 'alert-item';
                item.innerHTML = `
                    <div class="alert-time">🕐 ${new Date(latest.timestamp).toLocaleTimeString('vi-VN')}</div>
                    <div class="alert-msg">👤 ${latest.person_count} người trong khung hình</div>
                    <div class="alert-conf">Vật thể khác: ${latest.object_count}</div>
                `;
                feed.prepend(item);
            }
        } catch (_) {}
    }, 1000);
}

// ─── Settings ─────────────────────────────────────────────────────────────────
$('conf-slider').addEventListener('input', () => {
    const v = parseInt($('conf-slider').value);
    $('conf-val').textContent = (v >= 0 ? '+' : '') + v + '.0';
});

$('btn-save-settings').addEventListener('click', async () => {
    const s = {
        conf_threshold    : parseInt($('conf-slider').value),
        skip_if_no_motion : $('toggle-skip').classList.contains('on'),
    };
    try {
        await fetch(`${API}/settings`, {
            method : 'POST',
            headers: { 'Content-Type': 'application/json' },
            body   : JSON.stringify(s)
        });
        toast('✅ Đã lưu cài đặt');
    } catch (err) {
        toast(`❌ ${err.message}`);
    }
});

$('toggle-skip').addEventListener('click', function() {
    this.classList.toggle('on');
});

// ─── Clear alerts ──────────────────────────────────────────────────────────────
$('btn-clear-alerts').addEventListener('click', async () => {
    try { await fetch(`${API}/alerts/clear`, { method: 'POST' }); } catch (_) {}
    $('alert-feed').innerHTML = `<div style="color:var(--txt2);font-size:11px;text-align:center;padding:20px" data-placeholder>Chưa có cảnh báo nào</div>`;
    $('alert-banner').classList.remove('show');
    $('alert-status').style.display = 'none';
    sessionStats.alerts = 0;
    updateStats();
    toast('🗑 Đã xóa log');
});

// ─── Load model status ────────────────────────────────────────────────────────
async function loadModelStatus() {
    try {
        const res  = await fetch(`${API}/model/status`);
        const data = await res.json();
        const badge = $('model-badge-container');
        if (data.model_loaded) {
            badge.innerHTML = `<span class="model-badge loaded">✅ HOG+SVM sẵn sàng</span>`;
            $('model-dot').classList.add('active');
            $('model-status-txt').textContent = 'HOG+SVM loaded';
            const flops = data.flops_per_frame
                ? `${(data.flops_per_frame/1e6).toFixed(0)}M FLOPs/frame`
                : '';
            $('model-info').textContent =
                `HOG+LinearSVC • ${data.feature_dim} features • `+
                `acc ${data.test_acc} • ${data.fps_benchmark} FPS • ${flops}`;
        } else {
            badge.innerHTML = `<span class="model-badge missing">⚠️ Chưa có model — xem hướng dẫn</span>`;
            $('model-status-txt').textContent = 'Model chưa load';
            $('model-info').textContent = 'Chạy colab/train_hog_svm.py → đặt model vào model/hog_svm_model.pkl';
        }
    } catch (_) {
        $('model-status-txt').textContent = 'Backend offline';
    }
}

// ─── Init ─────────────────────────────────────────────────────────────────────
(async () => {
    await loadModelStatus();
    try {
        const res  = await fetch(`${API}/health`);
        const data = await res.json();
        if (!data.model_loaded) {
            toast('⚠ Chưa có model — đặt hog_svm_model.pkl vào thư mục model/');
        }
    } catch (_) {
        toast('⚠ Backend chưa chạy — hãy chạy python app.py');
    }
})();
