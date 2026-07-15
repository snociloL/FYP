const statusEl    = document.getElementById('statusEl');
const videoInput  = document.getElementById('videoInput');
const uploadLabel = document.getElementById('uploadLabel');
const progressEl  = document.getElementById('progressEl');
const resultEl    = document.getElementById('resultEl');
const resultText  = document.getElementById('resultText');
const confText    = document.getElementById('confText');
const ytBtn       = document.getElementById('ytBtn');

const BACKEND = 'https://snocilol-deepfake-validator.hf.space';

function setStatus(text) { statusEl.textContent = text; }

function showResult(label, pct, message) {
    if (label === 'NO_FACE') {
        resultEl.className     = 'result uncertain';
        resultText.textContent = 'NO FACE DETECTED';
        confText.textContent   = message || 'This tool only analyses faces.';
        progressEl.textContent = '';
        return;
    }
    const cls = label === 'DEEPFAKE' ? 'fake'
              : label === 'AUTHENTIC' ? 'real'
              : 'uncertain';
    resultEl.className    = 'result ' + cls;
    resultText.textContent = label === 'DEEPFAKE' ? 'DEEPFAKE DETECTED'
                           : label === 'AUTHENTIC' ? 'AUTHENTIC'
                           : 'UNCERTAIN';
    confText.textContent   = pct + '% confidence';
    progressEl.textContent = '';
}

function setInputsEnabled(enabled) {
    videoInput.disabled = !enabled;
    ytBtn.disabled      = !enabled;
    if (enabled) uploadLabel.classList.remove('disabled');
    else         uploadLabel.classList.add('disabled');
}

// ── Backend health check (retries every 5 s) ────────────────────────────────
async function checkBackend() {
    try {
        const r = await fetch(BACKEND + '/health', { signal: AbortSignal.timeout(3000) });
        if (r.ok) {
            setStatus('Backend ready — upload a video or capture the current tab.');
            setInputsEnabled(true);
            return;
        }
    } catch { /* ignore */ }
    setStatus('Waking up the cloud backend… (first load can take ~30s)');
    setTimeout(checkBackend, 5000);
}
checkBackend();

// ── Messages from background (YouTube capture result) ───────────────────────
chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'TAB_CAPTURE_FRAMES') {
        progressEl.textContent = 'Frames captured — running inference…';
        postToBackend(msg.frames, true).finally(() => setInputsEnabled(true));

    } else if (msg.action === 'CAPTURE_ERROR') {
        progressEl.textContent = 'Capture error: ' + msg.error;
        setInputsEnabled(true);
    }
});

// ── YouTube tab capture ─────────────────────────────────────────────────────
ytBtn.addEventListener('click', () => {
    resultEl.className     = 'result';
    progressEl.textContent = 'Capturing frames from the current tab (~3 s)…';
    setInputsEnabled(false);
    chrome.runtime.sendMessage({ action: 'CAPTURE_ACTIVE_TAB' });
});

// ── Video file upload ───────────────────────────────────────────────────────
videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    resultEl.className     = 'result';
    progressEl.textContent = 'Loading video…';
    setInputsEnabled(false);

    try {
        const frames = await extractFrames(file);
        progressEl.textContent = 'Running inference…';
        await postToBackend(frames);
    } catch (err) {
        progressEl.textContent = 'Error: ' + err.message;
    } finally {
        setInputsEnabled(true);
        videoInput.value = '';
    }
});

// ── Shared: POST frames to backend ──────────────────────────────────────────
async function postToBackend(frames, onPage = false) {
    const resp = await fetch(BACKEND + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frames }),
    });

    if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.error || 'Backend error ' + resp.status);
    }

    const { label, confidence, message } = await resp.json();
    showResult(label, confidence, message);

    // For tab analysis, also surface the verdict on the page itself (status pill)
    if (onPage) {
        const map = {
            DEEPFAKE:  ['red',   `Deepfake detected · ${confidence}%`],
            AUTHENTIC: ['green', `Authentic · ${confidence}%`],
            UNCERTAIN: ['amber', 'Uncertain'],
            NO_FACE:   ['amber', 'No face detected'],
        };
        const [state, text] = map[label] || ['amber', 'Result'];
        chrome.runtime.sendMessage({ action: 'SHOW_ON_PAGE', state, text }).catch(() => {});
    }
}

// ── Frame extraction from uploaded file ─────────────────────────────────────
async function extractFrames(file) {
    const BATCH = 30;
    const url   = URL.createObjectURL(file);
    try {
        const video = document.createElement('video');
        video.src        = url;
        video.muted      = true;
        video.playsInline = true;

        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = () => reject(new Error('Could not load video'));
            setTimeout(() => reject(new Error('Video load timeout')), 15000);
        });

        const { duration } = video;
        if (!duration || !isFinite(duration)) throw new Error('Cannot determine video duration');

        // Send larger, aspect-correct frames so the backend can detect faces.
        // (The model still sees 224×224 face crops — cropping happens server-side.)
        const MAX_SIDE = 720;
        const vw = video.videoWidth || 640, vh = video.videoHeight || 360;
        const scale = Math.min(1, MAX_SIDE / Math.max(vw, vh));
        const canvas = document.createElement('canvas');
        canvas.width  = Math.round(vw * scale);
        canvas.height = Math.round(vh * scale);
        const ctx    = canvas.getContext('2d');
        const frames = [];

        for (let i = 0; i < BATCH; i++) {
            video.currentTime = (duration * i) / BATCH;
            await new Promise(res => { video.onseeked = res; video.onerror = res; });
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            frames.push(canvas.toDataURL('image/jpeg', 0.8));
            progressEl.textContent = `Extracting frames (${i + 1}/${BATCH})…`;
        }

        return frames;
    } finally {
        URL.revokeObjectURL(url);
    }
}
