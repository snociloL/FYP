const statusEl   = document.getElementById('statusEl');
const captureBtn = document.getElementById('captureBtn');
const videoInput = document.getElementById('videoInput');
const progressEl = document.getElementById('progressEl');
const resultEl   = document.getElementById('resultEl');
const resultText = document.getElementById('resultText');
const confText   = document.getElementById('confText');
const sandbox    = document.getElementById('sandboxFrame');

let capturing  = false;
let modelReady = false;
let outgoingShards = null;   // buffers awaiting flow-controlled send to the sandbox

function setStatus(text) { statusEl.textContent = text; }

function showResult(state, confidence) {
    resultEl.className = 'result ' + (state === 'red' ? 'fake' : state === 'green' ? 'real' : 'uncertain');
    resultText.textContent = state === 'red'   ? 'DEEPFAKE DETECTED'
                           : state === 'green' ? 'AUTHENTIC'
                           :                     'UNCERTAIN';
    confText.textContent = confidence + '% confidence';
    progressEl.textContent = '';
}

// ── Sandbox iframe (TF.js lives here — popup renderer has a full memory budget) ──
window.addEventListener('message', (event) => {
    if (!event.data) return;
    const { action } = event.data;

    if (action === 'SANDBOX_READY') {
        loadModelIntoSandbox();

    } else if (action === 'SHARD_ACK') {
        sendShard(event.data.next);

    } else if (action === 'SANDBOX_LOG') {
        setStatus(event.data.text);

    } else if (action === 'MODEL_READY') {
        modelReady = true;
        setStatus('AI Ready — upload a video to analyze.');

    } else if (action === 'SANDBOX_ERROR') {
        setStatus('Model error: ' + event.data.text);

    } else if (action === 'RESULT') {
        showResult(event.data.state, event.data.confidence);
    }
});

async function loadModelIntoSandbox() {
    try {
        const base = chrome.runtime.getURL('lib/web_model/');
        setStatus('Fetching model.json...');
        const modelJson = await fetch(base + 'model.json').then(r => r.json());

        const shardPaths = modelJson.weightsManifest.flatMap(g => g.paths);
        setStatus('Loading weights (0/' + shardPaths.length + ')...');

        const fileBuffers = [];
        for (let i = 0; i < shardPaths.length; i++) {
            fileBuffers.push(await fetch(base + shardPaths[i]).then(r => r.arrayBuffer()));
            setStatus('Loading weights (' + (i + 1) + '/' + shardPaths.length + ')...');
        }

        // Re-chunk into 4 MB pieces — large single transfers are dropped by the IPC layer
        outgoingShards = chunkBuffers(fileBuffers, 4 * 1024 * 1024);
        const totalBytes = fileBuffers.reduce((s, b) => s + b.byteLength, 0);
        setStatus('Initializing model...');
        sandbox.contentWindow.postMessage({
            action: 'INIT_MODEL_START',
            modelJson,
            totalShards: outgoingShards.length,
            totalBytes
        }, '*');
    } catch (e) {
        setStatus('Model load failed: ' + e.message);
    }
}

function sendShard(index) {
    if (!outgoingShards || index >= outgoingShards.length) return;
    const buf = outgoingShards[index];
    sandbox.contentWindow.postMessage(
        { action: 'WEIGHT_SHARD', index, data: buf },
        '*',
        [buf]
    );
    outgoingShards[index] = null;
    if (index === outgoingShards.length - 1) outgoingShards = null;
}

// Concatenate file buffers, then slice into fixed-size ArrayBuffer chunks
function chunkBuffers(fileBuffers, chunkSize) {
    const total = fileBuffers.reduce((s, b) => s + b.byteLength, 0);
    const all = new Uint8Array(total);
    let off = 0;
    for (const b of fileBuffers) { all.set(new Uint8Array(b), off); off += b.byteLength; }
    const chunks = [];
    for (let i = 0; i < total; i += chunkSize) {
        chunks.push(all.buffer.slice(i, Math.min(i + chunkSize, total)));
    }
    return chunks;
}

// ── Tab capture button (still routed through background/offscreen) ───────────────
captureBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) return;

    if (capturing) {
        capturing = false;
        captureBtn.textContent = 'Analyze Current Tab';
        captureBtn.className = 'btn btn-tab';
        chrome.runtime.sendMessage({ action: 'TOGGLE_CAPTURE', tabId: tab.id, stop: true });
    } else {
        chrome.runtime.sendMessage({ action: 'TOGGLE_CAPTURE', tabId: tab.id, stop: false });
        window.close();
    }
});

// ── Video upload — inference runs in this popup's sandbox iframe ─────────────────
videoInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!modelReady) {
        progressEl.textContent = 'Model still loading — please wait.';
        videoInput.value = '';
        return;
    }

    resultEl.className = 'result';
    progressEl.textContent = 'Loading video...';
    videoInput.disabled = true;

    try {
        const frames = await extractFrames(file);
        progressEl.textContent = 'Running inference...';
        sandbox.contentWindow.postMessage({ action: 'PROCESS_FRAME', frames }, '*');
    } catch (err) {
        progressEl.textContent = 'Error: ' + err.message;
    } finally {
        videoInput.disabled = false;
        videoInput.value = '';
    }
});

async function extractFrames(file) {
    const BATCH = 30;
    const url = URL.createObjectURL(file);

    try {
        const video = document.createElement('video');
        video.src = url;
        video.muted = true;
        video.playsInline = true;

        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = () => reject(new Error('Could not load video'));
            setTimeout(() => reject(new Error('Video load timeout')), 15000);
        });

        const { duration } = video;
        if (!duration || !isFinite(duration)) throw new Error('Cannot determine video duration');

        const canvas = document.createElement('canvas');
        canvas.width = 224;
        canvas.height = 224;
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const frames = [];

        for (let i = 0; i < BATCH; i++) {
            video.currentTime = (duration * i) / BATCH;
            await new Promise(res => { video.onseeked = res; video.onerror = res; });
            ctx.drawImage(video, 0, 0, 224, 224);
            frames.push(Array.from(ctx.getImageData(0, 0, 224, 224).data));
            progressEl.textContent = `Extracting frames (${i + 1}/${BATCH})...`;
        }

        return frames;
    } finally {
        URL.revokeObjectURL(url);
    }
}
