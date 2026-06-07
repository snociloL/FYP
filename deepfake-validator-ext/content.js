console.log("Deepfake Validator: Content script & TF.js loaded.");

let activeVideo = null;
let captureInterval = null;
const FRAME_RATE = 1000 / 30; // 30 FPS
let frameBuffer = [];
const BATCH_SIZE = 30; // 1 second of video

// Hidden canvas for processing frames
const canvas = document.createElement('canvas');
canvas.width = 224;
canvas.height = 224;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// --- NEW: INITIALIZE TENSORFLOW WITH STRICT CSP SAFETY ---
async function initTensorFlow() {
    try {
        // Force TF.js to not use evaluated code generation features
        tf.env().set('ENGINE_COMPILE_ONLY', true);
        
        // Try setting to webgl first (GPU-accelerated, usually safe in content scripts)
        // If it still errors out on specific sites, we default down to CPU mode.
        await tf.setBackend('webgl');
        console.log("Deepfake Validator: TF.js initialized using WebGL backend.");
    } catch (e) {
        console.warn("WebGL fallback triggered. Using CPU backend due to policy restrictions:", e);
        await tf.setBackend('cpu');
        console.log("Deepfake Validator: TF.js initialized using CPU backend.");
    }
}
// Run initialization as soon as the script injects
initTensorFlow();

// 1. Observe the DOM for video elements
const observer = new MutationObserver(() => {
    const video = document.querySelector('video');
    if (video && video !== activeVideo) {
        activeVideo = video;
        console.log("Deepfake Validator: Video found!");
        injectUIOverlay(activeVideo);
        
        activeVideo.addEventListener('play', startCapture);
        activeVideo.addEventListener('pause', stopCapture);
    }
});
observer.observe(document.body, { childList: true, subtree: true });

// 2. Capture and Process Frames Locally
function startCapture() {
    if (captureInterval) return;
    
    captureInterval = setInterval(() => {
        if (!activeVideo || activeVideo.paused || activeVideo.ended) return;

        // Draw current video frame to canvas
        ctx.drawImage(activeVideo, 0, 0, 224, 224);
        const imageData = ctx.getImageData(0, 0, 224, 224);
        const pixelArray = Array.from(imageData.data);
        
        frameBuffer.push(pixelArray);

        // Once we have 30 frames, run inference locally!
        if (frameBuffer.length >= BATCH_SIZE) {
            analyzeSequenceLocally(frameBuffer);
            frameBuffer = []; // Reset buffer
        }

    }, FRAME_RATE);
}

function stopCapture() {
    clearInterval(captureInterval);
    captureInterval = null;
}

// 3. The Local TensorFlow.js Engine
async function analyzeSequenceLocally(frames) {
    try {
        // --- SIMULATION (While we wait for your Python model) ---
        const score = Math.random(); 

        let state = 'gray';
        if (score > 0.85) state = 'red';
        else if (score < 0.15) state = 'green';

        updateUI(state, Math.round(score * 100));

    } catch (error) {
        console.error("Inference error:", error);
        updateUI('gray', 0);
    }
}

// 4. UI Injection and Updating
function injectUIOverlay(videoNode) {
    if(document.getElementById('df-validator-overlay')) return;
    
    const overlay = document.createElement('div');
    overlay.id = 'df-validator-overlay';
    overlay.className = 'df-state-gray';
    overlay.innerHTML = `<span id="df-status-text">Analyzing...</span>`;
    videoNode.parentNode.insertBefore(overlay, videoNode.nextSibling);
}

function updateUI(state, confidence) {
    const overlay = document.getElementById('df-validator-overlay');
    const text = document.getElementById('df-status-text');
    if(!overlay) return;

    overlay.className = `df-state-${state}`;
    text.innerText = state === 'red' ? `Fake (${confidence}%)` : 
                     state === 'green' ? `Real (${confidence}%)` : `Analyzing...`;
}