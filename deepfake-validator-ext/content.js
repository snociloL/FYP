// content.js
console.log("Deepfake Validator: Content script active.");

let activeVideo = null;
let captureInterval = null;
const FRAME_RATE = 1000 / 30; 

// 1. Inject the Invisible AI Sandbox
const iframe = document.createElement('iframe');
iframe.src = chrome.runtime.getURL('sandbox.html');
iframe.style.display = 'none';
document.body.appendChild(iframe);

// 2. Setup the Canvas
const canvas = document.createElement('canvas');
canvas.width = 224;
canvas.height = 224;
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// 3. Find the Video
const observer = new MutationObserver(() => {
    const video = document.querySelector('video');
    if (video && video !== activeVideo) {
        activeVideo = video;
        injectUIOverlay(activeVideo);
        activeVideo.addEventListener('play', startCapture);
        activeVideo.addEventListener('pause', stopCapture);
    }
});
observer.observe(document.body, { childList: true, subtree: true });

// 4. Capture and Send Frames to the Sandbox
function startCapture() {
    if (captureInterval) return;
    captureInterval = setInterval(() => {
        if (!activeVideo || activeVideo.paused || activeVideo.ended) return;

        ctx.drawImage(activeVideo, 0, 0, 224, 224);
        const imageData = ctx.getImageData(0, 0, 224, 224);
        
        // Fast, direct communication to our sandbox
        iframe.contentWindow.postMessage({
            action: "PROCESS_FRAME",
            pixels: Array.from(imageData.data)
        }, "*");

    }, FRAME_RATE);
}

function stopCapture() {
    clearInterval(captureInterval);
    captureInterval = null;
}

// 5. Receive the AI's Decision and Update the UI
window.addEventListener('message', (event) => {
    if (event.data && event.data.action === "RESULT") {
        updateUI(event.data.state, event.data.confidence);
    }
});

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