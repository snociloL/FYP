const video  = document.getElementById('tab-video');
const canvas = document.getElementById('frame-canvas');
const ctx    = canvas.getContext('2d');

const BATCH_SIZE = 30;
let captureInterval = null;
let frameBuffer = [];

async function startCapture(streamId) {
    if (captureInterval) {
        clearInterval(captureInterval);
        captureInterval = null;
    }
    frameBuffer = [];

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                mandatory: {
                    chromeMediaSource: 'tab',
                    chromeMediaSourceId: streamId,
                    maxWidth: 1280,
                    maxHeight: 720
                }
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        captureInterval = setInterval(() => {
            if (video.readyState < 2) return;

            ctx.drawImage(video, 0, 0, 224, 224);
            frameBuffer.push(canvas.toDataURL('image/jpeg', 0.85));

            if (frameBuffer.length >= BATCH_SIZE) {
                clearInterval(captureInterval);
                captureInterval = null;
                stopStream();

                chrome.runtime.sendMessage({
                    action: 'TAB_CAPTURE_FRAMES',
                    frames: frameBuffer.slice()
                }).catch(() => {});
                frameBuffer = [];
            }
        }, 100);

    } catch (e) {
        chrome.runtime.sendMessage({
            action: 'CAPTURE_ERROR',
            error: 'Stream capture failed: ' + e.message
        }).catch(() => {});
    }
}

function stopStream() {
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
}

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'START_CAPTURE') {
        startCapture(msg.streamId);
    } else if (msg.action === 'STOP_CAPTURE') {
        if (captureInterval) { clearInterval(captureInterval); captureInterval = null; }
        stopStream();
        frameBuffer = [];
    }
});
