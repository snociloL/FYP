console.log("Deepfake Validator: Background Router Active.");

// Create the offscreen document if it doesn't exist
async function setupOffscreen() {
    if (await chrome.offscreen.hasDocument()) return;
    await chrome.offscreen.createDocument({
        url: 'offscreen.html',
        reasons: ['WORKERS'], // Flag indicating computational work
        justification: 'Run TensorFlow.js local inference safely bypassing website CSP restrictions.'
    });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "FRAME_TO_BACKGROUND") {
        setupOffscreen().then(() => {
            // Forward the frame directly to our safe offscreen sandbox
            chrome.runtime.sendMessage({
                action: "FRAME_TO_SANDBOX",
                pixels: request.pixels
            }, (sandboxResponse) => {
                sendResponse(sandboxResponse);
            });
        });
        return true; // Keep message channel open
    }
});