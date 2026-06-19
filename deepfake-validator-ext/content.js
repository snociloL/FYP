const statusDiv = document.createElement('div');
statusDiv.id = 'deepfake-status';
statusDiv.innerText = 'Deepfake Validator: Click the extension icon to start.';
statusDiv.style.cssText = `
    position: fixed; top: 10px; right: 10px; z-index: 2147483647;
    background: rgba(0,0,0,0.85); color: white; padding: 10px 14px;
    font-weight: bold; border-radius: 6px; font-size: 13px;
    font-family: sans-serif; max-width: 280px; line-height: 1.4;
`;
document.body.appendChild(statusDiv);

chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'STATUS') {
        statusDiv.style.background = 'rgba(0,0,0,0.85)';
        statusDiv.innerText = msg.text;
    } else if (msg.action === 'RESULT') {
        if (msg.state === 'red') {
            statusDiv.style.background = 'rgba(180,0,0,0.9)';
            statusDiv.innerText = `DEEPFAKE DETECTED (${msg.confidence}% confidence)`;
        } else if (msg.state === 'green') {
            statusDiv.style.background = 'rgba(0,140,0,0.9)';
            statusDiv.innerText = `REAL VIDEO (${100 - msg.confidence}% confidence)`;
        } else {
            statusDiv.style.background = 'rgba(100,100,0,0.9)';
            statusDiv.innerText = `UNCERTAIN (score: ${msg.confidence}%)`;
        }
    }
});
