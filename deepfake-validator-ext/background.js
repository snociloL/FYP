// The UI is now an action popup (analyzer.html). The popup receives capture
// results directly from the offscreen document's runtime broadcasts, so the
// background only needs to kick off capture and report setup errors.

async function ensureOffscreen() {
    if (await chrome.offscreen.hasDocument()) return;
    await chrome.offscreen.createDocument({
        url: chrome.runtime.getURL('offscreen.html'),
        reasons: ['USER_MEDIA'],
        justification: 'Capture tab stream for deepfake frame extraction'
    });
}

function reportError(error) {
    // Broadcast to the popup (ignored if the popup is closed)
    chrome.runtime.sendMessage({ action: 'CAPTURE_ERROR', error }).catch(() => {});
}

let capturedTabId = null;   // the tab currently being analysed (for the on-page pill)

function toPage(msg) {
    if (capturedTabId != null) chrome.tabs.sendMessage(capturedTabId, msg).catch(() => {});
}

chrome.runtime.onMessage.addListener((msg) => {
    // Popup asks us to show a verdict on the analysed page (on-page status pill)
    if (msg.action === 'SHOW_ON_PAGE') {
        toPage({ action: 'RESULT', state: msg.state, text: msg.text });
        return;
    }

    if (msg.action !== 'CAPTURE_ACTIVE_TAB') return;

    // Capture whatever tab the user is currently viewing — works on any video
    // site (YouTube watch & shorts, Vimeo, X, TikTok, news pages, …)
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tab = tabs[0];
        if (!tab || !tab.id) {
            reportError('No active tab found.');
            return;
        }

        // tabCapture can't capture browser-internal pages
        const url = tab.url || '';
        if (/^(chrome|edge|about|chrome-extension|moz-extension|devtools|view-source):/i.test(url)) {
            reportError('Open a video in a normal web tab, then click Analyze.');
            return;
        }

        chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id }, async (streamId) => {
            if (chrome.runtime.lastError || !streamId) {
                reportError('Tab capture failed: ' + (chrome.runtime.lastError?.message ?? 'unknown'));
                return;
            }
            capturedTabId = tab.id;
            toPage({ action: 'STATUS', text: 'Analysing this video…' });   // pulsing on-page pill
            await ensureOffscreen();
            chrome.runtime.sendMessage({ action: 'START_CAPTURE', streamId }).catch(() => {});
        });
    });
});
