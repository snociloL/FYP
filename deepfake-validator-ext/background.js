const ANALYZER_URL = chrome.runtime.getURL('analyzer.html');
let analyzerTabId = null;

// ── Open / focus the analyzer tab ───────────────────────────────────────────
chrome.action.onClicked.addListener(async () => {
    const existing = await chrome.tabs.query({ url: ANALYZER_URL });
    if (existing.length) {
        analyzerTabId = existing[0].id;
        chrome.tabs.update(existing[0].id, { active: true });
        chrome.windows.update(existing[0].windowId, { focused: true });
    } else {
        const tab = await chrome.tabs.create({ url: ANALYZER_URL });
        analyzerTabId = tab.id;
    }
});

// ── Offscreen document helper ───────────────────────────────────────────────
async function ensureOffscreen() {
    if (await chrome.offscreen.hasDocument()) return;
    await chrome.offscreen.createDocument({
        url: chrome.runtime.getURL('offscreen.html'),
        reasons: ['USER_MEDIA'],
        justification: 'Capture tab stream for deepfake frame extraction'
    });
}

// ── Message router ──────────────────────────────────────────────────────────
chrome.runtime.onMessage.addListener((msg, sender) => {

    // Analyzer asks to capture a YouTube tab
    if (msg.action === 'CAPTURE_YOUTUBE_TAB') {
        // Keep track of which tab is the analyzer
        if (sender.tab?.id) analyzerTabId = sender.tab.id;

        chrome.tabs.query({ url: '*://*.youtube.com/watch*' }, (tabs) => {
            if (!tabs.length) {
                forwardToAnalyzer({
                    action: 'CAPTURE_ERROR',
                    error: 'No YouTube video tab found. Open a YouTube video first.'
                });
                return;
            }

            const ytTab = tabs[0];
            chrome.tabCapture.getMediaStreamId({ targetTabId: ytTab.id }, async (streamId) => {
                if (chrome.runtime.lastError || !streamId) {
                    forwardToAnalyzer({
                        action: 'CAPTURE_ERROR',
                        error: 'Tab capture failed: ' + (chrome.runtime.lastError?.message ?? 'unknown')
                    });
                    return;
                }

                await ensureOffscreen();
                chrome.runtime.sendMessage({ action: 'START_CAPTURE', streamId }).catch(() => {});
            });
        });
        return;
    }

    // Offscreen finished extracting frames → forward to analyzer tab
    if (msg.action === 'TAB_CAPTURE_FRAMES') {
        forwardToAnalyzer({ action: 'TAB_CAPTURE_FRAMES', frames: msg.frames });
        return;
    }

    // Offscreen reports an error during capture
    if (msg.action === 'CAPTURE_ERROR') {
        forwardToAnalyzer({ action: 'CAPTURE_ERROR', error: msg.error });
        return;
    }
});

function forwardToAnalyzer(msg) {
    if (analyzerTabId == null) return;
    chrome.tabs.sendMessage(analyzerTabId, msg).catch(() => {});
}
