const pill = document.createElement('div');
pill.id = 'deepfake-status';
const dot = document.createElement('span');
dot.className = 'df-dot';
const txt = document.createElement('span');
txt.className = 'df-text';
pill.append(dot, txt);
document.body.appendChild(pill);

let hideTimer;

function show(text, state /* '', 'busy', 'red', 'green', 'amber' */) {
    pill.classList.remove('df-busy', 'df-red', 'df-green', 'df-amber');
    if (state) pill.classList.add('df-' + state);
    txt.textContent = text;
    // force-reflow so the transition replays even on rapid updates
    void pill.offsetWidth;
    pill.classList.add('df-show');
}

function autoHide(ms) {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => pill.classList.remove('df-show'), ms);
}

// The pill stays hidden until an analysis is running — passive and non-intrusive.
chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'STATUS') {
        show(msg.text, 'busy');
        clearTimeout(hideTimer);            // keep visible while analyzing
    } else if (msg.action === 'RESULT') {
        show(msg.text || 'Result', msg.state || 'amber');
        autoHide(6000);
    }
});
