// sandbox.js
console.log("Deepfake Validator: Sandboxed TF.js Engine Loaded successfully!");

let frameBuffer = [];
const BATCH_SIZE = 30;

// Listen for frames sent directly from the webpage
window.addEventListener('message', async (event) => {
    if (event.data && event.data.action === "PROCESS_FRAME") {
        
        frameBuffer.push(event.data.pixels);

        // Once we have 30 frames, run inference!
        if (frameBuffer.length >= BATCH_SIZE) {
            
            // --- TENSORFLOW LOGIC WILL GO HERE ---
            // You now have full WebGL + Eval privileges!
            
            const score = Math.random(); // Simulation placeholder
            let state = 'gray';
            if (score > 0.85) state = 'red';
            else if (score < 0.15) state = 'green';

            // Send the Tri-State decision back to the webpage
            event.source.postMessage({ 
                action: "RESULT", 
                state: state, 
                confidence: Math.round(score * 100) 
            }, event.origin);

            frameBuffer = []; // Reset for the next second of video
        }
    }
});