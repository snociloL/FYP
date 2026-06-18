console.log("HELLO FROM THE SANDBOX! If you see this, the file is running.");
let deepfakeModel;

async function loadModel(absoluteUrl) {
    try {
        console.log("1. Initializing TensorFlow backend...");
        await tf.ready();
        console.log("2. Backend ready! Using:", tf.getBackend());
        
        // If this line doesn't print, Chrome is still caching!
        console.log(`3. Fetching model from absolute URL: ${absoluteUrl}`);
        console.time("ModelLoadTimer");
        
        deepfakeModel = await tf.loadLayersModel(absoluteUrl);

        console.timeEnd("ModelLoadTimer");
        console.log("✅ Model loaded successfully!");
        window.parent.postMessage({ action: "MODEL_READY" }, "*");
    } catch (e) {
        console.error("❌ Critical Model Load Error:", e);
    }
}

let frameBuffer = [];
const BATCH_SIZE = 30;

window.addEventListener('message', async (event) => {
    if (event.data && event.data.action === "INIT_MODEL") {
        loadModel(event.data.url);
        return;
    }

    if (event.data && event.data.action === "PROCESS_FRAME") {
        try {
            frameBuffer.push(event.data.pixels);
            
            // This proves the sandbox is catching frames
            if (frameBuffer.length % 10 === 0) {
                console.log(`[Sandbox] Buffer filling: ${frameBuffer.length}/${BATCH_SIZE}`);
            }

            if (frameBuffer.length >= BATCH_SIZE) {
                if (deepfakeModel) {
                    console.log("[Sandbox] 30 frames reached! Starting tensor conversion...");
                    
                    const tensorArray = frameBuffer.map((pixels) => {
                        let flatArray = pixels;
                        if (!(pixels instanceof Uint8ClampedArray) && !(pixels instanceof Float32Array) && !Array.isArray(pixels)) {
                            flatArray = Object.values(pixels);
                        }
                        return tf.tensor3d(flatArray, [224, 224, 4]).slice([0,0,0], [-1,-1,3]).div(255.0);
                    });
                    
                    console.log("[Sandbox] Tensor array built. Stacking sequence...");
                    const sequenceTensor = tf.stack(tensorArray).expandDims(0);
                    
                    console.log("[Sandbox] Math starting! Sequence Shape:", sequenceTensor.shape);
                    const prediction = deepfakeModel.predict(sequenceTensor);
                    const score = prediction.dataSync()[0];
                    
                    console.log("✅ Validation Raw Prediction Score:", score);
                    
                    let state = score > 0.85 ? 'red' : (score < 0.15 ? 'green' : 'gray');
                    window.parent.postMessage({
                        action: "RESULT",
                        state: state,
                        confidence: Math.round(score * 100)
                    }, "*");
                    
                    tensorArray.forEach(t => t.dispose());
                    sequenceTensor.dispose();
                    prediction.dispose();
                } else {
                    console.warn("[Sandbox] Model not ready yet! Skipping batch.");
                }
                frameBuffer = [];
            }
        } catch (err) {
            console.error("❌ CRITICAL SANDBOX MATH ERROR:", err);
            frameBuffer = [];
        }
    }
});