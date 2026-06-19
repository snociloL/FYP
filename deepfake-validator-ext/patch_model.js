const fs = require('fs');
const path = require('path');

// Target your specific model json file path
const jsonPath = path.join(__dirname, 'lib', 'web_model', 'model.json');

console.log("Reading model blueprint...");
let rawData = fs.readFileSync(jsonPath, 'utf8');
let model = JSON.parse(rawData);

let fixCount = 0;

// Recursive function to deeply traverse layers and flatten Keras 3 nodes to Keras 2 arrays
function repairNodes(obj) {
    if (!obj || typeof obj !== 'object') return;

    if (obj.class_name === 'InputLayer' && obj.config) {
        if (obj.config.batch_shape) {
            obj.config.batch_input_shape = obj.config.batch_shape;
            delete obj.config.batch_shape;
        }
        if ('optional' in obj.config) delete obj.config.optional;
    }

    // Look specifically for the Keras 3 functional connection inbound_nodes structure
    if (obj.inbound_nodes && Array.isArray(obj.inbound_nodes)) {
        obj.inbound_nodes = obj.inbound_nodes.map(node => {
            // If the node item is an object instead of a flat array, convert it natively
            if (node && typeof node === 'object' && !Array.isArray(node)) {
                fixCount++;
                const nodeName = node.node_key || node.layer_name || "";
                const outIdx = node.output_tensor_index !== undefined ? node.output_tensor_index : 0;
                return [nodeName, 0, outIdx, {}];
            }
            // If it's a nested array containing objects (common layer config structure)
            if (Array.isArray(node)) {
                return node.map(subNode => {
                    if (subNode && typeof subNode === 'object' && !Array.isArray(subNode)) {
                        fixCount++;
                        const nodeName = subNode.node_key || subNode.layer_name || "";
                        const outIdx = subNode.output_tensor_index !== undefined ? subNode.output_tensor_index : 0;
                        return [nodeName, 0, outIdx, {}];
                    }
                    return subNode;
                });
            }
            return node;
        });
    }

    // Traverse into every structural subtree
    for (let key in obj) {
        if (obj.hasOwnProperty(key)) {
            repairNodes(obj[key]);
        }
    }
}

// Execute the recursive graph patcher
repairNodes(model);

if (fixCount > 0) {
    console.log(`\nSuccessfully converted ${fixCount} graph structural points!`);
    fs.writeFileSync(jsonPath, JSON.stringify(model, null, 2), 'utf8');
    console.log("Saved patched model.json configuration back to disk cleanly.");
} else {
    console.log("\nNo unpatched node structural elements detected inside the file.");
}