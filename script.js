import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2';

env.allowLocalModels = false;

let embedder = null;
let sessionPhase1 = null;
let sessionPhase2 = null;
let categoryLabels = [];

window.showView = function(viewId, navEl) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById(`view-${viewId}`).classList.add('active');

    document.querySelectorAll('.nav-item').forEach(b => b.classList.remove('active'));
    const targetNav = document.querySelector(`.nav-item[data-view="${viewId}"]`);
    if (targetNav) targetNav.classList.add('active');
    if (navEl) navEl.classList.add('active');
};

window.toggleSidebar = function() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('collapsed');
};

async function initializeModels() {
    const statusEl = document.getElementById('loading-status');
    try {
        const res = await fetch('models/categories.json');
        categoryLabels = await res.json();
        embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
        sessionPhase1 = await ort.InferenceSession.create('models/model_phase1_v2.onnx');
        sessionPhase2 = await ort.InferenceSession.create('models/model_phase2_v2.onnx');

        statusEl.querySelector('.nav-text').innerText = 'Ready Models';
        statusEl.classList.add('ready');
        document.getElementById('btn-1').disabled = false;
        document.getElementById('btn-2').disabled = false;
        setTimeout(() => {
            statusEl.style.display = 'none';
        }, 3000);
    } catch (error) {
        console.error("Error loading models:", error);
        statusEl.querySelector('.nav-text').innerText = 'Error loading models (F12)';
    }
}

window.classify = async function(phase) {
    const resultBox = document.getElementById(`result-${phase}`);
    const outputSpan = document.getElementById(`category-output-${phase}`);
    const probDetails = document.getElementById(`prob-details-${phase}`);
    const probListElement = document.getElementById(`prob-list-${phase}`);
    const errorMsg = document.getElementById(`error-${phase}`);

    errorMsg.style.display = 'none';
    errorMsg.style.color = '#dc2626';

    try {
        let text, float32Data, tensor, session;

        if (phase === 'phase1') {
            text = document.getElementById('desc-1').value.trim();
        } else {
            text = document.getElementById('desc-2').value.trim();
        }

        if (text.length < 5) {
            errorMsg.innerText = "The description must contain at least 5 valid characters";
            errorMsg.style.display = 'block';
            return;
        }

        outputSpan.innerText = "Analyzing...";
        resultBox.style.display = 'flex';
        probDetails.style.display = 'none';

        if (phase === 'phase1') {
            session = sessionPhase1;
            const embOutput = await embedder(text, { pooling: 'mean', normalize: true });
            float32Data = embOutput.data;
            tensor = new ort.Tensor('float32', float32Data, [1, 384]);
            
        } else {
            session = sessionPhase2;
            
            const featureIds = ['v-vol', 'v-sales', 'v-sup', 'v-own', 'v-avg', 'v-mcap', 'v-tr', 'v-ed', 'v-floor'];
            const features = featureIds.map(id => {
                const inputElement = document.getElementById(id);
                if (inputElement.value.trim() === '' || parseFloat(inputElement.value) < 0) {
                    inputElement.value = '0';
                }
                return parseFloat(inputElement.value);
            });

            const embOutput = await embedder(text, { pooling: 'mean', normalize: true });
            float32Data = new Float32Array(384 + 9);
            float32Data.set(embOutput.data, 0);
            float32Data.set(features, 384);
            tensor = new ort.Tensor('float32', float32Data, [1, 393]);
        }

        const inputName = session.inputNames[0];
        const probOutputName = session.outputNames[1];  

        const feeds = {};
        feeds[inputName] = tensor;
        const results = await session.run(feeds);

        const probabilities = results[probOutputName].data;

        let maxProbIndex = 0;
        let maxProbValue = probabilities[0];
        
        for (let i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProbValue) {
                maxProbValue = probabilities[i];
                maxProbIndex = i;
            }
        }

        const predictedCategory = categoryLabels[maxProbIndex];
        const mainConfidence = (maxProbValue * 100).toFixed(2);

        outputSpan.innerText = `${predictedCategory} (${mainConfidence}%)`;

        probListElement.innerHTML = '';
        let probArray = [];
        for (let i = 0; i < categoryLabels.length; i++) {
            probArray.push({
                category: categoryLabels[i],
                percent: (probabilities[i] * 100)
            });
        }

        probArray.sort((a, b) => b.percent - a.percent);
        probArray.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${item.category}</strong><span>${item.percent.toFixed(2)}%</span>`;
            probListElement.appendChild(li);
        });

        probDetails.style.display = 'block';
    } catch (error) {
        console.error("Error in inference:", error);
        outputSpan.innerText = "Error in prediction (check console, F12).";
        probDetails.style.display = 'none';
    }
};

initializeModels();
