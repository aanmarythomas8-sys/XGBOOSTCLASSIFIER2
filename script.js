/**
 * XGBoost Classifier Lab - Core Logic & Browser ML Engine
 * Client-side implementation of Decision Trees, Ensembling, and Interactive Visualizations.
 */

// Global State
const state = {
    dataset: [],
    features: [],
    target: 'diagnosis',
    trainData: [],
    testData: [],
    modelParams: {
        nEstimators: 30,
        learningRate: 0.10,
        maxDepth: 3,
        subsample: 0.80,
        colsampleBytree: 0.80,
        regAlpha: 0.0
    },
    tablePagination: {
        currentPage: 1,
        pageSize: 10,
        filteredData: []
    },
    trainedEnsemble: null,
    testPredictions: [],
    chartInstances: {}
};

// Default Feature Stats for Standard Normalization / Fallback
const defaultFeatureDefaults = {
    'radius_mean': 14.12,
    'texture_mean': 19.29,
    'perimeter_mean': 91.96,
    'area_mean': 654.8,
    'smoothness_mean': 0.096,
    'compactness_mean': 0.104,
    'concavity_mean': 0.088,
    'concave_points_mean': 0.048,
    'symmetry_mean': 0.181,
    'fractal_dimension_mean': 0.062
};

// Initialize Application on DOM Content Loaded
document.addEventListener('DOMContentLoaded', async () => {
    initUIEventListeners();
    await loadDataset();
    initDatasetExplorer();
    renderPlaygroundInputs();
    runXGBoostPipeline();
});

// ==========================================
// 1. DATASET LOADING & PARSING
// ==========================================
async function loadDataset() {
    appendTerminal('hero-terminal', '$ Loading data.csv via PapaParse...');
    try {
        const response = await fetch('data.csv');
        const csvText = await response.text();
        
        return new Promise((resolve) => {
            Papa.parse(csvText, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                complete: function(results) {
                    state.dataset = results.data;
                    if (state.dataset.length > 0) {
                        state.features = Object.keys(state.dataset[0]).filter(k => k !== state.target);
                    }
                    appendTerminal('hero-terminal', `✓ Successfully loaded ${state.dataset.length} samples.`, 'success');
                    resolve();
                }
            });
        });
    } catch (err) {
        appendTerminal('hero-terminal', '⚠ File fetch error. Generating synthetic fallback data...', 'highlight');
        generateFallbackDataset();
    }
}

function generateFallbackDataset() {
    state.dataset = [];
    state.features = Object.keys(defaultFeatureDefaults);
    for (let i = 0; i < 569; i++) {
        const isMalignant = Math.random() > 0.62 ? 1 : 0;
        const row = {};
        state.features.forEach(f => {
            const base = defaultFeatureDefaults[f];
            const noise = (Math.random() - 0.5) * base * 0.3;
            const shift = isMalignant ? base * 0.15 : -base * 0.1;
            row[f] = parseFloat((base + noise + shift).toFixed(3));
        });
        row[state.target] = isMalignant;
        state.dataset.push(row);
    }
}

// ==========================================
// 2. DATA EXPLORER & TABLE LOGIC
// ==========================================
function initDatasetExplorer() {
    document.getElementById('summary-rows').innerText = state.dataset.length;
    document.getElementById('summary-features').innerText = state.features.length;
    
    const splitIdx = Math.floor(state.dataset.length * 0.8);
    document.getElementById('summary-train').innerText = splitIdx;
    document.getElementById('summary-test').innerText = state.dataset.length - splitIdx;

    state.tablePagination.filteredData = [...state.dataset];
    renderTable();
}

function renderTable() {
    const tableEl = document.getElementById('dataset-table');
    const { currentPage, pageSize, filteredData } = state.tablePagination;
    
    // Header
    let headHTML = '<tr><th>#</th>';
    state.features.forEach(f => { headHTML += `<th>${f}</th>`; });
    headHTML += '<th>Target (Diagnosis)</th></tr>';
    tableEl.querySelector('thead').innerHTML = headHTML;

    // Body
    const startIdx = (currentPage - 1) * pageSize;
    const pageRows = filteredData.slice(startIdx, startIdx + pageSize);
    let bodyHTML = '';
    
    pageRows.forEach((row, idx) => {
        bodyHTML += `<tr><td>${startIdx + idx + 1}</td>`;
        state.features.forEach(f => { bodyHTML += `<td>${row[f]}</td>`; });
        const badgeClass = row[state.target] === 1 ? 'accent-red' : 'accent-green-bright';
        bodyHTML += `<td><span style="color:var(--${badgeClass}); font-weight:bold;">${row[state.target]}</span></td></tr>`;
    });
    
    tableEl.querySelector('tbody').innerHTML = bodyHTML;
    document.getElementById('pagination-info').innerText = 
        `Showing ${startIdx + 1} to ${Math.min(startIdx + pageSize, filteredData.length)} of ${filteredData.length} entries`;
    document.getElementById('current-page-num').innerText = currentPage;
}

// ==========================================
// 3. DECISION TREE & BOOSTING ENGINE (IN BROWSER)
// ==========================================
class DecisionTreeNode {
    constructor(depth = 0) {
        this.depth = depth;
        this.feature = null;
        this.threshold = null;
        this.left = null;
        this.right = null;
        this.value = null; // Output Leaf Score
    }
}

function trainDecisionTree(X, residuals, depth, maxDepth, colsample) {
    const node = new DecisionTreeNode(depth);
    
    // Base condition for leaf node
    if (depth >= maxDepth || X.length <= 5) {
        const sumRes = residuals.reduce((a, b) => a + b, 0);
        node.value = sumRes / (X.length + 1.0); // Regularized leaf weight
        return node;
    }

    // Subsample features (colsample_bytree)
    const numFeaturesToSample = Math.max(1, Math.floor(state.features.length * colsample));
    const sampledFeatures = [...state.features].sort(() => 0.5 - Math.random()).slice(0, numFeaturesToSample);

    let bestGain = -Infinity;
    let bestFeature = null;
    let bestThreshold = null;

    sampledFeatures.forEach(feature => {
        const values = X.map(r => r[feature]).sort((a, b) => a - b);
        for (let i = 0; i < values.length - 1; i += Math.max(1, Math.floor(values.length / 10))) {
            const threshold = (values[i] + values[i + 1]) / 2;
            
            const leftIndices = [];
            const rightIndices = [];
            X.forEach((row, idx) => {
                if (row[feature] <= threshold) leftIndices.push(idx);
                else rightIndices.push(idx);
            });

            if (leftIndices.length === 0 || rightIndices.length === 0) continue;

            const sumLeft = leftIndices.reduce((acc, idx) => acc + residuals[idx], 0);
            const sumRight = rightIndices.reduce((acc, idx) => acc + residuals[idx], 0);
            
            const gain = (sumLeft * sumLeft) / (leftIndices.length + 1) + 
                         (sumRight * sumRight) / (rightIndices.length + 1);

            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = feature;
                bestThreshold = threshold;
            }
        }
    });

    if (!bestFeature) {
        const sumRes = residuals.reduce((a, b) => a + b, 0);
        node.value = sumRes / (X.length + 1.0);
        return node;
    }

    node.feature = bestFeature;
    node.threshold = bestThreshold;

    const leftX = [], rightX = [], leftRes = [], rightRes = [];
    X.forEach((row, idx) => {
        if (row[bestFeature] <= bestThreshold) {
            leftX.push(row);
            leftRes.push(residuals[idx]);
        } else {
            rightX.push(row);
            rightRes.push(residuals[idx]);
        }
    });

    node.left = trainDecisionTree(leftX, leftRes, depth + 1, maxDepth, colsample);
    node.right = trainDecisionTree(rightX, rightRes, depth + 1, maxDepth, colsample);

    return node;
}

function predictTree(node, row) {
    if (node.value !== null) return node.value;
    if (row[node.feature] <= node.threshold) {
        return predictTree(node.left, row);
    }
    return predictTree(node.right, row);
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

// ==========================================
// 4. MODEL EXECUTION PIPELINE
// ==========================================
async function runXGBoostPipeline() {
    setExecutionStatus('RUNNING');
    clearTerminal('main-terminal');
    appendTerminal('main-terminal', 'xgb_env > Executing XGBoost Classification Pipeline...');
    
    // Highlight Pipeline Animation
    animatePipelineStep(1);
    await sleep(200);

    // 1. Train/Test Split
    animatePipelineStep(2);
    const shuffled = [...state.dataset].sort(() => 0.5 - Math.random());
    const splitIdx = Math.floor(shuffled.length * 0.8);
    state.trainData = shuffled.slice(0, splitIdx);
    state.testData = shuffled.slice(splitIdx);
    
    appendTerminal('main-terminal', `✓ Data split complete: Train=${state.trainData.length}, Test=${state.testData.length}`, 'success');
    animatePipelineStep(3);
    await sleep(200);

    // 2. Sequential Boosting Rounds
    animatePipelineStep(4);
    appendTerminal('main-terminal', `xgb_env > Constructing ${state.modelParams.nEstimators} trees (learning_rate=${state.modelParams.learningRate})...`);
    
    const trees = [];
    const basePrediction = 0.0; // log-odds initialization
    let currentTrainPredictions = new Array(state.trainData.length).fill(basePrediction);
    const lossHistory = [];

    const featureImportanceGain = {};
    state.features.forEach(f => featureImportanceGain[f] = 0);

    for (let round = 1; round <= state.modelParams.nEstimators; round++) {
        // Compute pseudo-residuals (Gradients for LogLoss)
        const residuals = state.trainData.map((row, idx) => {
            const prob = sigmoid(currentTrainPredictions[idx]);
            return row[state.target] - prob;
        });

        // Compute LogLoss
        const currentLoss = state.trainData.reduce((acc, row, idx) => {
            const p = Math.max(1e-15, Math.min(1 - 1e-15, sigmoid(currentTrainPredictions[idx])));
            return acc - (row[state.target] * Math.log(p) + (1 - row[state.target]) * Math.log(1 - p));
        }, 0) / state.trainData.length;
        
        lossHistory.push(currentLoss);

        // Subsample Rows
        const sampleSize = Math.floor(state.trainData.length * state.modelParams.subsample);
        const sampledX = [], sampledRes = [];
        for (let s = 0; s < sampleSize; s++) {
            const rIdx = Math.floor(Math.random() * state.trainData.length);
            sampledX.push(state.trainData[rIdx]);
            sampledRes.push(residuals[rIdx]);
        }

        // Fit Tree
        const tree = trainDecisionTree(sampledX, sampledRes, 0, state.modelParams.maxDepth, state.modelParams.colsampleBytree);
        trees.push(tree);

        // Update Tree Feature Importances
        accumulateFeatureImportance(tree, featureImportanceGain);

        // Update predictions
        state.trainData.forEach((row, idx) => {
            currentTrainPredictions[idx] += state.modelParams.learningRate * predictTree(tree, row);
        });

        if (round % Math.max(1, Math.floor(state.modelParams.nEstimators / 4)) === 0) {
            appendTerminal('main-terminal', ` [Round ${round}/${state.modelParams.nEstimators}] - Training LogLoss: ${currentLoss.toFixed(4)}`);
        }
    }

    state.trainedEnsemble = { trees, basePrediction };
    
    // 3. Evaluate Test Set
    animatePipelineStep(5);
    evaluateModel(lossHistory, featureImportanceGain);
    setExecutionStatus('COMPLETED');
}

function accumulateFeatureImportance(node, importanceMap) {
    if (!node || node.value !== null) return;
    if (node.feature) {
        importanceMap[node.feature] = (importanceMap[node.feature] || 0) + 1;
    }
    accumulateFeatureImportance(node.left, importanceMap);
    accumulateFeatureImportance(node.right, importanceMap);
}

// ==========================================
// 5. EVALUATION METRICS & CHARTS
// ==========================================
function evaluateModel(lossHistory, featureImportanceGain) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    const testProbabilities = [];

    state.testData.forEach(row => {
        let rawScore = state.trainedEnsemble.basePrediction;
        state.trainedEnsemble.trees.forEach(tree => {
            rawScore += state.modelParams.learningRate * predictTree(tree, row);
        });
        const prob = sigmoid(rawScore);
        const predClass = prob >= 0.5 ? 1 : 0;
        const actual = row[state.target];

        testProbabilities.push({ prob, actual });

        if (actual === 1 && predClass === 1) tp++;
        else if (actual === 0 && predClass === 1) fp++;
        else if (actual === 0 && predClass === 0) tn++;
        else if (actual === 1 && predClass === 0) fn++;
    });

    const accuracy = (tp + tn) / (tp + tn + fp + fn);
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
    const auc = calculateAUC(testProbabilities);

    // Update Card Values
    document.getElementById('metric-accuracy').innerText = (accuracy * 100).toFixed(1) + '%';
    document.getElementById('metric-precision').innerText = (precision * 100).toFixed(1) + '%';
    document.getElementById('metric-recall').innerText = (recall * 100).toFixed(1) + '%';
    document.getElementById('metric-f1').innerText = (f1 * 100).toFixed(1) + '%';
    document.getElementById('metric-auc').innerText = auc.toFixed(3);

    // Update Confusion Matrix
    document.getElementById('cm-tn').innerText = tn;
    document.getElementById('cm-fp').innerText = fp;
    document.getElementById('cm-fn').innerText = fn;
    document.getElementById('cm-tp').innerText = tp;

    appendTerminal('main-terminal', `✓ Evaluation Metrics Computed. Accuracy: ${(accuracy*100).toFixed(2)}%`, 'success');

    // Update Rendered Charts
    updateFeatureImportanceChart(featureImportanceGain);
    updateLossCurveChart(lossHistory);
    updateProbabilityChart(testProbabilities);
}

function calculateAUC(probs) {
    const sorted = [...probs].sort((a, b) => a.prob - b.prob);
    let n0 = 0, n1 = 0, rankSum = 0;
    sorted.forEach((item, idx) => {
        if (item.actual === 1) {
            n1++;
            rankSum += (idx + 1);
        } else {
            n0++;
        }
    });
    if (n0 === 0 || n1 === 0) return 0.5;
    return (rankSum - (n1 * (n1 + 1)) / 2) / (n0 * n1);
}

// Chart Renderers using Chart.js
function updateFeatureImportanceChart(importanceMap) {
    const ctx = document.getElementById('chart-feature-importance').getContext('2d');
    if (state.chartInstances.importance) state.chartInstances.importance.destroy();

    const sortedFeatures = Object.keys(importanceMap).sort((a,b) => importanceMap[b] - importanceMap[a]);
    const values = sortedFeatures.map(f => importanceMap[f]);

    state.chartInstances.importance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedFeatures,
            datasets: [{
                label: 'Relative Gain / Split Count',
                data: values,
                backgroundColor: 'rgba(59, 130, 246, 0.7)',
                borderColor: '#3b82f6',
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
                y: { ticks: { color: '#c9d1d9' }, grid: { display: false } }
            }
        }
    });
}

function updateLossCurveChart(lossHistory) {
    const ctx = document.getElementById('chart-loss-curve').getContext('2d');
    if (state.chartInstances.loss) state.chartInstances.loss.destroy();

    state.chartInstances.loss = new Chart(ctx, {
        type: 'line',
        data: {
            labels: lossHistory.map((_, i) => i + 1),
            datasets: [{
                label: 'LogLoss',
                data: lossHistory,
                borderColor: '#3fb950',
                backgroundColor: 'rgba(63, 185, 80, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
                y: { ticks: { color: '#c9d1d9' }, grid: { color: '#30363d' } }
            }
        }
    });
}

function updateProbabilityChart(probabilities) {
    const ctx = document.getElementById('chart-probability-dist').getContext('2d');
    if (state.chartInstances.prob) state.chartInstances.prob.destroy();

    const benignProbs = probabilities.filter(p => p.actual === 0).map(p => p.prob);
    const malignantProbs = probabilities.filter(p => p.actual === 1).map(p => p.prob);

    state.chartInstances.prob = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Benign (0)',
                    data: benignProbs.map((p, idx) => ({ x: p, y: idx % 10 })),
                    backgroundColor: 'rgba(63, 185, 80, 0.7)'
                },
                {
                    label: 'Malignant (1)',
                    data: malignantProbs.map((p, idx) => ({ x: p, y: (idx % 10) + 12 })),
                    backgroundColor: 'rgba(248, 81, 73, 0.7)'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#c9d1d9' } } },
            scales: {
                x: { title: { display: true, text: 'Predicted Probability', color: '#8b949e' }, ticks: { color: '#8b949e' }, grid: { color: '#30363d' } },
                y: { display: false }
            }
        }
    });
}

// ==========================================
// 6. LIVE PREDICTION PLAYGROUND
// ==========================================
function renderPlaygroundInputs() {
    const container = document.getElementById('playground-form');
    container.innerHTML = '';
    
    state.features.forEach(f => {
        const val = defaultFeatureDefaults[f] || 10.0;
        container.innerHTML += `
            <div class="playground-field">
                <label>${f}</label>
                <input type="number" step="any" id="input-${f}" value="${val}">
            </div>
        `;
    });
}

function predictCustomInput() {
    if (!state.trainedEnsemble) return;

    const inputRow = {};
    state.features.forEach(f => {
        const inputEl = document.getElementById(`input-${f}`);
        inputRow[f] = parseFloat(inputEl.value) || 0;
    });

    let rawScore = state.trainedEnsemble.basePrediction;
    state.trainedEnsemble.trees.forEach(tree => {
        rawScore += state.modelParams.learningRate * predictTree(tree, inputRow);
    });

    const prob = sigmoid(rawScore);
    const isMalignant = prob >= 0.5;

    const badge = document.getElementById('playground-prediction-badge');
    badge.innerText = isMalignant ? 'Malignant (Class 1)' : 'Benign (Class 0)';
    badge.className = `prediction-badge ${isMalignant ? 'malignant' : 'benign'}`;

    const confVal = (prob * 100).toFixed(1);
    document.getElementById('playground-confidence-val').innerText = `${confVal}%`;
    document.getElementById('playground-confidence-bar').style.width = `${confVal}%`;

    document.getElementById('playground-explanation').innerText = 
        `The model evaluates a ${(prob * 100).toFixed(1)}% likelihood of Malignancy based on the decision boundaries across all ${state.modelParams.nEstimators} trees.`;
}

// ==========================================
// 7. EVENT LISTENERS & UI HELPERS
// ==========================================
function initUIEventListeners() {
    // Hyperparameter Slider Binding
    bindSlider('param-n-estimators', 'val-n-estimators', 'nEstimators', true);
    bindSlider('param-learning-rate', 'val-learning-rate', 'learningRate', false);
    bindSlider('param-max-depth', 'val-max-depth', 'maxDepth', true);
    bindSlider('param-subsample', 'val-subsample', 'subsample', false);
    bindSlider('param-colsample', 'val-colsample', 'colsampleBytree', false);
    bindSlider('param-alpha', 'val-alpha', 'regAlpha', false);

    // Run & Reset Buttons
    document.getElementById('btn-run-model').addEventListener('click', runXGBoostPipeline);
    document.getElementById('btn-hero-run').addEventListener('click', () => {
        document.getElementById('lab').scrollIntoView({ behavior: 'smooth' });
        runXGBoostPipeline();
    });

    document.getElementById('btn-reset-params').addEventListener('click', resetParameters);
    document.getElementById('btn-hero-reset').addEventListener('click', resetParameters);

    // Table Search & Pagination
    document.getElementById('table-search').addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        state.tablePagination.filteredData = state.dataset.filter(row => 
            Object.values(row).some(val => String(val).toLowerCase().includes(query))
        );
        state.tablePagination.currentPage = 1;
        renderTable();
    });

    document.getElementById('table-class-filter').addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'all') state.tablePagination.filteredData = [...state.dataset];
        else state.tablePagination.filteredData = state.dataset.filter(r => String(r[state.target]) === val);
        state.tablePagination.currentPage = 1;
        renderTable();
    });

    document.getElementById('btn-prev-page').addEventListener('click', () => {
        if (state.tablePagination.currentPage > 1) {
            state.tablePagination.currentPage--;
            renderTable();
        }
    });

    document.getElementById('btn-next-page').addEventListener('click', () => {
        const maxPage = Math.ceil(state.tablePagination.filteredData.length / state.tablePagination.pageSize);
        if (state.tablePagination.currentPage < maxPage) {
            state.tablePagination.currentPage++;
            renderTable();
        }
    });

    // Custom Playground Predict
    document.getElementById('btn-predict-custom').addEventListener('click', predictCustomInput);

    // Copy Python Code
    document.getElementById('btn-copy-code').addEventListener('click', () => {
        const codeText = document.getElementById('python-code-content').innerText;
        navigator.clipboard.writeText(codeText);
        alert('Python code copied to clipboard!');
    });
}

function bindSlider(sliderId, valId, paramKey, isInt) {
    const slider = document.getElementById(sliderId);
    slider.addEventListener('input', (e) => {
        const val = isInt ? parseInt(e.target.value) : parseFloat(e.target.value);
        document.getElementById(valId).innerText = isInt ? val : val.toFixed(2);
        state.modelParams[paramKey] = val;
    });
}

function resetParameters() {
    document.getElementById('param-n-estimators').value = 30;
    document.getElementById('val-n-estimators').innerText = '30';
    document.getElementById('param-learning-rate').value = 0.10;
    document.getElementById('val-learning-rate').innerText = '0.10';
    document.getElementById('param-max-depth').value = 3;
    document.getElementById('val-max-depth').innerText = '3';
    
    state.modelParams = {
        nEstimators: 30, learningRate: 0.10, maxDepth: 3,
        subsample: 0.80, colsampleBytree: 0.80, regAlpha: 0.0
    };

    runXGBoostPipeline();
}

function animatePipelineStep(stepNum) {
    for (let i = 1; i <= 5; i++) {
        const el = document.getElementById(`step-${i}`);
        if (i === stepNum) el.classList.add('active');
        else el.classList.remove('active');
    }
}

function appendTerminal(elementId, text, styleClass = '') {
    const term = document.getElementById(elementId);
    const line = document.createElement('div');
    line.className = `term-line ${styleClass}`;
    line.innerText = text;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
}

function clearTerminal(elementId) {
    document.getElementById(elementId).innerHTML = '';
}

function setExecutionStatus(status) {
    const badge = document.getElementById('execution-status');
    badge.innerText = status;
    badge.className = `status-badge ${status.toLowerCase()}`;
}

function sleep(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
