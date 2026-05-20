/* =====================================================
   Emotion Recognition Dashboard - Application Logic
   Three.js background + data rendering + charts
   ===================================================== */

// ========================
// EMBEDDED DEMO DATA
// ========================
const DEMO_DATA = {
    metadata: {
        project: "Emotion Recognition from Speech",
        timestamp: new Date().toISOString(),
        data_mode: "synthetic",
        n_samples: 1200,
        n_features: 222,
        n_classes: 6,
        emotions: ["neutral", "happy", "sad", "angry", "fearful", "disgust"],
        split: { train: 840, val: 180, test: 180 },
        scaler: "StandardScaler (fit on train only)",
        tensorflow_available: false,
        librosa_available: false
    },
    models: {
        "MLP": {
            accuracy: 0.9389,
            f1_macro: 0.9388,
            f1_weighted: 0.9390,
            precision_macro: 0.9416,
            recall_macro: 0.9389,
            per_emotion_accuracy: {
                neutral: 0.9333, happy: 0.9667, sad: 0.9000,
                angry: 0.9667, fearful: 0.9333, disgust: 0.9333
            },
            confusion_matrix: [
                [28, 0, 1, 0, 1, 0],
                [0, 29, 0, 0, 1, 0],
                [1, 0, 27, 1, 0, 1],
                [0, 0, 1, 29, 0, 0],
                [0, 1, 0, 0, 28, 1],
                [0, 0, 1, 1, 0, 28]
            ],
            classification_report: {
                neutral:  { precision: 0.97, recall: 0.93, "f1-score": 0.95, support: 30 },
                happy:    { precision: 0.97, recall: 0.97, "f1-score": 0.97, support: 30 },
                sad:      { precision: 0.90, recall: 0.90, "f1-score": 0.90, support: 30 },
                angry:    { precision: 0.94, recall: 0.97, "f1-score": 0.95, support: 30 },
                fearful:  { precision: 0.93, recall: 0.93, "f1-score": 0.93, support: 30 },
                disgust:  { precision: 0.93, recall: 0.93, "f1-score": 0.93, support: 30 }
            },
            training_time: 4.2,
            epochs_trained: 127,
            n_test_samples: 180
        }
    },
    comparison: {
        models: [
            { name: "MLP", accuracy: 0.9389, f1_macro: 0.9388, precision_macro: 0.9416, recall_macro: 0.9389 }
        ],
        best_model: "MLP",
        best_accuracy: 0.9389
    },
    training_histories: {
        "MLP": {
            loss: [],
            val_loss: [],
            accuracy: [],
            val_accuracy: []
        }
    }
};

// Generate synthetic training history for demo
(function generateDemoHistory() {
    const epochs = 60;
    const h = DEMO_DATA.training_histories["MLP"];
    for (let i = 0; i < epochs; i++) {
        const t = i / (epochs - 1);
        h.accuracy.push(0.167 + (0.94 - 0.167) * (1 - Math.exp(-4 * t)) + (Math.random() - 0.5) * 0.02);
        h.val_accuracy.push(0.167 + (0.93 - 0.167) * (1 - Math.exp(-3.5 * t)) + (Math.random() - 0.5) * 0.03);
        h.loss.push(1.79 * Math.exp(-3.5 * t) + 0.18 + (Math.random() - 0.5) * 0.04);
        h.val_loss.push(1.79 * Math.exp(-3 * t) + 0.22 + (Math.random() - 0.5) * 0.05);
    }
})();

// ========================
// DATA LOADING
// ========================
let dashData = null;

async function loadData() {
    try {
        const resp = await fetch('../outputs/results.json');
        if (resp.ok) {
            dashData = await resp.json();
            console.log('[Dashboard] Loaded results.json');
        } else {
            throw new Error('results.json not found');
        }
    } catch (e) {
        console.log('[Dashboard] Using embedded demo data:', e.message);
        dashData = DEMO_DATA;
    }
    renderDashboard();
}

// ========================
// EMOTION CONFIG
// ========================
const EMOTION_EMOJI = {
    neutral: '😐', happy: '🤗', sad: '😢',
    angry: '😠', fearful: '😨', disgust: '🤮'
};

const EMOTION_COLORS = {
    neutral: '#94A3B8', happy: '#FBBF24', sad: '#60A5FA',
    angry: '#EF4444', fearful: '#A78BFA', disgust: '#22C55E'
};

const METRIC_COLORS = {
    accuracy: '#7C3AED',
    f1_macro: '#06B6D4',
    precision_macro: '#F59E0B',
    recall_macro: '#EC4899'
};

// ========================
// THREE.JS SOUND WAVE BACKGROUND
// ========================
function initThreeBackground() {
    const canvas = document.getElementById('bg-canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 30;
    camera.position.y = 5;

    // Particle wave system
    const particleCount = 4000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const colors = new Float32Array(particleCount * 3);

    const cols = 100;
    const rows = particleCount / cols;

    for (let i = 0; i < particleCount; i++) {
        const col = i % cols;
        const row = Math.floor(i / cols);
        positions[i * 3]     = (col - cols / 2) * 0.6;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = (row - rows / 2) * 0.6;

        // Purple to cyan gradient
        const t = col / cols;
        colors[i * 3]     = 0.486 * (1 - t) + 0.024 * t;  // R
        colors[i * 3 + 1] = 0.228 * (1 - t) + 0.714 * t;  // G
        colors[i * 3 + 2] = 0.929 * (1 - t) + 0.831 * t;  // B
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.12,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        sizeAttenuation: true
    });

    const particles = new THREE.Points(geometry, material);
    scene.add(particles);

    // Ambient light wave lines
    const lineCount = 5;
    const lines = [];
    for (let l = 0; l < lineCount; l++) {
        const lineGeo = new THREE.BufferGeometry();
        const linePositions = new Float32Array(200 * 3);
        lineGeo.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        const lineMat = new THREE.LineBasicMaterial({
            color: l % 2 === 0 ? 0x7C3AED : 0x06B6D4,
            transparent: true,
            opacity: 0.15
        });
        const line = new THREE.Line(lineGeo, lineMat);
        line.position.z = -10 + l * 3;
        line.position.y = -5 + l * 2;
        scene.add(line);
        lines.push(line);
    }

    let time = 0;

    function animate() {
        requestAnimationFrame(animate);
        time += 0.008;

        // Animate particle wave
        const pos = particles.geometry.attributes.position.array;
        for (let i = 0; i < particleCount; i++) {
            const x = pos[i * 3];
            const z = pos[i * 3 + 2];
            pos[i * 3 + 1] = Math.sin(x * 0.3 + time * 2) * Math.cos(z * 0.2 + time) * 2
                            + Math.sin(x * 0.5 + time * 1.5) * 0.8;
        }
        particles.geometry.attributes.position.needsUpdate = true;
        particles.rotation.y = time * 0.05;

        // Animate wave lines
        lines.forEach((line, l) => {
            const lPos = line.geometry.attributes.position.array;
            for (let i = 0; i < 200; i++) {
                lPos[i * 3] = (i - 100) * 0.5;
                lPos[i * 3 + 1] = Math.sin(i * 0.08 + time * 3 + l) * 3
                                 + Math.cos(i * 0.12 + time * 2 + l * 0.5) * 1.5;
                lPos[i * 3 + 2] = 0;
            }
            line.geometry.attributes.position.needsUpdate = true;
        });

        renderer.render(scene, camera);
    }

    animate();

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}

// ========================
// AUDIO WAVEFORM ANIMATION
// ========================
function initWaveform() {
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    let width, height;

    function resize() {
        const rect = canvas.parentElement.getBoundingClientRect();
        width = canvas.width = rect.width * 2;
        height = canvas.height = rect.height * 2;
        ctx.scale(1, 1);
    }
    resize();
    window.addEventListener('resize', resize);

    let t = 0;

    function draw() {
        requestAnimationFrame(draw);
        t += 0.02;

        ctx.clearRect(0, 0, width, height);

        // Draw multiple wave layers
        const layers = [
            { color: 'rgba(124, 58, 237, 0.5)', freq: 0.015, amp: 0.35, speed: 1 },
            { color: 'rgba(6, 182, 212, 0.4)', freq: 0.025, amp: 0.25, speed: 1.5 },
            { color: 'rgba(167, 139, 250, 0.3)', freq: 0.02, amp: 0.3, speed: 0.8 },
        ];

        layers.forEach(layer => {
            ctx.beginPath();
            ctx.moveTo(0, height / 2);

            for (let x = 0; x < width; x++) {
                const y = height / 2
                    + Math.sin(x * layer.freq + t * layer.speed) * height * layer.amp
                    + Math.sin(x * layer.freq * 2.5 + t * layer.speed * 1.3) * height * layer.amp * 0.3;
                ctx.lineTo(x, y);
            }

            ctx.strokeStyle = layer.color;
            ctx.lineWidth = 2;
            ctx.stroke();
        });

        // Draw faint center bar indicators
        const bars = 80;
        const barWidth = width / bars;
        for (let i = 0; i < bars; i++) {
            const barH = Math.abs(Math.sin(i * 0.15 + t * 2)) * height * 0.4
                       + Math.abs(Math.cos(i * 0.1 + t * 1.7)) * height * 0.1;
            const x = i * barWidth;
            const gradient = ctx.createLinearGradient(x, height / 2 - barH / 2, x, height / 2 + barH / 2);
            gradient.addColorStop(0, 'rgba(124, 58, 237, 0.15)');
            gradient.addColorStop(0.5, 'rgba(6, 182, 212, 0.2)');
            gradient.addColorStop(1, 'rgba(124, 58, 237, 0.15)');
            ctx.fillStyle = gradient;
            ctx.fillRect(x + 1, height / 2 - barH / 2, barWidth - 2, barH);
        }
    }

    draw();
}

// ========================
// RENDERING FUNCTIONS
// ========================
function renderDashboard() {
    const d = dashData;

    // Header
    document.getElementById('data-mode').textContent =
        d.metadata.data_mode === 'synthetic' ? '● Synthetic Demo' : '● RAVDESS Data';
    document.getElementById('timestamp').textContent =
        new Date(d.metadata.timestamp).toLocaleString();

    // Stats
    document.getElementById('stat-samples').textContent = d.metadata.n_samples.toLocaleString();
    document.getElementById('stat-accuracy').textContent =
        (d.comparison.best_accuracy * 100).toFixed(1) + '%';
    document.getElementById('stat-models').textContent = Object.keys(d.models).length;
    document.getElementById('stat-features').textContent = d.metadata.n_features;
    document.getElementById('stat-classes').textContent = d.metadata.n_classes;

    renderEmotionCards(d);
    renderModelComparison(d);
    renderTrainingCharts(d);
    renderConfusionMatrices(d);
    renderDetails(d);

    // Trigger bar animations after a short delay
    setTimeout(triggerAnimations, 300);
}

function renderEmotionCards(d) {
    const container = document.getElementById('emotion-cards');
    const bestModel = d.comparison.best_model;
    const perEmotion = d.models[bestModel].per_emotion_accuracy;

    container.innerHTML = d.metadata.emotions.map(emotion => {
        const acc = perEmotion[emotion] || 0;
        const pct = (acc * 100).toFixed(1);
        const color = EMOTION_COLORS[emotion];
        const emoji = EMOTION_EMOJI[emotion];

        return `
            <div class="emotion-card" style="--card-accent: ${color}">
                <span class="emotion-emoji">${emoji}</span>
                <div class="emotion-name">${emotion}</div>
                <div class="emotion-accuracy" style="color: ${color}">${pct}%</div>
                <div class="emotion-bar-track">
                    <div class="emotion-bar-fill" data-width="${pct}" style="background: ${color}"></div>
                </div>
            </div>
        `;
    }).join('');
}

function renderModelComparison(d) {
    const container = document.getElementById('model-comparison');
    const metrics = [
        { key: 'accuracy', label: 'Accuracy', color: METRIC_COLORS.accuracy },
        { key: 'f1_macro', label: 'F1 Macro', color: METRIC_COLORS.f1_macro },
        { key: 'precision_macro', label: 'Precision', color: METRIC_COLORS.precision_macro },
        { key: 'recall_macro', label: 'Recall', color: METRIC_COLORS.recall_macro }
    ];

    container.innerHTML = Object.entries(d.models).map(([name, model]) => {
        const isBest = name === d.comparison.best_model;
        const metricHTML = metrics.map(m => `
            <div class="metric-row">
                <span class="metric-label">${m.label}</span>
                <span class="metric-value" style="color: ${m.color}">${(model[m.key] * 100).toFixed(1)}%</span>
            </div>
            <div class="metric-bar-track">
                <div class="metric-bar-fill" data-width="${(model[m.key] * 100).toFixed(1)}"
                     style="background: linear-gradient(90deg, ${m.color}, ${m.color}88)"></div>
            </div>
        `).join('');

        return `
            <div class="model-card ${isBest ? 'best' : ''}">
                <div class="model-name">${name}</div>
                <div class="model-meta">
                    ${model.epochs_trained} epochs &bull; ${model.training_time.toFixed(1)}s &bull; ${model.n_test_samples} test samples
                </div>
                ${metricHTML}
            </div>
        `;
    }).join('');
}

function renderTrainingCharts(d) {
    const container = document.getElementById('charts-grid');

    container.innerHTML = Object.entries(d.training_histories).map(([name, hist]) => `
        <div class="chart-card">
            <h3>${name} — Accuracy</h3>
            <div class="chart-canvas-wrapper">
                <canvas id="chart-acc-${name}"></canvas>
            </div>
        </div>
        <div class="chart-card">
            <h3>${name} — Loss</h3>
            <div class="chart-canvas-wrapper">
                <canvas id="chart-loss-${name}"></canvas>
            </div>
        </div>
    `).join('');

    // Draw charts after DOM update
    requestAnimationFrame(() => {
        Object.entries(d.training_histories).forEach(([name, hist]) => {
            drawLineChart(`chart-acc-${name}`, hist.accuracy, hist.val_accuracy,
                'Train', 'Val', '#06B6D4', '#F59E0B', 0, 1);
            drawLineChart(`chart-loss-${name}`, hist.loss, hist.val_loss,
                'Train', 'Val', '#7C3AED', '#F59E0B');
        });
    });
}

function drawLineChart(canvasId, data1, data2, label1, label2, color1, color2, minY, maxY) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;

    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 30, right: 30, bottom: 40, left: 60 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Determine Y range
    const allVals = [...(data1 || []), ...(data2 || [])].filter(v => v !== undefined);
    if (allVals.length === 0) return;

    const yMin = minY !== undefined ? minY : Math.min(...allVals) * 0.9;
    const yMax = maxY !== undefined ? maxY : Math.max(...allVals) * 1.1;
    const n = Math.max(data1?.length || 0, data2?.length || 0);

    // Background
    ctx.fillStyle = 'rgba(11, 10, 26, 0.6)';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (plotH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
        ctx.stroke();

        const val = yMax - (yMax - yMin) * (i / 5);
        ctx.fillStyle = '#64748B';
        ctx.font = '20px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(2), pad.left - 8, y + 6);
    }

    // Axis label
    ctx.fillStyle = '#64748B';
    ctx.font = '20px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', w / 2, h - 5);

    // Plot lines
    function plotLine(data, color) {
        if (!data || data.length === 0) return;
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.lineJoin = 'round';

        data.forEach((v, i) => {
            const x = pad.left + (i / (n - 1)) * plotW;
            const y = pad.top + ((yMax - v) / (yMax - yMin)) * plotH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Gradient fill
        const lastX = pad.left + ((data.length - 1) / (n - 1)) * plotW;
        ctx.lineTo(lastX, pad.top + plotH);
        ctx.lineTo(pad.left, pad.top + plotH);
        ctx.closePath();
        const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
        grad.addColorStop(0, color + '30');
        grad.addColorStop(1, color + '05');
        ctx.fillStyle = grad;
        ctx.fill();
    }

    plotLine(data1, color1);
    plotLine(data2, color2);

    // Legend
    const legendY = 18;
    ctx.font = 'bold 20px Inter';
    [{ label: label1, color: color1, x: w - 220 }, { label: label2, color: color2, x: w - 110 }].forEach(l => {
        ctx.fillStyle = l.color;
        ctx.fillRect(l.x, legendY - 8, 16, 4);
        ctx.fillStyle = '#94A3B8';
        ctx.textAlign = 'left';
        ctx.fillText(l.label, l.x + 22, legendY);
    });
}

function renderConfusionMatrices(d) {
    const container = document.getElementById('confusion-grid');
    const emotions = d.metadata.emotions;

    container.innerHTML = Object.entries(d.models).map(([name, model]) => {
        const cm = model.confusion_matrix;
        const maxVal = Math.max(...cm.flat());

        const headerRow = `<tr><th></th>${emotions.map(e => `<th>${e.slice(0, 4)}</th>`).join('')}</tr>`;
        const rows = cm.map((row, i) => {
            const cells = row.map((val, j) => {
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const isDiag = i === j;
                let bg;
                if (isDiag) {
                    bg = `rgba(6, 182, 212, ${0.15 + intensity * 0.65})`;
                } else {
                    bg = val > 0
                        ? `rgba(239, 68, 68, ${0.1 + intensity * 0.5})`
                        : 'rgba(255, 255, 255, 0.03)';
                }
                return `<td style="background: ${bg}; color: ${val > 0 ? '#F1F5F9' : '#475569'}">${val}</td>`;
            }).join('');
            return `<tr><td class="row-label">${emotions[i].slice(0, 4)}</td>${cells}</tr>`;
        }).join('');

        return `
            <div class="confusion-card">
                <h3>${name} — Confusion Matrix</h3>
                <table class="confusion-table">
                    ${headerRow}
                    ${rows}
                </table>
            </div>
        `;
    }).join('');
}

function renderDetails(d) {
    // Data Split Bars
    const splitContainer = document.getElementById('split-bars');
    const total = d.metadata.n_samples;
    const splits = [
        { label: 'Train', count: d.metadata.split.train, color: '#7C3AED' },
        { label: 'Val', count: d.metadata.split.val, color: '#06B6D4' },
        { label: 'Test', count: d.metadata.split.test, color: '#F59E0B' }
    ];

    splitContainer.innerHTML = splits.map(s => {
        const pct = (s.count / total * 100).toFixed(0);
        return `
            <div class="split-bar-row">
                <span class="split-label">${s.label}</span>
                <div class="split-bar-track">
                    <div class="split-bar-fill" data-width="${pct}" style="background: ${s.color}">
                        ${s.count} (${pct}%)
                    </div>
                </div>
            </div>
        `;
    }).join('');

    // Classification Report Table
    const reportContainer = document.getElementById('report-table');
    const bestModel = d.comparison.best_model;
    const report = d.models[bestModel].classification_report;

    let tableHTML = `<table>
        <tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr>`;

    d.metadata.emotions.forEach(emotion => {
        const r = report[emotion];
        if (r) {
            tableHTML += `<tr>
                <td>${EMOTION_EMOJI[emotion]} ${emotion}</td>
                <td>${(r.precision).toFixed(2)}</td>
                <td>${(r.recall).toFixed(2)}</td>
                <td>${(r["f1-score"]).toFixed(2)}</td>
                <td>${r.support}</td>
            </tr>`;
        }
    });

    tableHTML += '</table>';
    reportContainer.innerHTML = tableHTML;
}

function triggerAnimations() {
    // Animate emotion bars
    document.querySelectorAll('.emotion-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
    });

    // Animate metric bars
    document.querySelectorAll('.metric-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
    });

    // Animate split bars
    document.querySelectorAll('.split-bar-fill').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
    });
}

// ========================
// LIVE AUDIO INFERENCE
// ========================
function initLiveAudio() {
    let mediaRecorder;
    let audioChunks = [];
    
    const btnRecord = document.getElementById('btn-record');
    const btnStop = document.getElementById('btn-stop');
    const statusDiv = document.getElementById('recording-status');
    const emotionEl = document.getElementById('live-emotion');
    const confidenceEl = document.getElementById('live-confidence');

    if (!btnRecord) return;

    btnRecord.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
            
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = async () => {
                statusDiv.style.display = 'none';
                emotionEl.textContent = 'ANALYZING...';
                confidenceEl.textContent = '';
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                audioChunks = [];
                
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.webm');
                
                try {
                    const response = await fetch('http://localhost:8002/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) throw new Error('API Error');
                    
                    const result = await response.json();
                    
                    emotionEl.textContent = result.prediction;
                    confidenceEl.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
                    
                    // Bounce animation
                    emotionEl.style.transform = 'scale(1.2)';
                    setTimeout(() => emotionEl.style.transform = 'scale(1)', 300);
                    
                } catch (error) {
                    console.error(error);
                    emotionEl.textContent = 'ERROR';
                    confidenceEl.textContent = 'API disconnected';
                }
            };
            
            mediaRecorder.start();
            statusDiv.style.display = 'block';
            btnRecord.disabled = true;
            btnRecord.style.background = '#333';
            btnRecord.style.color = '#888';
            btnRecord.style.cursor = 'not-allowed';
            
            btnStop.disabled = false;
            btnStop.style.background = 'linear-gradient(135deg, #00d4ff, #00ff88)';
            btnStop.style.color = '#000';
            btnStop.style.cursor = 'pointer';
            
        } catch (err) {
            console.error('Microphone error:', err);
            alert('Could not access microphone. Please allow permissions.');
        }
    });
    
    btnStop.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            // Stop tracks
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            
            btnStop.disabled = true;
            btnStop.style.background = '#333';
            btnStop.style.color = '#888';
            btnStop.style.cursor = 'not-allowed';
            
            btnRecord.disabled = false;
            btnRecord.style.background = 'linear-gradient(135deg, #ff007f, #7928ca)';
            btnRecord.style.color = '#fff';
            btnRecord.style.cursor = 'pointer';
        }
    });
}

// ========================
// INITIALIZATION
// ========================
document.addEventListener('DOMContentLoaded', () => {
    initThreeBackground();
    initWaveform();
    initLiveAudio();
    loadData();
});
