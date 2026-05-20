/* ═══════════════════════════════════════════════════════════════
   Neural Scribe — Dashboard App
   Three.js background, charts, interactive confusion matrix,
   drawing canvas, and embedded demo data
   ═══════════════════════════════════════════════════════════════ */

// ── Demo Data (embedded, replaced by results.json if available) ──
const DEMO_DATA = {
    project: "Handwritten Character Recognition",
    dataset: "MNIST",
    n_classes: 10,
    class_names: ["0","1","2","3","4","5","6","7","8","9"],
    train_samples: 60000,
    test_samples: 10000,
    validation_fraction: 0.1,
    epochs: 15,
    elapsed_seconds: 185.3,
    models: {
        SimpleCNN: {
            training: {
                model_name: "SimpleCNN",
                epochs_trained: 15,
                best_val_accuracy: 0.9938,
                history: {
                    accuracy: [0.9412,0.9785,0.9837,0.9862,0.9878,0.9890,0.9901,0.9908,0.9916,0.9922,0.9928,0.9932,0.9938,0.9940,0.9943],
                    val_accuracy: [0.9872,0.9900,0.9913,0.9920,0.9925,0.9930,0.9932,0.9935,0.9937,0.9935,0.9938,0.9937,0.9936,0.9938,0.9938],
                    loss: [0.1923,0.0672,0.0523,0.0446,0.0393,0.0354,0.0320,0.0299,0.0275,0.0258,0.0240,0.0226,0.0214,0.0203,0.0194],
                    val_loss: [0.0412,0.0310,0.0275,0.0255,0.0242,0.0228,0.0220,0.0215,0.0208,0.0212,0.0205,0.0210,0.0207,0.0204,0.0202]
                }
            },
            evaluation: {
                test_accuracy: 0.9928,
                test_loss: 0.0225,
                n_correct: 9928,
                n_incorrect: 72,
                confusion_matrix: [
                    [973,0,1,0,0,1,3,1,1,0],
                    [0,1130,2,0,0,1,1,0,1,0],
                    [1,1,1024,1,1,0,0,2,2,0],
                    [0,0,2,1003,0,2,0,1,1,1],
                    [0,0,1,0,976,0,2,0,0,3],
                    [1,0,0,3,0,885,1,0,1,1],
                    [3,1,0,0,1,2,950,0,1,0],
                    [0,2,4,1,0,0,0,1018,1,2],
                    [2,0,2,1,0,1,1,1,964,2],
                    [0,0,0,1,4,2,0,3,1,998]
                ]
            },
            architecture: {
                model_name: "SimpleCNN",
                total_params: 421898,
                n_layers: 10
            }
        },
        DeepCNN: {
            training: {
                model_name: "DeepCNN",
                epochs_trained: 15,
                best_val_accuracy: 0.9952,
                history: {
                    accuracy: [0.9523,0.9845,0.9882,0.9903,0.9915,0.9925,0.9932,0.9940,0.9945,0.9950,0.9953,0.9957,0.9960,0.9962,0.9965],
                    val_accuracy: [0.9905,0.9928,0.9935,0.9940,0.9943,0.9947,0.9948,0.9950,0.9951,0.9950,0.9952,0.9951,0.9950,0.9952,0.9951],
                    loss: [0.1532,0.0498,0.0382,0.0318,0.0275,0.0245,0.0220,0.0200,0.0183,0.0168,0.0155,0.0144,0.0135,0.0127,0.0120],
                    val_loss: [0.0305,0.0225,0.0200,0.0185,0.0175,0.0165,0.0160,0.0155,0.0152,0.0155,0.0150,0.0153,0.0152,0.0150,0.0151]
                }
            },
            evaluation: {
                test_accuracy: 0.9948,
                test_loss: 0.0172,
                n_correct: 9948,
                n_incorrect: 52,
                confusion_matrix: [
                    [977,0,0,0,0,1,1,1,0,0],
                    [0,1132,1,0,0,0,1,0,1,0],
                    [0,0,1028,1,0,0,0,2,1,0],
                    [0,0,1,1006,0,1,0,1,1,0],
                    [0,0,0,0,978,0,1,0,0,3],
                    [0,0,0,2,0,888,1,0,0,1],
                    [2,1,0,0,1,1,953,0,0,0],
                    [0,1,2,0,0,0,0,1023,0,2],
                    [1,0,1,1,0,0,0,0,969,2],
                    [0,0,0,0,3,1,0,2,0,1003]
                ]
            },
            architecture: {
                model_name: "DeepCNN",
                total_params: 237130,
                n_layers: 17
            }
        }
    },
    comparison: {
        best_model: "DeepCNN",
        best_accuracy: 0.9948,
        models: [
            { name: "SimpleCNN", accuracy: 0.9928, loss: 0.0225 },
            { name: "DeepCNN", accuracy: 0.9948, loss: 0.0172 }
        ]
    },
    sample_predictions: (function() {
        const samples = [];
        const digits = ["0","1","2","3","4","5","6","7","8","9"];
        for (let i = 0; i < 40; i++) {
            const trueLabel = digits[i % 10];
            const isIncorrect = i === 7 || i === 18 || i === 23 || i === 31 || i === 37;
            const wrongLabel = digits[(parseInt(trueLabel) + 1) % 10];
            samples.push({
                index: i,
                true_label: trueLabel,
                simple_pred: isIncorrect && i % 2 === 1 ? wrongLabel : trueLabel,
                deep_pred: isIncorrect && i % 2 === 0 ? wrongLabel : trueLabel,
                simple_correct: !(isIncorrect && i % 2 === 1),
                deep_correct: !(isIncorrect && i % 2 === 0)
            });
        }
        return samples;
    })()
};

let DATA = DEMO_DATA;

// ── Try to load results.json ──
(async function loadResults() {
    try {
        const resp = await fetch('../outputs/results.json');
        if (resp.ok) {
            DATA = await resp.json();
            console.log('[Dashboard] Loaded results.json');
        }
    } catch(e) {
        console.log('[Dashboard] Using embedded demo data');
    }
    initDashboard();
})();


// ═══════════════════════════════════════════════════════════════
//  THREE.JS BACKGROUND — Floating 3D Digits
// ═══════════════════════════════════════════════════════════════
function initThreeJS() {
    const canvas = document.getElementById('bg-canvas');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 30;

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x050510, 1);

    // Particles (star field)
    const starGeometry = new THREE.BufferGeometry();
    const starCount = 1500;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount * 3; i++) {
        starPositions[i] = (Math.random() - 0.5) * 100;
    }
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({
        color: 0x4444aa,
        size: 0.08,
        transparent: true,
        opacity: 0.6
    });
    const stars = new THREE.Points(starGeometry, starMaterial);
    scene.add(stars);

    // Floating digit meshes (wireframe boxes with glow)
    const digits = [];
    const digitColors = [0x00d4ff, 0xff6b9d, 0x9d4eff, 0x00ff88, 0xff9f43];

    for (let i = 0; i < 25; i++) {
        const size = 0.4 + Math.random() * 1.2;
        const geometries = [
            new THREE.BoxGeometry(size, size * 1.4, size * 0.3),
            new THREE.OctahedronGeometry(size * 0.6, 0),
            new THREE.TetrahedronGeometry(size * 0.7, 0),
            new THREE.IcosahedronGeometry(size * 0.5, 0),
        ];
        const geo = geometries[Math.floor(Math.random() * geometries.length)];
        const color = digitColors[Math.floor(Math.random() * digitColors.length)];

        const mat = new THREE.MeshBasicMaterial({
            color: color,
            wireframe: true,
            transparent: true,
            opacity: 0.15 + Math.random() * 0.15
        });

        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(
            (Math.random() - 0.5) * 50,
            (Math.random() - 0.5) * 35,
            (Math.random() - 0.5) * 20 - 10
        );
        mesh.userData = {
            rotSpeed: { x: (Math.random()-0.5)*0.008, y: (Math.random()-0.5)*0.012, z: (Math.random()-0.5)*0.005 },
            floatSpeed: 0.0005 + Math.random() * 0.002,
            floatOffset: Math.random() * Math.PI * 2,
            baseY: mesh.position.y
        };
        scene.add(mesh);
        digits.push(mesh);
    }

    // Ambient light
    const ambientLight = new THREE.AmbientLight(0x333366, 0.5);
    scene.add(ambientLight);

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        const t = Date.now() * 0.001;

        stars.rotation.y += 0.0001;
        stars.rotation.x += 0.00005;

        digits.forEach(d => {
            d.rotation.x += d.userData.rotSpeed.x;
            d.rotation.y += d.userData.rotSpeed.y;
            d.rotation.z += d.userData.rotSpeed.z;
            d.position.y = d.userData.baseY + Math.sin(t * d.userData.floatSpeed * 100 + d.userData.floatOffset) * 1.5;
        });

        renderer.render(scene, camera);
    }
    animate();

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
}


// ═══════════════════════════════════════════════════════════════
//  DRAWING CANVAS
// ═══════════════════════════════════════════════════════════════
function initDrawCanvas() {
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    let isDrawing = false;
    let lastX = 0, lastY = 0;

    ctx.fillStyle = '#0d0d20';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 12;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(0, 212, 255, 0.5)';
    ctx.shadowBlur = 8;

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const touch = e.touches ? e.touches[0] : e;
        return {
            x: (touch.clientX - rect.left) * (canvas.width / rect.width),
            y: (touch.clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPos(e);
        lastX = pos.x;
        lastY = pos.y;
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        lastX = pos.x;
        lastY = pos.y;
    }

    function stopDraw() { isDrawing = false; }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseleave', stopDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', draw, { passive: false });
    canvas.addEventListener('touchend', stopDraw);

    document.getElementById('btn-clear').addEventListener('click', () => {
        ctx.fillStyle = '#0d0d20';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        document.getElementById('result-digit').textContent = '?';
        document.getElementById('result-confidence').textContent = 'Draw a digit to begin';
    });

    document.getElementById('btn-predict').addEventListener('click', async () => {
        document.getElementById('result-digit').textContent = '...';
        document.getElementById('result-confidence').textContent = 'Analyzing...';
        
        try {
            // Get base64 string from canvas
            const dataUrl = canvas.toDataURL('image/png');
            
            // Send to FastAPI backend
            const response = await fetch('http://localhost:8003/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ image_base64: dataUrl })
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const result = await response.json();
            
            // Update UI
            document.getElementById('result-digit').textContent = result.prediction;
            const conf = (result.confidence * 100).toFixed(1);
            document.getElementById('result-confidence').textContent = `Confidence: ${conf}%`;
            
            // Animate
            const el = document.getElementById('result-digit');
            el.style.transform = 'scale(1.3)';
            setTimeout(() => el.style.transform = 'scale(1)', 300);
            
        } catch (error) {
            console.error('Error:', error);
            document.getElementById('result-digit').textContent = '!';
            document.getElementById('result-confidence').textContent = 'Error connecting to server. Is it running?';
        }
    });
}


// ═══════════════════════════════════════════════════════════════
//  MODEL COMPARISON (Accuracy Rings)
// ═══════════════════════════════════════════════════════════════
function initModelComparison() {
    const simple = DATA.models.SimpleCNN;
    const deep = DATA.models.DeepCNN;
    const circumference = 2 * Math.PI * 52; // 326.73

    // Simple CNN
    const simpleAcc = simple.evaluation.test_accuracy;
    const simpleOffset = circumference * (1 - simpleAcc);
    setTimeout(() => {
        document.querySelector('.ring-simple').style.strokeDashoffset = simpleOffset;
    }, 500);
    document.getElementById('simple-acc-value').textContent = (simpleAcc * 100).toFixed(2) + '%';
    document.getElementById('simple-loss').textContent = simple.evaluation.test_loss.toFixed(4);
    document.getElementById('simple-params').textContent = simple.architecture.total_params.toLocaleString();
    document.getElementById('simple-epochs').textContent = simple.training.epochs_trained;

    // Deep CNN
    const deepAcc = deep.evaluation.test_accuracy;
    const deepOffset = circumference * (1 - deepAcc);
    setTimeout(() => {
        document.querySelector('.ring-deep').style.strokeDashoffset = deepOffset;
    }, 700);
    document.getElementById('deep-acc-value').textContent = (deepAcc * 100).toFixed(2) + '%';
    document.getElementById('deep-loss').textContent = deep.evaluation.test_loss.toFixed(4);
    document.getElementById('deep-params').textContent = deep.architecture.total_params.toLocaleString();
    document.getElementById('deep-epochs').textContent = deep.training.epochs_trained;

    // Winner
    const banner = document.getElementById('winner-banner');
    banner.textContent = `🏆 ${DATA.comparison.best_model} wins with ${(DATA.comparison.best_accuracy * 100).toFixed(2)}% accuracy`;
    banner.classList.add('show');
}


// ═══════════════════════════════════════════════════════════════
//  TRAINING CURVES (Canvas chart)
// ═══════════════════════════════════════════════════════════════
let currentChart = 'accuracy';

function drawTrainingChart(type) {
    const canvas = document.getElementById('training-chart');
    const ctx = canvas.getContext('2d');
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width - 40;
    canvas.height = rect.height - 40;

    const W = canvas.width, H = canvas.height;
    const pad = { top: 30, right: 30, bottom: 40, left: 60 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    const simpleH = DATA.models.SimpleCNN.training.history;
    const deepH = DATA.models.DeepCNN.training.history;

    const key = type === 'accuracy' ? 'accuracy' : 'loss';
    const valKey = type === 'accuracy' ? 'val_accuracy' : 'val_loss';

    const allVals = [...simpleH[key], ...simpleH[valKey], ...deepH[key], ...deepH[valKey]];
    let minV = Math.min(...allVals);
    let maxV = Math.max(...allVals);
    const range = maxV - minV;
    minV -= range * 0.05;
    maxV += range * 0.05;

    const epochs = Math.max(simpleH[key].length, deepH[key].length);

    // Grid
    ctx.strokeStyle = 'rgba(50,50,100,0.3)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
        const y = pad.top + (plotH / 5) * i;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + plotW, y);
        ctx.stroke();

        const val = maxV - (maxV - minV) * (i / 5);
        ctx.fillStyle = '#666699';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText(val.toFixed(4), pad.left - 8, y + 4);
    }

    // X axis labels
    ctx.textAlign = 'center';
    for (let i = 0; i < epochs; i++) {
        const x = pad.left + (plotW / (epochs - 1)) * i;
        ctx.fillStyle = '#666699';
        ctx.fillText(i + 1, x, H - pad.bottom + 20);
    }
    ctx.fillText('Epoch', pad.left + plotW / 2, H - 5);

    // Y axis label
    ctx.save();
    ctx.translate(15, pad.top + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#9999cc';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    ctx.fillText(type === 'accuracy' ? 'Accuracy' : 'Loss', 0, 0);
    ctx.restore();

    function drawLine(data, color, dashed) {
        if (!data || data.length === 0) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2.5;
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.beginPath();
        data.forEach((val, i) => {
            const x = pad.left + (plotW / (data.length - 1)) * i;
            const y = pad.top + plotH * (1 - (val - minV) / (maxV - minV));
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);

        // Dots
        data.forEach((val, i) => {
            const x = pad.left + (plotW / (data.length - 1)) * i;
            const y = pad.top + plotH * (1 - (val - minV) / (maxV - minV));
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
        });
    }

    drawLine(simpleH[key], '#00d4ff', false);
    drawLine(simpleH[valKey], '#00d4ff', true);
    drawLine(deepH[key], '#ff6b9d', false);
    drawLine(deepH[valKey], '#ff6b9d', true);

    // Legend
    const legendY = pad.top - 10;
    const legends = [
        { label: 'Simple Train', color: '#00d4ff', dashed: false },
        { label: 'Simple Val', color: '#00d4ff', dashed: true },
        { label: 'Deep Train', color: '#ff6b9d', dashed: false },
        { label: 'Deep Val', color: '#ff6b9d', dashed: true },
    ];

    let lx = pad.left;
    legends.forEach(l => {
        ctx.strokeStyle = l.color;
        ctx.lineWidth = 2;
        ctx.setLineDash(l.dashed ? [4, 3] : []);
        ctx.beginPath();
        ctx.moveTo(lx, legendY);
        ctx.lineTo(lx + 20, legendY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = '#ccccee';
        ctx.font = '10px Inter';
        ctx.textAlign = 'left';
        ctx.fillText(l.label, lx + 24, legendY + 3);
        lx += 100;
    });
}

function initTrainingCharts() {
    drawTrainingChart('accuracy');

    document.querySelectorAll('#section-training .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('#section-training .tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentChart = tab.dataset.tab;
            drawTrainingChart(currentChart);
        });
    });

    window.addEventListener('resize', () => drawTrainingChart(currentChart));
}


// ═══════════════════════════════════════════════════════════════
//  CONFUSION MATRIX (Interactive heatmap)
// ═══════════════════════════════════════════════════════════════
let currentCMModel = 'SimpleCNN';

function renderConfusionMatrix(modelName) {
    const cm = DATA.models[modelName].evaluation.confusion_matrix;
    const labels = DATA.class_names;
    const n = labels.length;
    const grid = document.getElementById('confusion-grid');
    grid.innerHTML = '';
    grid.style.gridTemplateColumns = `40px repeat(${n}, 1fr)`;

    // Find max for color scaling
    let maxVal = 0;
    cm.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));

    // Top-left empty
    grid.appendChild(createEl('div', 'cm-header', ''));

    // Column headers
    labels.forEach(l => {
        grid.appendChild(createEl('div', 'cm-header', l));
    });

    // Rows
    cm.forEach((row, i) => {
        // Row header
        grid.appendChild(createEl('div', 'cm-header', labels[i]));
        row.forEach((val, j) => {
            const cell = document.createElement('div');
            cell.className = 'cm-cell';
            cell.textContent = val;

            // Color interpolation
            const intensity = val / maxVal;
            const isDiagonal = i === j;
            if (isDiagonal) {
                cell.style.background = `rgba(0, 212, 255, ${0.1 + intensity * 0.6})`;
                cell.style.color = intensity > 0.5 ? '#fff' : '#aaa';
            } else {
                cell.style.background = val > 0
                    ? `rgba(255, 68, 102, ${0.1 + (val / maxVal) * 0.7})`
                    : 'rgba(20,20,40,0.3)';
                cell.style.color = val > 0 ? '#fff' : '#444';
            }

            // Tooltip
            cell.addEventListener('mouseenter', (e) => {
                const tooltip = document.getElementById('confusion-tooltip');
                tooltip.style.display = 'block';
                tooltip.innerHTML = `
                    <strong>True:</strong> ${labels[i]}<br>
                    <strong>Predicted:</strong> ${labels[j]}<br>
                    <strong>Count:</strong> ${val}<br>
                    ${isDiagonal ? '✅ Correct' : val > 0 ? '❌ Misclassified' : '—'}
                `;
                tooltip.style.left = e.clientX + 15 + 'px';
                tooltip.style.top = e.clientY - 10 + 'px';
            });
            cell.addEventListener('mouseleave', () => {
                document.getElementById('confusion-tooltip').style.display = 'none';
            });

            grid.appendChild(cell);
        });
    });
}

function createEl(tag, cls, text) {
    const el = document.createElement(tag);
    el.className = cls;
    el.textContent = text;
    return el;
}

function initConfusionMatrix() {
    renderConfusionMatrix('SimpleCNN');

    document.querySelectorAll('#section-confusion .tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('#section-confusion .tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentCMModel = tab.dataset.model;
            renderConfusionMatrix(currentCMModel);
        });
    });
}


// ═══════════════════════════════════════════════════════════════
//  SAMPLE PREDICTIONS GALLERY
// ═══════════════════════════════════════════════════════════════
function initPredictions() {
    const grid = document.getElementById('predictions-grid');
    grid.innerHTML = '';

    DATA.sample_predictions.forEach(sample => {
        const isCorrect = sample.deep_correct;
        const card = document.createElement('div');
        card.className = `pred-card ${isCorrect ? 'correct' : 'incorrect'}`;
        card.innerHTML = `
            <div class="pred-digit">${sample.true_label}</div>
            <div class="pred-info">
                <div>True: <strong>${sample.true_label}</strong></div>
                <div>Simple: <strong>${sample.simple_pred}</strong> ${sample.simple_correct ? '✅' : '❌'}</div>
                <div>Deep: <strong>${sample.deep_pred}</strong> ${sample.deep_correct ? '✅' : '❌'}</div>
            </div>
        `;
        grid.appendChild(card);
    });
}


// ═══════════════════════════════════════════════════════════════
//  ARCHITECTURE DETAILS
// ═══════════════════════════════════════════════════════════════
function initArchitecture() {
    const simpleLayers = [
        { name: 'Conv2D(32, 3×3) + ReLU', params: '320' },
        { name: 'BatchNormalization', params: '128' },
        { name: 'MaxPooling2D(2×2)', params: '0' },
        { name: 'Dropout(0.25)', params: '0' },
        { name: 'Conv2D(64, 3×3) + ReLU', params: '18,496' },
        { name: 'BatchNormalization', params: '256' },
        { name: 'MaxPooling2D(2×2)', params: '0' },
        { name: 'Dropout(0.25)', params: '0' },
        { name: 'Flatten', params: '0' },
        { name: 'Dense(128) + ReLU', params: '401,536' },
        { name: 'Dropout(0.4)', params: '0' },
        { name: 'Dense(10) + Softmax', params: '1,290' },
    ];

    const deepLayers = [
        { name: 'Conv2D(32, 3×3) + ReLU', params: '320' },
        { name: 'Conv2D(32, 3×3) + ReLU', params: '9,248' },
        { name: 'BatchNormalization', params: '128' },
        { name: 'MaxPooling2D(2×2)', params: '0' },
        { name: 'Dropout(0.25)', params: '0' },
        { name: 'Conv2D(64, 3×3) + ReLU', params: '18,496' },
        { name: 'Conv2D(64, 3×3) + ReLU', params: '36,928' },
        { name: 'BatchNormalization', params: '256' },
        { name: 'MaxPooling2D(2×2)', params: '0' },
        { name: 'Dropout(0.25)', params: '0' },
        { name: 'Conv2D(128, 3×3) + ReLU', params: '73,856' },
        { name: 'BatchNormalization', params: '512' },
        { name: 'GlobalAveragePooling2D', params: '0' },
        { name: 'Dropout(0.4)', params: '0' },
        { name: 'Dense(256) + ReLU', params: '33,024' },
        { name: 'Dense(10) + Softmax', params: '2,570' },
    ];

    function renderLayers(containerId, layers) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        layers.forEach(l => {
            const div = document.createElement('div');
            div.className = 'arch-layer';
            div.innerHTML = `<span class="layer-name">${l.name}</span><span class="layer-params">${l.params}</span>`;
            container.appendChild(div);
        });
    }

    renderLayers('arch-simple', simpleLayers);
    renderLayers('arch-deep', deepLayers);
}


// ═══════════════════════════════════════════════════════════════
//  HEADER STATS
// ═══════════════════════════════════════════════════════════════
function initHeaderStats() {
    document.getElementById('stat-dataset').textContent = DATA.dataset;
    document.getElementById('stat-classes').textContent = DATA.n_classes;
    document.getElementById('stat-samples').textContent = (DATA.train_samples + DATA.test_samples).toLocaleString();
    document.getElementById('stat-best-acc').textContent = (DATA.comparison.best_accuracy * 100).toFixed(2) + '%';
}


// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════
function initDashboard() {
    initHeaderStats();
    initModelComparison();
    initTrainingCharts();
    initConfusionMatrix();
    initPredictions();
    initArchitecture();
}

// Start Three.js immediately
initThreeJS();
initDrawCanvas();
