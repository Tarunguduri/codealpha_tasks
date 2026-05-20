/* ═══════════════════════════════════════════════════════════
   Credit Scoring Dashboard – app.js
   Three.js particle background + interactive charts
   ═══════════════════════════════════════════════════════════ */

// ── Demo Data (works standalone without results.json) ──────
const DEMO_DATA = {
    best_model: "Gradient Boosting",
    feature_names: [
        "age", "income", "debt", "loan_amount",
        "credit_history_years", "num_late_payments",
        "num_credit_lines", "employment_years",
        "debt_to_income_ratio"
    ],
    models: {
        "Logistic Regression": {
            accuracy: 0.7340,
            f1_score: 0.6927,
            roc_auc: 0.8022,
            cv_auc_mean: 0.7985,
            cv_auc_std: 0.0112,
            confusion_matrix: [[452, 114], [152, 282]],
            feature_importances: {
                age: 0.0612, income: 0.1823, debt: 0.0734, loan_amount: 0.0301,
                credit_history_years: 0.1452, num_late_payments: 0.2341,
                num_credit_lines: 0.0423, employment_years: 0.1012,
                debt_to_income_ratio: 0.1302
            },
            fpr: [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            tpr: [0, 0.18, 0.35, 0.48, 0.58, 0.7, 0.78, 0.85, 0.9, 0.94, 0.97, 0.99, 1.0]
        },
        "Decision Tree": {
            accuracy: 0.7220,
            f1_score: 0.6813,
            roc_auc: 0.7645,
            cv_auc_mean: 0.7523,
            cv_auc_std: 0.0198,
            confusion_matrix: [[438, 128], [150, 284]],
            feature_importances: {
                age: 0.0521, income: 0.1624, debt: 0.0893, loan_amount: 0.0412,
                credit_history_years: 0.1287, num_late_payments: 0.2563,
                num_credit_lines: 0.0612, employment_years: 0.0876,
                debt_to_income_ratio: 0.1212
            },
            fpr: [0, 0.06, 0.12, 0.18, 0.25, 0.33, 0.42, 0.53, 0.63, 0.73, 0.83, 0.92, 1.0],
            tpr: [0, 0.15, 0.30, 0.43, 0.54, 0.65, 0.74, 0.82, 0.88, 0.93, 0.96, 0.98, 1.0]
        },
        "Random Forest": {
            accuracy: 0.7580,
            f1_score: 0.7205,
            roc_auc: 0.8312,
            cv_auc_mean: 0.8245,
            cv_auc_std: 0.0095,
            confusion_matrix: [[468, 98], [144, 290]],
            feature_importances: {
                age: 0.0834, income: 0.1567, debt: 0.0976, loan_amount: 0.0523,
                credit_history_years: 0.1345, num_late_payments: 0.2012,
                num_credit_lines: 0.0723, employment_years: 0.0912,
                debt_to_income_ratio: 0.1108
            },
            fpr: [0, 0.03, 0.07, 0.12, 0.17, 0.25, 0.35, 0.45, 0.55, 0.65, 0.77, 0.88, 1.0],
            tpr: [0, 0.22, 0.40, 0.54, 0.64, 0.75, 0.83, 0.89, 0.93, 0.96, 0.98, 0.99, 1.0]
        },
        "Gradient Boosting": {
            accuracy: 0.7690,
            f1_score: 0.7312,
            roc_auc: 0.8456,
            cv_auc_mean: 0.8389,
            cv_auc_std: 0.0087,
            confusion_matrix: [[475, 91], [140, 294]],
            feature_importances: {
                age: 0.0645, income: 0.1734, debt: 0.0812, loan_amount: 0.0398,
                credit_history_years: 0.1512, num_late_payments: 0.2245,
                num_credit_lines: 0.0534, employment_years: 0.0923,
                debt_to_income_ratio: 0.1197
            },
            fpr: [0, 0.02, 0.05, 0.10, 0.15, 0.22, 0.32, 0.42, 0.52, 0.63, 0.75, 0.87, 1.0],
            tpr: [0, 0.25, 0.44, 0.58, 0.68, 0.78, 0.85, 0.91, 0.95, 0.97, 0.99, 1.0, 1.0]
        }
    }
};

const MODEL_COLORS = {
    "Logistic Regression": { main: "#3b82f6", rgb: "59,130,246" },
    "Decision Tree":       { main: "#f59e0b", rgb: "245,158,11" },
    "Random Forest":       { main: "#10b981", rgb: "16,185,129" },
    "Gradient Boosting":   { main: "#ef4444", rgb: "239,68,68" }
};

let DATA = DEMO_DATA;

// try loading results.json (optional – falls back to demo)
fetch("../outputs/results.json")
    .then(r => { if (r.ok) return r.json(); throw new Error(); })
    .then(d => { DATA = d; render(); })
    .catch(() => render());

// ── Three.js Particle Background ────────────────────────────
(function initThree() {
    const canvas = document.getElementById("bg-canvas");
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 50;

    // particles
    const count = 1800;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors    = new Float32Array(count * 3);

    const goldC    = new THREE.Color("#d4a017");
    const emeraldC = new THREE.Color("#10b981");
    const blueC    = new THREE.Color("#3b82f6");

    for (let i = 0; i < count; i++) {
        positions[i * 3]     = (Math.random() - 0.5) * 120;
        positions[i * 3 + 1] = (Math.random() - 0.5) * 120;
        positions[i * 3 + 2] = (Math.random() - 0.5) * 80;

        const pick = Math.random();
        const c = pick < 0.33 ? goldC : pick < 0.66 ? emeraldC : blueC;
        colors[i * 3]     = c.r;
        colors[i * 3 + 1] = c.g;
        colors[i * 3 + 2] = c.b;
    }

    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color",    new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
        size: 0.25,
        vertexColors: true,
        transparent: true,
        opacity: 0.6,
        sizeAttenuation: true,
    });

    const points = new THREE.Points(geo, mat);
    scene.add(points);

    let mouseX = 0, mouseY = 0;
    document.addEventListener("mousemove", e => {
        mouseX = (e.clientX / window.innerWidth - 0.5) * 2;
        mouseY = (e.clientY / window.innerHeight - 0.5) * 2;
    });

    window.addEventListener("resize", () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate() {
        requestAnimationFrame(animate);
        points.rotation.y += 0.0008;
        points.rotation.x += 0.0003;
        camera.position.x += (mouseX * 5 - camera.position.x) * 0.02;
        camera.position.y += (-mouseY * 5 - camera.position.y) * 0.02;
        camera.lookAt(scene.position);
        renderer.render(scene, camera);
    }
    animate();
})();

// ── Render Dashboard ────────────────────────────────────────
function render() {
    renderKPIs();
    renderModelCards();
    renderROC();
    renderFeatureImportance();
    renderConfusionMatrices();
    initScrollAnimations();
}

// ── KPIs ────────────────────────────────────────────────────
function renderKPIs() {
    const best = DATA.best_model;
    const m = DATA.models[best];
    document.getElementById("kpi-best-model").textContent = best;
    animateCounter("kpi-best-auc", m.roc_auc);
    animateCounter("kpi-best-f1", m.f1_score);
    animateCounter("kpi-best-acc", m.accuracy);
}

function animateCounter(id, target) {
    const el = document.getElementById(id);
    let current = 0;
    const step = target / 60;
    const interval = setInterval(() => {
        current += step;
        if (current >= target) { current = target; clearInterval(interval); }
        el.textContent = current.toFixed(4);
    }, 16);
}

// ── Model Cards ─────────────────────────────────────────────
function renderModelCards() {
    const grid = document.getElementById("model-grid");
    const metrics = [
        { key: "accuracy",    label: "Accuracy",   color: "#3b82f6" },
        { key: "f1_score",    label: "F1 Score",   color: "#10b981" },
        { key: "roc_auc",     label: "ROC-AUC",    color: "#d4a017" },
        { key: "cv_auc_mean", label: "CV AUC",     color: "#ef4444" },
    ];

    Object.entries(DATA.models).forEach(([name, m]) => {
        const isBest = name === DATA.best_model;
        const card = document.createElement("div");
        card.className = `model-card fade-in${isBest ? " best" : ""}`;

        let metricsHTML = "";
        metrics.forEach(({ key, label, color }) => {
            const pct = (m[key] * 100).toFixed(1);
            metricsHTML += `
                <div class="metric-row">
                    <span class="metric-label">${label}</span>
                    <span class="metric-value">${m[key].toFixed(4)}</span>
                </div>
                <div class="metric-bar-track">
                    <div class="metric-bar-fill" style="background:${color}" data-width="${pct}%"></div>
                </div>`;
        });

        if (m.cv_auc_std !== undefined) {
            metricsHTML += `
                <div class="metric-row" style="margin-top:8px">
                    <span class="metric-label">CV Std</span>
                    <span class="metric-value" style="color:#9ca3af">± ${m.cv_auc_std.toFixed(4)}</span>
                </div>`;
        }

        card.innerHTML = `
            <div class="model-name">${name}</div>
            <div class="model-badge">${isBest ? "⭐ BEST MODEL" : "CANDIDATE"}</div>
            ${metricsHTML}`;

        grid.appendChild(card);
    });

    // animate bars after a tick
    setTimeout(() => {
        document.querySelectorAll(".metric-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 300);
}

// ── ROC Curves (Canvas 2D) ──────────────────────────────────
function renderROC() {
    const canvas = document.getElementById("roc-canvas");
    const dpr = window.devicePixelRatio || 1;
    const W = 800, H = 500;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    const pad = { top: 30, right: 30, bottom: 60, left: 65 };
    const pw = W - pad.left - pad.right;
    const ph = H - pad.top - pad.bottom;

    // background
    ctx.fillStyle = "rgba(17,24,39,0.5)";
    ctx.fillRect(0, 0, W, H);

    // grid
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const x = pad.left + (pw * i / 10);
        const y = pad.top + (ph * i / 10);
        ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + ph); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pw, y); ctx.stroke();
    }

    // diagonal
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top + ph);
    ctx.lineTo(pad.left + pw, pad.top);
    ctx.stroke();
    ctx.setLineDash([]);

    // curves
    const legendItems = [];
    Object.entries(DATA.models).forEach(([name, m]) => {
        const col = MODEL_COLORS[name] || { main: "#fff" };
        ctx.strokeStyle = col.main;
        ctx.lineWidth = 2.5;
        ctx.shadowColor = col.main;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        m.fpr.forEach((fpr, i) => {
            const x = pad.left + fpr * pw;
            const y = pad.top + ph - m.tpr[i] * ph;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.shadowBlur = 0;
        legendItems.push({ name, color: col.main, auc: m.roc_auc });
    });

    // axis labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "13px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("False Positive Rate", pad.left + pw / 2, H - 12);

    ctx.save();
    ctx.translate(16, pad.top + ph / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("True Positive Rate", 0, 0);
    ctx.restore();

    // tick labels
    ctx.font = "11px Inter, sans-serif";
    ctx.fillStyle = "#6b7280";
    ctx.textAlign = "center";
    for (let i = 0; i <= 10; i += 2) {
        const v = (i / 10).toFixed(1);
        ctx.fillText(v, pad.left + pw * i / 10, pad.top + ph + 20);
        ctx.textAlign = "right";
        ctx.fillText(v, pad.left - 8, pad.top + ph - ph * i / 10 + 4);
        ctx.textAlign = "center";
    }

    // legend
    const lx = pad.left + pw - 200, ly = pad.top + ph - 20;
    ctx.fillStyle = "rgba(0,0,0,0.4)";
    ctx.roundRect(lx - 10, ly - legendItems.length * 22 - 5, 210, legendItems.length * 22 + 15, 8);
    ctx.fill();
    legendItems.forEach((item, i) => {
        const y = ly - (legendItems.length - 1 - i) * 22;
        ctx.fillStyle = item.color;
        ctx.fillRect(lx, y - 8, 14, 3);
        ctx.fillStyle = "#e5e7eb";
        ctx.font = "11px Inter, sans-serif";
        ctx.textAlign = "left";
        ctx.fillText(`${item.name} (AUC=${item.auc.toFixed(3)})`, lx + 20, y - 3);
    });
}

// ── Feature Importance ──────────────────────────────────────
function renderFeatureImportance() {
    const container = document.getElementById("feature-chart");
    const best = DATA.best_model;
    const imp = DATA.models[best].feature_importances;
    if (!imp) { container.innerHTML = "<p>No feature importance data available.</p>"; return; }

    const sorted = Object.entries(imp).sort((a, b) => b[1] - a[1]);
    const max = sorted[0][1];

    const viridis = [
        "#440154","#482878","#3e4989","#31688e","#26828e",
        "#1f9e89","#35b779","#6ece58","#b5de2b","#fde725"
    ];

    sorted.forEach(([feat, val], i) => {
        const pct = (val / max * 100).toFixed(1);
        const color = viridis[Math.min(i, viridis.length - 1)];
        const row = document.createElement("div");
        row.className = "feature-row fade-in";
        row.innerHTML = `
            <div class="feature-name">${feat}</div>
            <div class="feature-bar-track">
                <div class="feature-bar-fill" style="background:${color}" data-width="${pct}%">
                    ${(val * 100).toFixed(1)}%
                </div>
            </div>`;
        container.appendChild(row);
    });

    setTimeout(() => {
        container.querySelectorAll(".feature-bar-fill").forEach(bar => {
            bar.style.width = bar.dataset.width;
        });
    }, 400);
}

// ── Confusion Matrices ──────────────────────────────────────
function renderConfusionMatrices() {
    const grid = document.getElementById("cm-grid");

    Object.entries(DATA.models).forEach(([name, m]) => {
        const cm = m.confusion_matrix; // [[TN, FP], [FN, TP]]
        const total = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];
        const maxVal = Math.max(...cm.flat());

        const card = document.createElement("div");
        card.className = "cm-card fade-in";

        const cellColor = (val) => {
            const ratio = val / maxVal;
            const r = Math.round(16 + ratio * 40);
            const g = Math.round(185 * ratio);
            const b = Math.round(129 + ratio * 50);
            return `rgba(${r},${g},${b},${0.2 + ratio * 0.5})`;
        };

        card.innerHTML = `
            <h3>${name}</h3>
            <table class="cm-table">
                <tr><th></th><th>Pred Good</th><th>Pred Default</th></tr>
                <tr>
                    <th>Good</th>
                    <td class="cm-cell" style="background:${cellColor(cm[0][0])}">${cm[0][0]}</td>
                    <td class="cm-cell" style="background:${cellColor(cm[0][1])}">${cm[0][1]}</td>
                </tr>
                <tr>
                    <th>Default</th>
                    <td class="cm-cell" style="background:${cellColor(cm[1][0])}">${cm[1][0]}</td>
                    <td class="cm-cell" style="background:${cellColor(cm[1][1])}">${cm[1][1]}</td>
                </tr>
            </table>`;

        grid.appendChild(card);
    });
}

// ── Scroll Animations ───────────────────────────────────────
function initScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.15 });

    document.querySelectorAll(".fade-in").forEach(el => observer.observe(el));
}

// Live Prediction Logic
function initLivePrediction() {
    const form = document.getElementById('prediction-form');
    if (!form) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        document.getElementById('live-outcome').textContent = '...';
        document.getElementById('live-score').textContent = '...';
        document.getElementById('live-prob').textContent = '...';
        
        const payload = {
            age: parseFloat(document.getElementById('inp-age').value) || 0,
            income: parseFloat(document.getElementById('inp-income').value) || 0,
            debt: parseFloat(document.getElementById('inp-debt').value) || 0,
            loan_amount: parseFloat(document.getElementById('inp-loan').value) || 0,
            credit_history_years: parseFloat(document.getElementById('inp-history').value) || 0,
            num_late_payments: parseFloat(document.getElementById('inp-late').value) || 0,
            num_credit_lines: parseFloat(document.getElementById('inp-lines').value) || 0,
            employment_years: parseFloat(document.getElementById('inp-employment').value) || 0
        };

        try {
            const response = await fetch('http://localhost:8001/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) throw new Error("API error");

            const result = await response.json();
            
            const outcomeEl = document.getElementById('live-outcome');
            outcomeEl.textContent = result.prediction;
            outcomeEl.style.color = result.prediction === 'APPROVED' ? '#00ff88' : '#ff4d4d';
            
            document.getElementById('live-score').textContent = result.credit_score;
            document.getElementById('live-prob').textContent = (result.probability_of_default * 100).toFixed(1) + '%';
            
            // Bounce animation
            outcomeEl.style.transform = 'scale(1.2)';
            setTimeout(() => outcomeEl.style.transform = 'scale(1)', 300);

        } catch (err) {
            console.error(err);
            document.getElementById('live-outcome').textContent = 'ERROR';
            document.getElementById('live-outcome').style.color = '#ff4d4d';
            document.getElementById('live-score').textContent = 'API disconnected';
            document.getElementById('live-prob').textContent = 'API disconnected';
        }
    });
}

// Init everything
initLivePrediction();
