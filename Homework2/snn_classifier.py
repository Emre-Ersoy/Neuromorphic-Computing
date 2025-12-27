import math
import random
import time
import pickle
import os
from flask import Flask, render_template_string, request, jsonify
from threading import Thread

# ==========================================
# PART 1: LIF NEURON (Leaky Integrate-and-Fire)
# ==========================================

class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron.
    Equation: v[t] = v[t-1] - (v[t-1]/tau) + Input
    """
    def __init__(self, tau=20.0, threshold=1.0, reset_val=0.0):
        self.v = reset_val
        self.tau = tau
        self.threshold = threshold
        self.reset_val = reset_val
        self.spike_count = 0
        
    def step(self, input_current, dt=1.0):
        # Apply leakage and integrate input current
        dv = (-(self.v) + input_current) / self.tau
        self.v += dv * dt
        
        is_spike = False
        track_v = self.v 
        
        if self.v >= self.threshold:
            self.v = self.reset_val # Reset voltage after spike
            self.spike_count += 1
            is_spike = True
        elif self.v < -5.0:
            self.v = -5.0 # Cap negative voltage to avoid infinite buildup
            track_v = self.v
            
        return is_spike, track_v
    
    def reset(self):
        self.v = self.reset_val
        self.spike_count = 0

class SpikingNeuralNetwork:
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        # Initialize weights randomly (-0.5 to 0.5)
        # Using negative weights effectively inhibits the 'Square' neuron when Triangle harmonics are weak.
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(num_outputs)] for _ in range(num_inputs)]
        
        # Neurons: Sine, Square, Triangle
        # Square neuron gets more energy, so we give it a higher threshold or different tau.
        self.neurons = [LIFNeuron(tau=15.0, threshold=2.5) for _ in range(num_outputs)]
        
    def forward(self, input_features, time_steps=50):
        for neuron in self.neurons:
            neuron.reset()
            
        voltage_history = [] 
        spike_history = [] # Keeping this if we need it for debugging later
            
        for t in range(time_steps):
            step_voltages = []
            step_spikes = []
            for j in range(self.num_outputs):
                input_current = 0.0
                for i in range(self.num_inputs):
                    # Current = Input * Weight
                    # Multiplied by gain (15.0) to ensure spikes occur
                    input_current += input_features[i] * self.weights[i][j] * 15.0 
                
                spike, v_val = self.neurons[j].step(input_current)
                step_spikes.append(1 if spike else 0)
                step_voltages.append(v_val)
                
            spike_history.append(step_spikes)
            voltage_history.append(step_voltages)
        
        return [n.spike_count for n in self.neurons], voltage_history

    def train_step(self, input_features, target_idx, learning_rate=0.05):
        spike_counts, _ = self.forward(input_features)
        
        total = sum(spike_counts)
        if total == 0: total = 1
        probs = [s/total for s in spike_counts]
        
        targets = [0.0] * self.num_outputs
        targets[target_idx] = 1.0
        
        # Delta Learning Rule
        for j in range(self.num_outputs):
            error = targets[j] - probs[j]
            for i in range(self.num_inputs):
                # Update weights only if the input was active enough to matter
                if input_features[i] > 0.001:
                    self.weights[i][j] += learning_rate * error * input_features[i]

# ==========================================
# PART 2: SIGNAL PROCESSING (Raw Ratios)
# ==========================================

def fft_recursive(x):
    N = len(x)
    if N <= 1: return x
    even = fft_recursive(x[0::2])
    odd =  fft_recursive(x[1::2])
    combined = [0] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        w = complex(math.cos(angle), math.sin(angle))
        t = w * odd[k]
        combined[k] = even[k] + t
        combined[k + N // 2] = even[k] - t
    return combined

def get_harmonics(signal):
    """
    Strictly extracts harmonic ratios relative to the Fundamental.
    We removed sqrt scaling to keep the contrast sharp between Square (1/3) and Triangle (1/9).
    """
    # 1. FFT
    complex_signal = [complex(val, 0) for val in signal]
    fft_res = fft_recursive(complex_signal)
    mags = [abs(val) for val in fft_res[:len(signal)//2]]
    
    search_limit = max(1, len(mags) // 2)
    valid_mags = mags[1:search_limit]
    
    if not valid_mags: return [0.0]*10
    
    max_val = max(valid_mags)
    if max_val == 0: return [0.0]*10
    
    f0_index = valid_mags.index(max_val) + 1
    
    features = []
    
    for h in range(1, 11): 
        target_idx = f0_index * h
        
        # Window to catch peak leaks
        window = 1 
        start = max(0, target_idx - window)
        end = min(len(mags), target_idx + window + 1)
        
        if start < len(mags):
            local_peak = max(mags[start:end])
        else:
            local_peak = 0.0
            
        ratio = local_peak / max_val
        features.append(ratio)
        
    return features

def generate_data(samples=512):
    data = []
    # Focus training on distinguishing these shapes
    for _ in range(150): 
        f = random.uniform(1.0, 8.0)
        phase = random.uniform(0, math.pi)
        
        # 1. Sine (Pure fundamental)
        wave = [math.sin((i/samples)*2*math.pi*f + phase) for i in range(samples)]
        data.append((wave, 0, "Sine"))
        
        # 2. Square (Strong odd harmonics: 1/3, 1/5...)
        wave_sq = []
        for i in range(samples):
            t = (i/samples)*f + phase
            val = 1.0 if (t % 1) < 0.5 else -1.0
            wave_sq.append(val)
        data.append((wave_sq, 1, "Square"))
        
        # 3. Triangle (Weak odd harmonics: 1/9, 1/25...)
        wave_tri = []
        for i in range(samples):
            t = (i/samples)*f + phase
            val = 4 * abs((t % 1) - 0.5) - 1
            wave_tri.append(val)
        data.append((wave_tri, 2, "Triangle"))
        
    return data

# ==========================================
# PART 3: FLASK APP
# ==========================================

app = Flask(__name__)

snn_model = None
is_trained = False
training_progress = 0
MODEL_FILE = "snn_lif_v4_fix.pkl"

def save_model():
    try:
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(snn_model, f)
    except: pass

def load_model():
    global snn_model, is_trained
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                snn_model = pickle.load(f)
            is_trained = True
            print("LIF Model Loaded.")
        except: pass

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LIF SNN Classifier</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, sans-serif; background: #f0f2f5; padding: 20px; color: #444; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; }
        .box { background: #fff; border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 8px; }
        input { width: 70%; padding: 12px; border: 1px solid #ccc; border-radius: 4px; font-size: 16px; }
        button { padding: 12px 24px; border: none; border-radius: 4px; color: white; cursor: pointer; font-weight: bold; font-size: 14px; transition: 0.2s; }
        .btn-train { background-color: #3498db; }
        .btn-train:hover { background-color: #2980b9; }
        .btn-anlz { background-color: #27ae60; }
        .btn-anlz:hover { background-color: #219150; }
        button:disabled { background-color: #95a5a6; cursor: not-allowed; }
        .results { display: flex; justify-content: space-between; gap: 10px; margin-top: 20px; }
        .card { flex: 1; text-align: center; padding: 15px; color: white; border-radius: 6px; }
        .c-sine { background-color: #e74c3c; }
        .c-sq { background-color: #f1c40f; color: #333; }
        .c-tri { background-color: #3498db; }
        #chart-wrap { height: 250px; margin-top: 20px; }
        .chart-section { display: flex; flex-direction: column; gap: 20px; margin-top: 20px; }
        .chart-box { height: 250px; position: relative; }
        h4.chart-title { margin: 0 0 5px 0; color: #7f8c8d; font-weight: 600; text-align: center; }
        .prediction { text-align: center; font-size: 1.5em; margin-top: 20px; color: #34495e; font-weight: bold; }
    </style>
</head>
<body>
<div class="container">
    <h1>LIF Spiking Neural Network</h1>
    
    <div class="box">
        <h3>1. Train Model</h3>
        <p>Training LIF neurons using raw harmonic ratios (helps fix Square vs Triangle overlap).</p>
        <button id="trainBtn" class="btn-train" onclick="runTrain()">Train LIF Network</button>
        <span id="status" style="margin-left:15px; font-weight:bold; color:#e67e22">Checking...</span>
        <div style="width:100%; background:#eee; height:10px; margin-top:10px; border-radius:5px; overflow:hidden">
            <div id="prog" style="width:0%; height:100%; background:#2ecc71; transition:width 0.3s"></div>
        </div>
    </div>

    <div class="box">
        <h3>2. Analyze Waveform</h3>
        <p>Test with: <code>4 * abs(((x / (2 * pi)) % 1) - 0.5) - 1</code></p>
        <div style="display:flex; gap:10px;">
            <input type="text" id="func" disabled placeholder="Enter Python math expression...">
            <button id="anlzBtn" class="btn-anlz" onclick="runAnalyze()" disabled>Analyze</button>
        </div>
        
        <div class="chart-section">
            <div>
                <h4 class="chart-title">Input Waveform</h4>
                <div class="chart-box"><canvas id="waveChart"></canvas></div>
            </div>
            <div>
                <h4 class="chart-title">Neuron Membrane Potentials (v) - Real-time Response</h4>
                <div class="chart-box"><canvas id="voltageChart"></canvas></div>
            </div>
        </div>
        
        <div class="results" id="resPanel" style="opacity:0.3; transition:opacity 0.5s">
            <div class="card c-sine">Sine<br><h2 id="sSine">0</h2></div>
            <div class="card c-sq">Square<br><h2 id="sSq">0</h2></div>
            <div class="card c-tri">Triangle<br><h2 id="sTri">0</h2></div>
        </div>
        
        <div class="prediction">Prediction: <span id="finalPred">-</span></div>
    </div>
</div>

<script>
    let waveChartRef = null;
    let voltageChartRef = null;
    let timer = null;

    window.onload = () => {
        const trained = {{ 'true' if trained else 'false' }};
        if(trained) enableUI();
    };

    function enableUI() {
        document.getElementById('status').innerText = "Model Ready";
        document.getElementById('status').style.color = "#27ae60";
        document.getElementById('func').disabled = false;
        document.getElementById('anlzBtn').disabled = false;
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('trainBtn').innerText = "Retrain";
        document.getElementById('prog').style.width = "100%";
    }

    function runTrain() {
        document.getElementById('trainBtn').disabled = true;
        document.getElementById('status').innerText = "Training...";
        document.getElementById('status').style.color = "#e67e22";
        fetch('/train', {method:'POST'});
        
        timer = setInterval(async () => {
            const r = await fetch('/progress');
            const d = await r.json();
            document.getElementById('prog').style.width = d.p + "%";
            if(d.done) { clearInterval(timer); enableUI(); }
        }, 300);
    }

    async function runAnalyze() {
        const f = document.getElementById('func').value;
        if(!f) return;
        
        try {
            const r = await fetch('/analyze', {
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body: JSON.stringify({f:f})
            });
            const d = await r.json();
            if(d.err) { alert(d.err); return; }
            
            document.getElementById('resPanel').style.opacity = 1;
            document.getElementById('sSine').innerText = d.spikes[0];
            document.getElementById('sSq').innerText = d.spikes[1];
            document.getElementById('sTri').innerText = d.spikes[2];
            document.getElementById('finalPred').innerText = d.pred;
            
            renderCharts(d.wave, d.history);
        } catch(e) { alert("Error connecting to server: " + e); }
    }

    function renderCharts(waveData, voltageHistory) {
        // 1. Input Waveform
        const ctxWave = document.getElementById('waveChart').getContext('2d');
        if(waveChartRef) waveChartRef.destroy();
        
        waveChartRef = new Chart(ctxWave, {
            type: 'line',
            data: {
                labels: waveData.map((_,i)=>i),
                datasets: [{
                    label: 'Input Signal',
                    data: waveData,
                    borderColor: '#34495e',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                }]
            },
            options: { 
                maintainAspectRatio: false, 
                scales:{x:{display:false}}, 
                plugins:{legend:{display:false}},
                interaction: {mode: 'index', intersect: false}
            }
        });

        // 2. Voltage Traces (Multi-line)
        const ctxVolt = document.getElementById('voltageChart').getContext('2d');
        if(voltageChartRef) voltageChartRef.destroy();
        
        // Transform history [time][neuron] -> datasets [series][time]
        const sineData = voltageHistory.map(step => step[0]);
        const squareData = voltageHistory.map(step => step[1]);
        const triData = voltageHistory.map(step => step[2]);
        const labels = voltageHistory.map((_,i) => i);

        voltageChartRef = new Chart(ctxVolt, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Sine Neuron (v)',
                        data: sineData,
                        borderColor: '#e74c3c', // Red
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.2 // Slight smooth
                    },
                    {
                        label: 'Square Neuron (v)',
                        data: squareData,
                        borderColor: '#f1c40f', // Yellow
                        backgroundColor: 'rgba(241, 196, 15, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.2
                    },
                    {
                        label: 'Triangle Neuron (v)',
                        data: triData,
                        borderColor: '#3498db', // Blue
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        pointRadius: 0,
                        tension: 0.2
                    }
                ]
            },
            options: {
                maintainAspectRatio: false,
                scales: {
                    x: { display: true, title: {display: true, text: 'Time Step'} },
                    y: { 
                        beginAtZero: false, 
                        title: {display: true, text: 'Voltage (v)'},
                        grid: { color: '#eee' }
                    }
                },
                plugins: {
                    legend: { display: true, position: 'top' },
                    tooltip: { mode: 'index', intersect: false }
                },
                elements: {
                    point: { radius: 0 } // Performance
                }
            }
        });
    }
</script>
</body>
</html>
"""

def train_bg():
    global snn_model, is_trained, training_progress
    # 10 Harmonic Inputs
    snn_model = SpikingNeuralNetwork(num_inputs=10, num_outputs=3)
    
    epochs = 300
    batch_data = generate_data(samples=512)
    
    for epoch in range(epochs):
        random.shuffle(batch_data)
        for wave, label_idx, _ in batch_data:
            feats = get_harmonics(wave)
            snn_model.train_step(feats, label_idx, learning_rate=0.03)
        training_progress = int((epoch/epochs)*100)
        
    is_trained = True
    training_progress = 100
    save_model()

@app.route('/')
def index():
    load_model()
    return render_template_string(HTML_TEMPLATE, trained=is_trained)

@app.route('/train', methods=['POST'])
def train():
    global is_trained, training_progress
    is_trained = False
    training_progress = 0
    Thread(target=train_bg).start()
    return jsonify({'ok':True})

@app.route('/progress')
def progress():
    return jsonify({'p':training_progress, 'done':is_trained})

@app.route('/analyze', methods=['POST'])
def analyze():
    global snn_model
    data = request.json
    f_str = data.get('f', '')
    
    safe_dict = {k:v for k,v in math.__dict__.items() if not k.startswith('_')}
    safe_dict['abs'] = abs
    safe_dict['pi'] = math.pi
    
    wave = []
    samples = 512
    try:
        for i in range(samples):
            # Scan multiple periods (4*pi) to catch waveforms properly
            x = (i/samples) * 4 * math.pi
            safe_dict['x'] = x
            val = eval(f_str, {"__builtins__":None}, safe_dict)
            wave.append(val)
            
        feats = get_harmonics(wave)
        spikes, history = snn_model.forward(feats)
        
        labels = ["Sine Wave", "Square Wave", "Triangle Wave"]
        
        # Decision Logic (Softmax-ish via Spike Counts)
        max_s = max(spikes)
        if max_s < 3:
            pred = "Unknown / Noise"
        else:
            # Tie-breaking logic if needed, but trained weights should handle it
            winner = spikes.index(max_s)
            pred = labels[winner]
            
            
        return jsonify({'wave':wave, 'spikes':spikes, 'pred':pred, 'history': history})
        
    except Exception as e:
        return jsonify({'err':str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)