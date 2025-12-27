import math
import random
import time
import pickle
import os
from flask import Flask, render_template_string, request, jsonify
from threading import Thread

# ==========================================
# PART 1: FFT LOGIC (Enhanced)
# ==========================================

def fft_recursive(x):
    """Basic Radix-2 FFT Algorithm"""
    N = len(x)
    if N <= 1:
        return x
    
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

def calculate_spectrum(signal):
    """Calculates the frequency spectrum of the signal"""
    complex_signal = [complex(val, 0) for val in signal]
    fft_result = fft_recursive(complex_signal)
    # Calculate magnitude
    magnitudes = [abs(val) for val in fft_result[:len(signal)//2]]
    return magnitudes

def extract_harmonic_features(signal):
    """
    Harmonic Analysis (Dynamic Windowing):
    Narrows the search window for low frequencies to prevent
    harmonics from mixing together.
    """
    spectrum = calculate_spectrum(signal)
    
    # Skip DC component (index 0)
    valid_spectrum = spectrum[1:] 
    if not valid_spectrum: return [0]*5
    
    # Find the Fundamental Frequency
    max_val = max(valid_spectrum)
    if max_val == 0: max_val = 1
    
    max_index = valid_spectrum.index(max_val)
    fundamental_idx = max_index + 1 
    
    features = []
    
    # Calculate ratio for the first 5 harmonics
    for i in range(1, 6):
        target_center = fundamental_idx * i
        
        # --- DYNAMIC SEARCH WINDOW ---
        # If fundamental frequency is low (index < 5), we narrow the window (1).
        # If high, we keep it wider (2) to catch frequency drifts.
        # Rule: Window size shouldn't exceed half the distance between harmonics.
        search_radius = max(1, min(2, int(fundamental_idx / 2)))
        
        start = max(0, target_center - search_radius)
        end = min(len(spectrum), target_center + search_radius + 1)
        
        if start < len(spectrum):
            local_max = max(spectrum[start:end]) if start < end else 0
            ratio = local_max / max_val
            features.append(ratio)
        else:
            features.append(0.0)
            
    return features

def generate_signals(samples=512):
    """Training Data - Extended Frequency Range"""
    data = []
    
    # 50 batches
    for _ in range(50): 
        # Extended frequency range: 1.0 to 20.0
        # This helps it recognize slower waves (like 2pi period) properly.
        f = random.uniform(1.0, 20.0)
        
        # Sine
        wave_sin = []
        phase = random.uniform(0, math.pi)
        for i in range(samples):
            t = (i / samples) * 2 * math.pi * f
            wave_sin.append(math.sin(t + phase))
        data.append((wave_sin, [1, 0, 0], "Sine"))
        
        # Square
        wave_sq = []
        phase = random.uniform(0, 1)
        for i in range(samples):
            t = (i / samples) * f + phase
            val = 1.0 if (t % 1) < 0.5 else -1.0
            wave_sq.append(val)
        data.append((wave_sq, [0, 1, 0], "Square"))
        
        # Triangle
        wave_tri = []
        phase = random.uniform(0, 1)
        for i in range(samples):
            t = (i / samples) * f + phase
            val = 4 * abs((t % 1) - 0.5) - 1
            wave_tri.append(val)
        data.append((wave_tri, [0, 0, 1], "Triangle"))
            
    return data

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.w1 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.w2 = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]
        self.hidden_outputs = []
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-max(-500, min(500, x))))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        self.hidden_outputs = []
        for j in range(len(self.b1)):
            activation = self.b1[j]
            for i in range(len(inputs)):
                activation += inputs[i] * self.w1[i][j]
            self.hidden_outputs.append(self.sigmoid(activation))
        
        final_outputs = []
        for j in range(len(self.b2)):
            activation = self.b2[j]
            for i in range(len(self.hidden_outputs)):
                activation += self.hidden_outputs[i] * self.w2[i][j]
            final_outputs.append(self.sigmoid(activation))
            
        return final_outputs
    
    def train(self, inputs, targets, learning_rate):
        outputs = self.forward(inputs)
        
        output_errors = [targets[i] - outputs[i] for i in range(len(targets))]
        output_deltas = [output_errors[i] * self.sigmoid_derivative(outputs[i]) for i in range(len(outputs))]
        
        hidden_errors = [0] * len(self.hidden_outputs)
        for i in range(len(self.hidden_outputs)):
            error = 0
            for j in range(len(output_deltas)):
                error += output_deltas[j] * self.w2[i][j]
            hidden_errors[i] = error
            
        hidden_deltas = [hidden_errors[i] * self.sigmoid_derivative(self.hidden_outputs[i]) for i in range(len(self.hidden_outputs))]
        
        for i in range(len(self.hidden_outputs)):
            for j in range(len(outputs)):
                self.w2[i][j] += learning_rate * output_deltas[j] * self.hidden_outputs[i]
        for j in range(len(outputs)):
            self.b2[j] += learning_rate * output_deltas[j]
            
        for i in range(len(inputs)):
            for j in range(len(self.hidden_outputs)):
                self.w1[i][j] += learning_rate * hidden_deltas[j] * inputs[i]
        for j in range(len(self.hidden_outputs)):
            self.b1[j] += learning_rate * hidden_deltas[j]

# ==========================================
# PART 2: FLASK APPLICATION
# ==========================================

app = Flask(__name__)

nn_model = None
is_trained = False
training_progress = 0
MODEL_FILENAME = "waveform_model_final.pkl" 

def save_model():
    global nn_model
    try:
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(nn_model, f)
        print(f"Model saved to {MODEL_FILENAME}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model():
    global nn_model, is_trained
    if os.path.exists(MODEL_FILENAME):
        try:
            with open(MODEL_FILENAME, 'rb') as f:
                nn_model = pickle.load(f)
            is_trained = True
            print(f"Model loaded from {MODEL_FILENAME}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    return False

load_model()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fourier Waveform Classifier (Final)</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; }
        .panel { border: 1px solid #ddd; padding: 20px; margin-bottom: 20px; border-radius: 8px; }
        .btn { padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: 0.3s; }
        .btn-primary { background-color: #2c3e50; color: white; }
        .btn-primary:hover { background-color: #34495e; }
        .btn-success { background-color: #27ae60; color: white; }
        .btn-success:hover { background-color: #219150; }
        .btn:disabled { background-color: #bdc3c7; cursor: not-allowed; }
        input[type="text"] { width: 70%; padding: 10px; font-size: 16px; border: 1px solid #ccc; border-radius: 5px; }
        .result-box { display: flex; justify-content: space-around; margin-top: 20px; }
        .score-card { text-align: center; padding: 15px; border-radius: 8px; width: 30%; color: white; }
        .sine-bg { background-color: #e74c3c; }
        .square-bg { background-color: #f1c40f; color: #333; }
        .triangle-bg { background-color: #3498db; }
        #chart-container { position: relative; height: 300px; width: 100%; margin-top: 20px; }
        .progress-container { width: 100%; background-color: #ecf0f1; border-radius: 10px; margin-top: 10px; height: 25px; overflow: hidden; display: none; }
        .progress-bar { height: 100%; background: linear-gradient(90deg, #2c3e50, #3498db); transition: width 0.3s; text-align: center; line-height: 25px; color: white; font-weight: bold; }
    </style>
</head>
<body>

<div class="container">
    <h1>Fourier Waveform Classifier (Final)</h1>
        
    <div class="panel">
        <h3>Step 1: Model Training (FFT)</h3>
        <p>Train the neural network using Fast Fourier Transform (Accuracy Optimized).</p>
        <button id="trainBtn" class="btn btn-primary" onclick="trainModel()">Train Model</button>
        <span id="trainStatus" style="color: #e74c3c; font-weight: bold; margin-left: 10px;">Not Trained</span>
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar" id="progressBar">0%</div>
        </div>
    </div>

    <div class="panel">
        <h3>Step 2: Function Analysis</h3>
        <p>Enter a mathematical function (e.g., <code>sin(x)</code>, <code>1 if (x % 1) < 0.5 else -1</code>)</p>
        <div style="display: flex; gap: 10px;">
            <input type="text" id="funcInput" placeholder="e.g., 5 * sin(2 * x)" disabled>
            <button id="analyzeBtn" class="btn btn-success" onclick="analyzeFunction()" disabled>Analyze</button>
        </div>
        <div id="errorMsg" style="color: red; margin-top: 10px;"></div>
    </div>

    <div class="panel" id="resultPanel" style="display: none;">
        <h3>Results</h3>
        <div id="chart-container">
            <canvas id="waveChart"></canvas>
        </div>
        
        <div class="result-box">
            <div class="score-card sine-bg">
                <h4>Sine</h4>
                <h2 id="sineScore">0%</h2>
            </div>
            <div class="score-card square-bg">
                <h4>Square</h4>
                <h2 id="squareScore">0%</h2>
            </div>
            <div class="score-card triangle-bg">
                <h4>Triangle</h4>
                <h2 id="triangleScore">0%</h2>
            </div>
        </div>
        <h3 style="text-align: center; margin-top: 20px;">Prediction: <span id="finalPrediction" style="color: #2c3e50;">-</span></h3>
    </div>
</div>

<script>
    let myChart = null;
    let progressInterval = null;

    window.onload = function() {
        const isModelReady = {{ 'true' if model_ready else 'false' }};
        
        if (isModelReady) {
            const status = document.getElementById('trainStatus');
            const btn = document.getElementById('trainBtn');
            status.innerText = "Model Loaded (Ready)";
            status.style.color = "#2ecc71";
            btn.innerText = "Retrain";
            document.getElementById('funcInput').disabled = false;
            document.getElementById('analyzeBtn').disabled = false;
        }
    };

    async function trainModel() {
        const btn = document.getElementById('trainBtn');
        const status = document.getElementById('trainStatus');
        const progressContainer = document.getElementById('progressContainer');
        const progressBar = document.getElementById('progressBar');
        
        btn.disabled = true;
        btn.innerText = "Training...";
        status.innerText = "Processing FFT Spectra...";
        status.style.color = "#f39c12";
        progressContainer.style.display = "block";
        progressBar.style.width = "0%";
        progressBar.innerText = "0%";

        fetch('/train', { method: 'POST' });

        progressInterval = setInterval(async () => {
            const response = await fetch('/progress');
            const data = await response.json();
            const progress = data.progress;
            
            progressBar.style.width = progress + "%";
            progressBar.innerText = progress + "%";
            
            if (data.epoch > 0) {
                status.innerText = `Training Loop: Epoch ${data.epoch}/1500`;
            }

            if (data.completed) {
                clearInterval(progressInterval);
                status.innerText = "FFT Model Trained!";
                status.style.color = "#2ecc71";
                btn.innerText = "Retrain";
                btn.disabled = false;
                document.getElementById('funcInput').disabled = false;
                document.getElementById('analyzeBtn').disabled = false;
                setTimeout(() => { progressContainer.style.display = "none"; }, 2000);
            }
        }, 200);
    }

    async function analyzeFunction() {
        const funcStr = document.getElementById('funcInput').value;
        const errorMsg = document.getElementById('errorMsg');
        
        if (!funcStr) return;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ function: funcStr })
            });
            const data = await response.json();

            if (data.error) {
                errorMsg.innerText = data.error;
                return;
            } else {
                errorMsg.innerText = "";
            }

            document.getElementById('resultPanel').style.display = 'block';
            document.getElementById('sineScore').innerText = (data.scores.Sine * 100).toFixed(1) + '%';
            document.getElementById('squareScore').innerText = (data.scores.Square * 100).toFixed(1) + '%';
            document.getElementById('triangleScore').innerText = (data.scores.Triangle * 100).toFixed(1) + '%';
            document.getElementById('finalPrediction').innerText = data.prediction;

            drawChart(data.wave_data);

        } catch (error) {
            errorMsg.innerText = "Server error!";
        }
    }

    function drawChart(dataPoints) {
        const ctx = document.getElementById('waveChart').getContext('2d');
        const labels = Array.from({length: dataPoints.length}, (_, i) => i);

        if (myChart) myChart.destroy();

        myChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Generated Waveform',
                    data: dataPoints,
                    borderColor: '#2c3e50',
                    backgroundColor: 'rgba(44, 62, 80, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { beginAtZero: false }
                }
            }
        });
    }
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, model_ready=is_trained)

def train_model_background():
    global nn_model, is_trained, training_progress
    
    # 5 harmonic features
    nn_model = NeuralNetwork(input_size=5, hidden_size=8, output_size=3)
    
    # High resolution (512 samples)
    raw_data = generate_signals(samples=512)
        
    # Data preprocessing (Feature extraction via FFT)
    processed_data = []
    total_samples = len(raw_data)
    
    for i, (wave, target, _) in enumerate(raw_data):
        features = extract_harmonic_features(wave)
        processed_data.append((features, target))
        if i % 50 == 0:
            training_progress = int((i / total_samples) * 10)
    
    # Training
    epochs = 1500
    for epoch in range(epochs):
        random.shuffle(processed_data)
        for features, target in processed_data:
            nn_model.train(features, target, learning_rate=0.1)
        
        training_progress = 10 + int((epoch + 1) / epochs * 90)
            
    is_trained = True
    save_model()

@app.route('/train', methods=['POST'])
def train():
    global training_progress
    training_progress = 0
    Thread(target=train_model_background, daemon=True).start()
    return jsonify({"success": True})

@app.route('/progress')
def progress():
    global training_progress, is_trained
    epoch = int((training_progress - 10) / 90 * 1500) if training_progress > 10 else 0
    return jsonify({"progress": training_progress, "completed": is_trained, "epoch": epoch})

@app.route('/analyze', methods=['POST'])
def analyze():
    global nn_model
    if not is_trained:
        return jsonify({"error": "Please train the model first!"})

    data = request.get_json()
    func_str = data.get('function', '')
    
    safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    safe_dict['pi'] = math.pi
    safe_dict['abs'] = abs
    
    generated_wave = []
    samples = 512
    
    try:
        for i in range(samples):
            x = (i / samples) * 4 * math.pi
            safe_dict['x'] = x
            val = eval(func_str, {"__builtins__": None}, safe_dict)
            generated_wave.append(val)
            
        # Use FFT features again
        features = extract_harmonic_features(generated_wave)
        prediction = nn_model.forward(features)
        
        labels = ["Sine", "Square", "Triangle"]
        max_score = max(prediction)
        pred_idx = prediction.index(max_score)
        
        return jsonify({
            "wave_data": generated_wave,
            "prediction": labels[pred_idx],
            "scores": {
                "Sine": prediction[0],
                "Square": prediction[1],
                "Triangle": prediction[2]
            }
        })
        
    except Exception as e:
        return jsonify({"error": f"Function error: {str(e)}"})

if __name__ == '__main__':
    print("Starting Flask application...")
    load_model() 
    app.run(debug=True)