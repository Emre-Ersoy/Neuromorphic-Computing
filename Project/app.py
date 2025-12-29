from flask import Flask, render_template, jsonify, request
from snn_core import snn_model
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Store last prediction for feedback
last_prediction = {
    'pixels': None,
    'predicted_label': None
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current model state for page initialization"""
    stats = snn_model.get_stats()
    total = sum(stats.values())
    return jsonify({
        'mapping': snn_model.label_map,
        'labels': snn_model.get_labels(),
        'trained_count': total,
        'stats': stats
    })

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    pixels = data.get('pixels', [])
    label = data.get('label', '0').strip().upper()
    
    spike_counts = snn_model.train_step(pixels, label)
    snn_model.save()
    
    _, voltage_history, spike_times, input_spike_times = snn_model.predict(pixels)[1:]
    results = format_spike_data(spike_times, input_spike_times, voltage_history)
    
    stats = snn_model.get_stats()
    
    return jsonify({
        'status': 'trained', 
        'label': label,
        'mapping': snn_model.label_map,
        'stats': stats,
        'results': results
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    global last_prediction
    data = request.json
    pixels = data.get('pixels', [])
    
    winner_label, spike_counts, voltage_history, spike_times, input_spike_times = snn_model.predict(pixels)
    
    # Store for feedback
    last_prediction['pixels'] = pixels
    last_prediction['predicted_label'] = winner_label
    
    results = format_spike_data(spike_times, input_spike_times, voltage_history)
    
    return jsonify({
        'status': 'predicted',
        'winner': winner_label,
        'mapping': snn_model.get_labels(),
        'spike_counts': spike_counts,
        'results': results
    })

@app.route('/api/word_test', methods=['POST'])
def word_test():
    """
    Word recognition: receive image data, segment into characters, predict each.
    """
    data = request.json
    segments = data.get('segments', [])  # List of 16x16 pixel arrays
    
    if not segments:
        return jsonify({'status': 'error', 'message': 'No segments provided'})
    
    # Predict each segment
    predictions = snn_model.predict_word(segments)
    word = ''.join(predictions)
    
    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'word': word
    })

@app.route('/api/feedback/correct', methods=['POST'])
def feedback_correct():
    global last_prediction
    
    if last_prediction['pixels'] is None or last_prediction['predicted_label'] is None:
        return jsonify({'status': 'error', 'message': 'No prediction to reinforce'})
    
    success = snn_model.reinforce_correct(
        last_prediction['pixels'],
        last_prediction['predicted_label']
    )
    
    return jsonify({
        'status': 'reinforced' if success else 'error',
        'label': last_prediction['predicted_label'],
        'stats': snn_model.get_stats()
    })

@app.route('/api/feedback/wrong', methods=['POST'])
def feedback_wrong():
    global last_prediction
    data = request.json
    correct_label = data.get('correct_label', '').strip().upper()
    
    if not correct_label:
        return jsonify({'status': 'error', 'message': 'No correct label provided'})
    
    if last_prediction['pixels'] is None:
        return jsonify({'status': 'error', 'message': 'No prediction to correct'})
    
    success = snn_model.correct_wrong(
        last_prediction['pixels'],
        last_prediction['predicted_label'],
        correct_label
    )
    
    return jsonify({
        'status': 'corrected' if success else 'error',
        'wrong_label': last_prediction['predicted_label'],
        'correct_label': correct_label,
        'mapping': snn_model.label_map,
        'stats': snn_model.get_stats()
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    snn_model.reset()
    return jsonify({'status': 'reset'})

def format_spike_data(spike_times, input_spike_times, voltage_history):
    input_t = []
    input_i = []
    for neuron_idx, times in input_spike_times.items():
        for t in times:
            input_t.append(float(t) / 100.0)
            input_i.append(int(neuron_idx))
    input_spikes = {'t': input_t, 'i': input_i}
    
    output_t = []
    output_i = []
    for neuron_idx, times in spike_times.items():
        for t in times:
            output_t.append(float(t) / 100.0)
            output_i.append(int(neuron_idx))
    output_spikes = {'t': output_t, 'i': output_i}
    
    # Convert voltage history to native Python floats
    voltage_native = []
    for step in voltage_history:
        voltage_native.append([float(v) for v in step])
    
    return {
        'input': input_spikes,
        'hidden': {'t': [], 'i': []},
        'output': output_spikes,
        'voltage_history': voltage_native
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
