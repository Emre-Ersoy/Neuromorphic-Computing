import random
import os
import json
import io
import base64
from flask import Flask, render_template, request, jsonify
from snn_core import OCRSNN
from word_processor import process_word_image
import numpy as np
from PIL import Image

app = Flask(__name__)

# Initialize SNN Model
snn_model = OCRSNN()
if os.path.exists("ocr_snn_model.pkl"):
    snn_model.load()

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
    stroke_segments = data.get('stroke_segments', None)  # [[points], [points], ...] from canvas
    
    spike_counts, voltage_history, spike_times, input_spike_times = snn_model.train_step(
        pixels, label, stroke_segments=stroke_segments
    )
    snn_model.save()
    
    results = format_spike_data(spike_times, input_spike_times, voltage_history)
    
    stats = snn_model.get_stats()
    
    # Include motor memory status
    has_motor = label in snn_model.motor_cortex.motor_memories
    
    return jsonify({
        'status': 'trained', 
        'label': label,
        'mapping': snn_model.label_map,
        'stats': stats,
        'results': results,
        'spike_counts': spike_counts,
        'has_motor_memory': has_motor
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    global last_prediction
    data = request.json
    pixels = data.get('pixels', [])
    
    winner_label, winner_score, spike_counts, voltage_history, spike_times, input_spike_times = snn_model.predict(pixels, visual_mode=True)
    
    # Store for feedback
    last_prediction['pixels'] = pixels
    last_prediction['predicted_label'] = winner_label
    
    results = format_spike_data(spike_times, input_spike_times, voltage_history)
    
    return jsonify({
        'status': 'predicted',
        'winner': winner_label,
        'confidence': float(winner_score),
        'mapping': snn_model.get_labels(),
        'spike_counts': spike_counts,
        'results': results,
        'has_motor_memory': winner_label in snn_model.motor_cortex.motor_memories if winner_label else False
    })

@app.route('/api/motor_reconstruct', methods=['POST'])
def motor_reconstruct():
    """
    Motor Cortex Simulation: Regenerate trajectory for a recognized label.
    Uses Georgopoulos Population Vector Coding with 16 directional neurons.
    """
    data = request.json
    label = data.get('label', '').strip().upper()
    
    if not label:
        return jsonify({'status': 'error', 'message': 'No label provided'})
    
    # Run motor simulation
    result = snn_model.run_motor_simulation(label)
    
    if result['status'] == 'no_plan':
        return jsonify({
            'status': 'no_plan',
            'message': f"No motor memory for '{label}'. Train with stroke data first."
        })
    
    return jsonify({
        'status': 'success',
        'label': label,
        'trajectory': result['trajectory'],
        'motor_spikes': result['motor_spikes'],
        'direction_labels': result['direction_labels']
    })

@app.route('/api/motor_feedback', methods=['POST'])
def motor_feedback():
    """
    Motor Feedback: Remove last motor memory sample for a label.
    Called when user clicks 'Motor Wrong' button.
    """
    data = request.json
    label = data.get('label', '').strip().upper()
    
    if not label:
        return jsonify({'status': 'error', 'message': 'No label provided'})
    
    success = snn_model.motor_cortex.remove_last_sample(label)
    snn_model.save()  # Persist the change
    
    remaining = len(snn_model.motor_cortex.motor_memories.get(label, []))
    
    return jsonify({
        'status': 'removed' if success else 'no_samples',
        'label': label,
        'remaining_samples': remaining
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
    try:
        data = request.get_json(silent=True) or {}
    except Exception as e:
        print(f"JSON parsing error in feedback_correct: {e}")
        data = {}
    stroke_segments = data.get('stroke_segments', None)
    
    if last_prediction['pixels'] is None or last_prediction['predicted_label'] is None:
        return jsonify({'status': 'error', 'message': 'No prediction to reinforce'})
    
    label = last_prediction['predicted_label']
    
    success = snn_model.reinforce_correct(
        last_prediction['pixels'],
        label
    )
    
    # Also train motor cortex if stroke data provided
    motor_learned = False
    if stroke_segments and len(stroke_segments) > 0:
        motor_learned = snn_model.motor_cortex.store_trajectory(label, stroke_segments)
        if motor_learned:
            print(f"Motor trajectory learned from Correct feedback for '{label}'")
    
    snn_model.save()
    
    return jsonify({
        'status': 'reinforced' if success else 'error',
        'label': label,
        'motor_learned': motor_learned,
        'stats': snn_model.get_stats()
    })

@app.route('/api/feedback/wrong', methods=['POST'])
def feedback_wrong():
    global last_prediction
    data = request.json
    correct_label = data.get('correct_label', '').strip().upper()
    
    wrong_label = last_prediction['predicted_label']
    
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
        'wrong_label': wrong_label,
        'correct_label': correct_label,
        'mapping': snn_model.label_map,
        'stats': snn_model.get_stats()
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    snn_model.reset()
    return jsonify({'status': 'reset'})

def format_spike_data(spike_times, input_spike_times, voltage_history):
    # Format input spikes as list of objects
    input_spikes = []
    for neuron_idx, times in input_spike_times.items():
        for t in times:
            input_spikes.append({
                't': float(t) / 100.0, 
                'neuron_idx': int(neuron_idx)
            })
            
    # Format output spikes
    output_spikes = []
    for neuron_idx, times in spike_times.items():
        for t in times:
            output_spikes.append({
                't': float(t) / 100.0,
                'neuron_idx': int(neuron_idx)
            })
            
    # Convert voltage history to native Python floats
    voltage_native = []
    for step in voltage_history:
        voltage_native.append([float(v) for v in step])
        
    return {
        'input_spikes': input_spikes,
        'output_spikes': output_spikes,
        'voltage_history': voltage_native
    }

@app.route('/api/test_word', methods=['POST'])
def test_word():
    data = request.json
    image_data = data['image'] # Base64 string
    
    # Remove header if present
    if ',' in image_data:
        image_data = image_data.split(',')[1]
        
    try:
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image_file = io.BytesIO(image_bytes)
        
        # Process word
        segments = process_word_image(image_file)
        
        results = []
        for seg in segments:
            pixels = seg['pixels']
            bbox = seg['bbox']
            
            # Run inference
            # Ensure reset states to prevent leakage between letters
            snn_model.reset_states()
            # Use return_all=True to get class_scores (KNN Similarity) which is stable/deterministic
            spike_counts, _, _, _, class_scores = snn_model.run_inference(pixels, return_all=True)
            
            # Get winner based on SIMILARITY (Stable), not Spikes (Stochastic)
            winner_label = None
            max_spikes = max(spike_counts) if spike_counts else 0
            
            if class_scores:
                # Deterministic winner
                winner_label = max(class_scores, key=class_scores.get)
                # Confidence based on similarity (0-1)
                confidence = float(class_scores[winner_label])
            else:
                 # Fallback to spikes if no classes defined
                 if spike_counts and max_spikes > 0:
                     winners = [i for i, c in enumerate(spike_counts) if c == max_spikes]
                     winner_idx = winners[0]
                     winner_label = snn_model.neuron_labels.get(winner_idx)
                     confidence = max_spikes / 50.0
                 else:
                     confidence = 0.0
                
            results.append({
                'bbox': bbox,
                'label': winner_label if winner_label else "?",
                'confidence': confidence,
                'spikes': max_spikes
            })
            
        return jsonify({'status': 'success', 'segments': results})
        
    except Exception as e:
        print(f"Error in word test: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)
