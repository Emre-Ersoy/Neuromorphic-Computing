"""
Neuromorphic OCR - Motor Cortical Network
Exemplar-Based Learning with Smart Pruning + K-NN Prediction
"""

import random
import pickle
import os
import numpy as np
from collections import defaultdict

MODEL_FILE = "ocr_snn_model.pkl"

# ==========================================
# LIF Neuron
# ==========================================

class LIFNeuron:
    """Simple Leaky Integrate-and-Fire neuron"""
    def __init__(self, tau=15.0, threshold=1.5, reset_val=0.0):
        self.v = reset_val
        self.tau = tau
        self.threshold = threshold
        self.reset_val = reset_val
        self.spike_count = 0
        
    def step(self, input_current, dt=1.0):
        dv = (-(self.v - self.reset_val) + input_current) / self.tau
        self.v += dv * dt
        
        is_spike = False
        track_v = self.v 
        
        if self.v >= self.threshold:
            self.v = self.reset_val
            self.spike_count += 1
            is_spike = True
        elif self.v < -3.0:
            self.v = -3.0
            track_v = self.v
            
        return is_spike, track_v
    
    def reset(self):
        self.v = self.reset_val
        self.spike_count = 0


# ==========================================
# OCR SNN with Exemplar-Based Learning
# ==========================================

class OCRSNN:
    def __init__(self, n_input=256, n_output=20, max_exemplars=80):
        """
        Exemplar-based learning:
        - Store up to max_exemplars samples per class
        - K-NN prediction (find most similar exemplar)
        - Smart pruning (remove most similar when full)
        """
        self.n_input = n_input  # 16x16 = 256
        self.n_output = n_output
        self.max_exemplars = max_exemplars
        
        # Exemplar storage: {label: [sample1, sample2, ...]}
        self.exemplars = defaultdict(list)
        
        # For SNN visualization (dummy neurons for now)
        self.neurons = [LIFNeuron(tau=10.0, threshold=0.8) for _ in range(n_output)]
        
        # Label to index mapping
        self.label_map = {}
        self.neuron_labels = {}
        self.next_label_idx = 0
        
        # Simulation params
        self.time_steps = 50
        
    def _normalize(self, arr):
        """L2 normalize a vector"""
        norm = np.linalg.norm(arr)
        if norm > 0:
            return arr / norm
        return arr
    
    def _cosine_similarity(self, a, b):
        """Cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _find_most_similar_pair(self, samples):
        """Find indices of two most similar samples (for pruning)"""
        n = len(samples)
        if n < 2:
            return 0, 1
        
        max_sim = -1
        pair = (0, 1)
        
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_similarity(samples[i], samples[j])
                if sim > max_sim:
                    max_sim = sim
                    pair = (i, j)
        
        return pair
    
    def _smart_prune(self, label):
        """
        Smart pruning: when over limit, merge two most similar samples.
        Keeps diverse samples, removes redundant ones.
        """
        samples = self.exemplars[label]
        
        if len(samples) <= self.max_exemplars:
            return
        
        # Find two most similar
        i, j = self._find_most_similar_pair(samples)
        
        # Merge them (average) and remove the pair
        merged = (samples[i] + samples[j]) / 2
        merged = self._normalize(merged)
        
        # Remove higher index first to preserve lower index
        if i > j:
            i, j = j, i
        samples.pop(j)
        samples.pop(i)
        samples.append(merged)
        
    def train_step(self, input_data, label, repetitions=1):
        """
        Add sample to exemplar storage for this class.
        Smart pruning keeps collection diverse.
        """
        input_array = np.array(input_data).flatten().astype(np.float32)
        input_array = self._normalize(input_array)
        
        # Register label if new
        if label not in self.label_map:
            if self.next_label_idx < self.n_output:
                self.label_map[label] = self.next_label_idx
                self.neuron_labels[self.next_label_idx] = label
                self.next_label_idx += 1
            else:
                return None
        
        # Add to exemplars
        self.exemplars[label].append(input_array)
        
        # Prune if over limit
        self._smart_prune(label)
        
        # Run inference for visualization
        spike_counts, voltage_history, spike_times, input_spike_times = self.run_inference(input_data)
        return spike_counts
    
    def run_inference(self, input_data, return_all=False):
        """
        K-NN inference: find most similar exemplar across all classes.
        """
        input_array = np.array(input_data).flatten().astype(np.float32)
        input_array = self._normalize(input_array)
        
        # Calculate similarity to all exemplars
        class_scores = {}
        best_matches = {}
        
        for label, samples in self.exemplars.items():
            if len(samples) == 0:
                continue
            
            # Find max similarity to any exemplar in this class
            similarities = [self._cosine_similarity(input_array, s) for s in samples]
            max_sim = max(similarities) if similarities else 0
            class_scores[label] = max_sim
            best_matches[label] = max_sim
        
        # Reset neurons
        for neuron in self.neurons:
            neuron.reset()
            
        # Generate visualization (spike simulation based on similarity)
        voltage_history = []
        spike_times = {i: [] for i in range(self.n_output)}
        input_spike_times = {i: [] for i in range(self.n_input)}
        
        for t in range(self.time_steps):
            # Input spikes (Poisson)
            for i in range(self.n_input):
                if input_array[i] > 0.1:
                    if random.random() < input_array[i] * 0.3:
                        input_spike_times[i].append(t)
            
            step_voltages = []
            for j in range(self.n_output):
                # Get label for this neuron
                label = self.neuron_labels.get(j, None)
                score = class_scores.get(label, 0) if label else 0
                
                # Current based on similarity score
                current = score * 5.0 + random.gauss(0, 0.05)
                
                spike, v = self.neurons[j].step(current)
                step_voltages.append(v)
                
                if spike:
                    spike_times[j].append(t)
                    # Lateral inhibition
                    for k in range(self.n_output):
                        if k != j:
                            self.neurons[k].v -= 2.0
                            
            voltage_history.append(step_voltages)
        
        spike_counts = [n.spike_count for n in self.neurons]
        
        if return_all:
            return spike_counts, voltage_history, spike_times, input_spike_times, class_scores
        return spike_counts, voltage_history, spike_times, input_spike_times
    
    def predict(self, input_data):
        """
        K-NN prediction: return label of class with highest similarity.
        """
        spike_counts, voltage_history, spike_times, input_spike_times, class_scores = \
            self.run_inference(input_data, return_all=True)
        
        if not class_scores:
            return None, spike_counts, voltage_history, spike_times, input_spike_times
        
        # Winner: highest similarity
        winner_label = max(class_scores, key=class_scores.get)
        winner_score = class_scores[winner_label]
        
        return winner_label, spike_counts, voltage_history, spike_times, input_spike_times
    
    def predict_word(self, segments):
        """
        Predict multiple segments (for word recognition).
        Returns list of predicted labels.
        """
        results = []
        for segment in segments:
            label, _, _, _, _ = self.predict(segment)
            results.append(label if label else '?')
        return results
    
    def reinforce_correct(self, input_data, predicted_label):
        """
        Positive feedback: add this sample as an exemplar (reinforcement).
        """
        if predicted_label not in self.label_map:
            return False
        
        input_array = np.array(input_data).flatten().astype(np.float32)
        input_array = self._normalize(input_array)
        
        self.exemplars[predicted_label].append(input_array)
        self._smart_prune(predicted_label)
        
        self.save()
        return True
    
    def correct_wrong(self, input_data, wrong_label, correct_label):
        """
        Negative feedback: add to correct class, remove similar from wrong class.
        """
        input_array = np.array(input_data).flatten().astype(np.float32)
        input_array = self._normalize(input_array)
        
        # Add to correct class
        if correct_label not in self.label_map:
            if self.next_label_idx < self.n_output:
                self.label_map[correct_label] = self.next_label_idx
                self.neuron_labels[self.next_label_idx] = correct_label
                self.next_label_idx += 1
            else:
                return False
        
        self.exemplars[correct_label].append(input_array)
        self._smart_prune(correct_label)
        
        # Remove most similar exemplar from wrong class (if exists and similar)
        if wrong_label in self.exemplars and len(self.exemplars[wrong_label]) > 0:
            samples = self.exemplars[wrong_label]
            similarities = [self._cosine_similarity(input_array, s) for s in samples]
            max_idx = np.argmax(similarities)
            
            # Only remove if very similar (likely a mistake)
            if similarities[max_idx] > 0.7:
                samples.pop(max_idx)
        
        self.save()
        return True
    
    def get_labels(self):
        return {v: k for k, v in self.label_map.items()}
    
    def get_stats(self):
        """Return statistics about exemplar storage"""
        stats = {}
        for label in self.exemplars:
            stats[label] = len(self.exemplars[label])
        return stats
    
    def reset(self):
        self.exemplars = defaultdict(list)
        self.neurons = [LIFNeuron(tau=10.0, threshold=0.8) for _ in range(self.n_output)]
        self.label_map = {}
        self.neuron_labels = {}
        self.next_label_idx = 0
        
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
    
    def save(self):
        try:
            data = {
                'exemplars': dict(self.exemplars),
                'label_map': self.label_map,
                'neuron_labels': self.neuron_labels,
                'next_label_idx': self.next_label_idx,
                'max_exemplars': self.max_exemplars
            }
            with open(MODEL_FILE, 'wb') as f:
                pickle.dump(data, f)
            print(f"Model saved ({sum(len(v) for v in self.exemplars.values())} total exemplars)")
        except Exception as e:
            print(f"Error saving: {e}")
    
    def load(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                
                # Handle old format (weights-based)
                if 'exemplars' in data:
                    self.exemplars = defaultdict(list, data['exemplars'])
                else:
                    # Convert old weights to empty exemplars
                    self.exemplars = defaultdict(list)
                    
                self.label_map = data.get('label_map', {})
                self.neuron_labels = data.get('neuron_labels', {})
                self.next_label_idx = data.get('next_label_idx', 0)
                self.max_exemplars = data.get('max_exemplars', 80)
                
                total = sum(len(v) for v in self.exemplars.values())
                print(f"Model loaded ({total} exemplars across {len(self.label_map)} classes)")
                return True
            except Exception as e:
                print(f"Error loading: {e}")
        return False


# Singleton
snn_model = OCRSNN(n_input=256, n_output=20, max_exemplars=80)
snn_model.load()
