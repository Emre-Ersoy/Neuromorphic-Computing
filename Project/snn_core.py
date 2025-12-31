"""
Neuromorphic OCR - Motor Cortical Network
Exemplar-Based Learning with Smart Pruning + K-NN Prediction
"""
import hashlib
import random
import pickle
import os
import numpy as np
from PIL import Image, ImageFilter
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
        self.refractory_steps = 2
        self._refractory_left = 0
        
    def step(self, input_current, dt=1.0):
        if self._refractory_left > 0:
            self._refractory_left -= 1
            return False, self.v
        
        dv = (-(self.v - self.reset_val) + input_current) / self.tau
        self.v += dv * dt
        
        is_spike = False
        track_v = self.v 
        
        if self.v >= self.threshold:
            self.v = self.reset_val
            self.spike_count += 1
            is_spike = True
            self._refractory_left = self.refractory_steps

        elif self.v < -3.0:
            self.v = -3.0
            track_v = self.v
            
        return is_spike, track_v
    
    def reset(self):
        self.v = self.reset_val
        self.spike_count = 0
        self._refractory_left = 0



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
        self.neurons = [LIFNeuron(tau=10.0, threshold=0.5) for _ in range(n_output)]
        
        # Label to index mapping
        self.label_map = {}
        self.neuron_labels = {}
        self.next_label_idx = 0
        
        # Simulation params
        self.time_steps = 50
        
    def _preprocess(self, input_array):
        """
        Apply Gaussian Blur to prioritize topology (shape) over exact pixel match.
        This spreads activation to neighbors, making 'near misses' count as hits.
        """
        # Reshape to 16x16
        grid = input_array.reshape(16, 16)
        img = Image.fromarray((grid * 255).astype(np.uint8))
        
        # Apply slight blur (Radius 0.5 - 1.0)
        # This mimics 'receptive fields' in the eye
        # Reduced to 0.4 to prevent 'D' looking like 'F' (preserving gaps)
        blurred = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        
        # Flatten and normalize
        arr = np.array(blurred).flatten().astype(np.float32) / 255.0
        return self._normalize(arr)

    def _normalize(self, arr):
        """L2 normalize a vector"""
        norm = np.linalg.norm(arr)
        if norm > 0:
            return arr / norm
        return arr
    
    def _augment_data(self, input_array):
        """
        Generate shifted versions of the input array (up, down, left, right)
        Input: flattened or 1d array of size 256
        Output: list of arrays (including original)
        """
        grid = input_array.reshape(16, 16)
        augmented = [input_array]
        
        # Shift functions
        # Shift functions (Expanded to +/- 2 pixels for better centering invariance)
        shifts = [
            (-1, 0), (1, 0), # Up, Down (1px)
            (0, -1), (0, 1), # Left, Right (1px)
            (-2, 0), (2, 0), # Up, Down (2px) - Aggressive shift
            (0, -2), (0, 2)  # Left, Right (2px) - Aggressive shift
        ]
        
        for dy, dx in shifts:
            shifted = np.roll(grid, dy, axis=0)
            shifted = np.roll(shifted, dx, axis=1)
            
            # Mask rolling artifacts (zero out wrapped edges)
            # Mask rolling artifacts (zero out wrapped edges)
            if abs(dy) >= 1:
                if dy > 0: shifted[:dy, :] = 0
                else: shifted[dy:, :] = 0
            
            if abs(dx) >= 1:
                if dx > 0: shifted[:, :dx] = 0
                else: shifted[:, dx:] = 0
            
            augmented.append(shifted.flatten())
            
        # Rotation (Biological Invariance: Tilt)
        # Convert to PIL Image for high-quality rotation
        img = Image.fromarray((grid * 255).astype(np.uint8))
        
        for angle in [15, -15]:
            # Rotate with bilinear interpolation for smoothness
            rot_img = img.rotate(angle, resample=Image.BILINEAR)
            rot_arr = np.array(rot_img).astype(np.float32) / 255.0
            augmented.append(rot_arr.flatten())
            
        # Scaling (Biological Invariance: Size)
        # 1. Zoom Out (0.8x) - Padding
        w, h = 16, 16
        small_size = (int(w * 0.8), int(h * 0.8))
        small_img = img.resize(small_size, resample=Image.BILINEAR)
        new_img = Image.new("L", (w, h), 0)
        paste_x = (w - small_size[0]) // 2
        paste_y = (h - small_size[1]) // 2
        new_img.paste(small_img, (paste_x, paste_y))
        augmented.append(np.array(new_img).astype(np.float32).flatten() / 255.0)
        
        # 2. Zoom In (1.2x) - Cropping
        large_size = (int(w * 1.2), int(h * 1.2))
        large_img = img.resize(large_size, resample=Image.BILINEAR)
        crop_x = (large_size[0] - w) // 2
        crop_y = (large_size[1] - h) // 2
        cropped_img = large_img.crop((crop_x, crop_y, crop_x + w, crop_y + h))
        augmented.append(np.array(cropped_img).astype(np.float32).flatten() / 255.0)
            
        return augmented
    def _stable_rng(self, input_array):
        x = np.clip(input_array, 0.0, 1.0)
        x = (x * 1000).astype(np.int16)
        h = hashlib.sha256(x.tobytes()).digest()
        seed = int.from_bytes(h[:8], "little", signed=False) & 0xFFFFFFFF
        return random.Random(seed)
    
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
        # Use preprocess (blur + norm) instead of just norm
        input_array = self._preprocess(input_array)
        
        # Register label if new
        if label not in self.label_map:
            if self.next_label_idx < self.n_output:
                self.label_map[label] = self.next_label_idx
                self.neuron_labels[self.next_label_idx] = label
                self.next_label_idx += 1
            else:
                return None
        
        # Augment data (create 5 versions: original + 4 shifts)
        augmented_samples = self._augment_data(input_array)
        
        for sample in augmented_samples:
            norm_sample = self._normalize(sample)
            self.exemplars[label].append(norm_sample)
            self._smart_prune(label)
        
        # Run inference for visualization (using ORIGINAL input only)
        # Training -> Always use visual mode for feedback
        spike_counts, voltage_history, spike_times, input_spike_times = self.run_inference(input_data, visual_mode=True)
        return spike_counts, voltage_history, spike_times, input_spike_times
    
    def run_inference(self, input_data, return_all=False, visual_mode=False):
        """
        K-NN inference: find most similar exemplar across all classes.
        """
        # CRITICAL: Reset neuron states before every inference to prevent "State Leakage"
        self.reset_states()
        
        input_array = np.array(input_data).flatten().astype(np.float32)
        # Use preprocess (blur + norm)
        input_array = self._preprocess(input_array)
        rng = self._stable_rng(input_array)

        
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
            
        print(f"DEBUG: Max Similarity Score: {max(class_scores.values()) if class_scores else 0}")
        
        # Reset neurons
        self.reset_states()
            
        # Generate visualization (spike simulation based on similarity)
        voltage_history = [[0.0] * self.n_output for _ in range(self.time_steps)]
        spike_times = {i: [] for i in range(self.n_output)}
        input_spike_times = {i: [] for i in range(self.n_input)}
        
       
        
        for t in range(self.time_steps):
            # Input spikes (Poisson)
            for i in range(self.n_input):
                # Lowered threshold to 0.01 because L2 normalization can push values below 0.1
                # for dense inputs (thick characters).
                if input_array[i] > 0.01:
                    # Reverted to 0.3 Poisson rate (Original)
                    POISSON_SCALE = 0.3
                    if rng.random() < input_array[i] * POISSON_SCALE:
                        input_spike_times[i].append(t)
            
            # Randomize order to prevent Index 0 bias (Sequential Fairness)
            # Seed ensures this shuffle is also deterministic per run
            neuron_indices = list(range(self.n_output))
            rng.shuffle(neuron_indices)

            
            # Track spikes in this timestep for lateral inhibition
            output_spikes = [0] * self.n_output
            spike_occurred = False
            
            for j in neuron_indices:
                # Get label for this neuron
                lbl = self.neuron_labels.get(j, "?")
                score = class_scores.get(lbl, 0)
                
                # Debug top 5 scores at t=0
                if t == 0:
                     n_idx = j
                     n_lbl = lbl
                     n_score = score
                     if n_score > 0.5:
                         print(f"DEBUG: Neuron {n_idx} (Label '{n_lbl}') <- Score: {n_score:.2f}")

                # Current based on similarity score
                # Visual Mode: Sharper, stronger gain. Process Mode: Softer, safer.
                if visual_mode:
                    current = max(0, score) * 6.0 + random.gauss(0, 0.02)
                else: 
                    current = max(0, score) * 5.0 + random.gauss(0, 0.05)
                
                spike, v = self.neurons[j].step(current)
                
                
                
                if spike:
                    spike_occurred = True
                    voltage_history[t][j] = self.neurons[j].threshold
                    output_spikes[j] = 1
                    spike_times[j].append(t)  # Record spike time

                    # Lateral inhibition
                    # Visual Mode: Stronger (1.5 - 2.0) to separating lines
                    # Process Mode: Softer (0.85) to avoid false negatives
                    INH = 2.0 if visual_mode else 0.85

                    for k in range(self.n_output):
                        if k != j:
                            self.neurons[k].v -= INH
                else:
                    voltage_history[t][j] = v

        spike_counts = [n.spike_count for n in self.neurons]
        print(f"DEBUG: Spike Counts: {spike_counts}")
        
        if return_all:
            return spike_counts, voltage_history, spike_times, input_spike_times, class_scores
        return spike_counts, voltage_history, spike_times, input_spike_times
    
    def predict(self, input_data, visual_mode=False):
        """
        K-NN prediction: return label of class with highest similarity.
        """
        spike_counts, voltage_history, spike_times, input_spike_times, class_scores = \
            self.run_inference(input_data, return_all=True, visual_mode=visual_mode)
        
        if not class_scores:
            return None, 0.0, spike_counts, voltage_history, spike_times, input_spike_times
        
        # Winner: highest similarity
        winner_label = max(class_scores, key=class_scores.get)
        winner_score = class_scores[winner_label]
        
        return winner_label, winner_score, spike_counts, voltage_history, spike_times, input_spike_times
    

    
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
            
            # Only remove if very similar (likely a mistake) - Original 0.7
            if similarities[max_idx] > 0.7:
                samples.pop(max_idx)
        
        self.save()
        return True
    
    def get_labels(self):
        # Return Label -> Index mapping (e.g. {'A': 0, 'B': 1})
        return self.label_map
    
    def get_stats(self):
        """Return statistics about exemplar storage"""
        stats = {}
        for label in self.exemplars:
            stats[label] = len(self.exemplars[label])
        return stats
    
    def reset_states(self):
        """
        Reset only the neuron states (voltage, spikes), not the learned memory.
        """
        for neuron in self.neurons:
            neuron.reset()

    def reset(self):
        self.exemplars = defaultdict(list)
        self.neurons = [LIFNeuron(tau=10.0, threshold=0.7) for _ in range(self.n_output)]
        self.label_map = {}
        self.neuron_labels = {}
        self.next_label_idx = 0
        
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        
        # Save the empty state immediately to sync disk
        self.save()
        print("Network reset and empty model saved.")
    
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
