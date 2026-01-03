"""
Motor Cortex v2 - Raw Trajectory Replay

Stores actual stroke points and replays them directly for accurate handwriting reproduction.
Motor neurons fire based on movement direction for biological visualization.

Reference:
- Georgopoulos, A. P., et al. (1986). Neuronal population coding of movement direction.
"""

import math
import numpy as np
from collections import defaultdict


# ==========================================
# Local LIF Neuron (to avoid circular import)
# ==========================================

class LIFNeuron:
    """Simple Leaky Integrate-and-Fire neuron for motor cortex."""
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


# 16-Direction labels for motor neuron visualization
DIRECTION_LABELS = [
    'N', 'NNE', 'NE', 'ENE', 
    'E', 'ESE', 'SE', 'SSE',
    'S', 'SSW', 'SW', 'WSW', 
    'W', 'WNW', 'NW', 'NNW'
]


class MotorCortex:
    """
    Motor Cortex v2 - Raw Trajectory Storage and Replay.
    
    Instead of direction encoding (which loses shape info), we store 
    the actual normalized (x, y) points and replay them directly.
    
    Motor neurons still fire based on movement direction for visualization,
    but the trajectory reproduction uses the stored raw points.
    """
    
    def __init__(self):
        # 16 LIF neurons for direction visualization
        self.neurons = [LIFNeuron(tau=8.0, threshold=0.6) for _ in range(16)]
        
        # Motor memories: raw trajectory points for each class
        # {label: [[[x,y], [x,y], ...], ...]}  # Multiple samples of point lists
        self.motor_memories = defaultdict(list)
        
        # Maximum samples per label
        self.max_samples = 5
        
    def reset_neurons(self):
        """Reset all motor neuron states."""
        for neuron in self.neurons:
            neuron.reset()
    
    # ==========================================
    # Trajectory Storage (Learning)
    # ==========================================
    
    def store_trajectory(self, label, stroke_segments, max_points=200):
        """
        Store raw trajectory points for a label.
        
        Args:
            label: Character label (e.g., 'F')
            stroke_segments: List of stroke segments [[[x,y], [x,y], ...], ...]
            max_points: Maximum points to store (subsampled if exceeds)
        """
        # Flatten all segments into single trajectory
        all_points = []
        for segment in stroke_segments:
            for point in segment:
                all_points.append(point)
        
        if len(all_points) < 2:
            print(f"  Not enough points to store for '{label}'")
            return False
        
        # Subsample if too many points
        if len(all_points) > max_points:
            step = len(all_points) / max_points
            all_points = [all_points[int(i * step)] for i in range(max_points)]
        
        print(f"\n=== MOTOR: store_trajectory ===")
        print(f"  Label: '{label}'")
        print(f"  Points: {len(all_points)}")
        print(f"  First: {all_points[0]}, Last: {all_points[-1]}")
        
        # Store
        self.motor_memories[label].append(all_points)
        
        # Keep only recent samples
        if len(self.motor_memories[label]) > self.max_samples:
            self.motor_memories[label].pop(0)
        
        return True
    
    def remove_last_sample(self, label):
        """Remove the most recent sample for a label (feedback: wrong)."""
        if label in self.motor_memories and len(self.motor_memories[label]) > 0:
            removed = self.motor_memories[label].pop()
            remaining = len(self.motor_memories[label])
            print(f"Motor feedback: Removed last sample for '{label}'. {remaining} remaining.")
            return True
        return False
    
    def get_trajectory(self, label):
        """
        Get averaged trajectory for a label.
        
        If multiple samples exist, averages corresponding points.
        Returns: [[x,y], [x,y], ...] or None
        """
        if label not in self.motor_memories or len(self.motor_memories[label]) == 0:
            print(f"No motor memory for '{label}'")
            return None
        
        samples = self.motor_memories[label]
        print(f"\n=== MOTOR: get_trajectory ===")
        print(f"  Label: '{label}', Samples: {len(samples)}")
        
        if len(samples) == 1:
            # Single sample - return as is
            return samples[0]
        
        # Multiple samples - average them (align by resampling to same length)
        target_length = max(len(s) for s in samples)
        
        # Resample all to target length
        resampled = []
        for sample in samples:
            if len(sample) == target_length:
                resampled.append(sample)
            else:
                # Linear interpolation to target length
                new_sample = []
                for i in range(target_length):
                    t = i / (target_length - 1) * (len(sample) - 1)
                    idx = int(t)
                    frac = t - idx
                    if idx >= len(sample) - 1:
                        new_sample.append(sample[-1])
                    else:
                        x = sample[idx][0] * (1 - frac) + sample[idx + 1][0] * frac
                        y = sample[idx][1] * (1 - frac) + sample[idx + 1][1] * frac
                        new_sample.append([x, y])
                resampled.append(new_sample)
        
        # Average
        averaged = []
        for i in range(target_length):
            avg_x = sum(s[i][0] for s in resampled) / len(resampled)
            avg_y = sum(s[i][1] for s in resampled) / len(resampled)
            averaged.append([avg_x, avg_y])
        
        print(f"  Averaged {len(samples)} samples to {len(averaged)} points")
        return averaged
    
    # ==========================================
    # Motor Neuron Activation (for visualization)
    # ==========================================
    
    def get_direction_from_points(self, p1, p2):
        """Calculate direction from p1 to p2, returns (dx, dy) normalized."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        mag = math.sqrt(dx**2 + dy**2)
        if mag < 0.001:
            return 0, 0
        return dx / mag, dy / mag
    
    def activate_for_direction(self, dx, dy):
        """
        Activate motor neurons based on movement direction.
        Returns list of (spike, voltage) for each of 16 neurons.
        """
        results = []
        
        for i in range(16):
            # Calculate preferred direction for this neuron
            angle_rad = math.radians(i * 22.5 - 90)
            pref_dx = math.cos(angle_rad)
            pref_dy = math.sin(angle_rad)
            
            # Cosine tuning: activation = cos(angle between direction and preferred)
            dot = dx * pref_dx + dy * pref_dy
            activation = max(0, (dot + 1) / 2)  # [0, 1]
            
            # Step neuron
            current = activation * 2.0
            spike, voltage = self.neurons[i].step(current)
            results.append((spike, voltage))
        
        return results
    
    # ==========================================
    # Trajectory Replay (Simulation)
    # ==========================================
    
    def simulate_trajectory(self, label):
        """
        Get trajectory for replay and simulate motor neuron activity.
        
        Returns:
            dict with 'status', 'trajectory', 'motor_spikes', 'direction_labels'
        """
        trajectory = self.get_trajectory(label)
        
        if trajectory is None:
            return {
                'status': 'no_plan',
                'trajectory': [],
                'motor_spikes': [],
                'direction_labels': DIRECTION_LABELS
            }
        
        self.reset_neurons()
        
        # Generate motor neuron activity for visualization
        all_spikes = []
        
        for i in range(1, len(trajectory)):
            # Get direction of movement
            dx, dy = self.get_direction_from_points(trajectory[i-1], trajectory[i])
            
            # Activate neurons
            results = self.activate_for_direction(dx, dy)
            spikes = [r[0] for r in results]
            all_spikes.append(spikes)
        
        # Convert trajectory to list format (ensure JSON serializable)
        traj_list = [[p[0], p[1]] for p in trajectory]
        
        print(f"  Trajectory: {len(traj_list)} points, Motor activity: {len(all_spikes)} steps")
        
        return {
            'status': 'success',
            'trajectory': traj_list,
            'motor_spikes': all_spikes,
            'direction_labels': DIRECTION_LABELS
        }
    
    # ==========================================
    # State Persistence
    # ==========================================
    
    def get_state(self):
        """Get motor cortex state for saving."""
        return {
            'motor_memories': dict(self.motor_memories)
        }
    
    def set_state(self, state):
        """Restore motor cortex state."""
        if 'motor_memories' in state:
            self.motor_memories = defaultdict(list, state['motor_memories'])
