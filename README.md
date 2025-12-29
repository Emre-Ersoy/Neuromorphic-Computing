# Neuromorphic Computing Experiments

This repository contains a collection of exercises and projects exploring signal processing, artificial neural networks (ANN), and neuromorphic computing concepts using Spiking Neural Networks (SNN).

The projects demonstrate the transition from traditional signal processing methods to biologically inspired computing models.

---

## Project Structure

### 1. Homework 1: Waveform Classifier (FFT + ANN)
**Path:** `Homework1/waveform_classifier.py`

This project implements a classic machine learning approach to signal classification. It identifies fundamental waveforms (Sine, Square, Triangle) using frequency domain analysis.

*   **Key Techniques:**
    *   **Fast Fourier Transform (FFT):** Implements a recursive Radix-2 FFT algorithm to analyze signal frequency components.
    *   **Harmonic Analysis:** Extracts feature vectors based on the ratios of harmonic frequencies (2nd, 3rd, etc.) relative to the fundamental frequency.
    *   **Artificial Neural Network (ANN):** Uses a standard feed-forward neural network with backpropagation to classify the waveforms based on their harmonic signatures.
*   **Visualization:** Displays the generated waveform and classification confidence scores.

### 2. Homework 2: Spiking Neural Network Classifier (LIF)
**Path:** `Homework2/snn_classifier.py`

This project solves the same classification problem but adopts a neuromorphic approach, simulating biological neurons.

*   **Key Techniques:**
    *   **Leaky Integrate-and-Fire (LIF) Neurons:** Simulates neurons that accumulate membrane potential over time and "spike" when a threshold is reached.
    *   **Temporal Coding:** Information is processed over time steps, mimicking how real brains process signals.
    *   **Spike-Based Learning:** Weights are adjusted based on the spiking activity and error signal (using a delta-rule adaptation for SNNs).
*   **Visualization:**
    *   **Membrane Potential Traces:** Features a real-time oscilloscope-style visualization showing the voltage ($v$) of each neuron.
    *   **Dynamics:** Users can observe the voltage building up, hitting the threshold, firing a spike, and resetting, providing a clear view of the SNN dynamics.

---

## Getting Started

### Prerequisites

Ensure you have the required Python packages installed:

```bash
pip install -r requirements.txt
```

### Running the Projects

Each project runs as a local web application using **Flask**.

**To run Homework 1 (ANN):**
```bash
cd Homework1
python waveform_classifier.py
```

**To run Homework 2 (SNN):**
```bash
cd Homework2
python snn_classifier.py
```

Once running, open your browser and navigate to `http://127.0.0.1:5000` (or the port specified in the console).

---

## Concepts Covered

| Concept | Homework 1 (ANN) | Homework 2 (SNN) |
| :--- | :--- | :--- |
| **Input Processing** | Static Feature Vector | Temporal Input Stream |
| **Neuron Model** | Sigmoid Activation | Leaky Integrate-and-Fire (LIF) |
| **Output** | Probability (0.0 - 1.0) | Spike Counts / Rate Coding |
| **Biorealism** | Low (Abstract Math) | High (Bio-simulation) |

