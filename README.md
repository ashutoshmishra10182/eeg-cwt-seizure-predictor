# Electroencephalogram (EEG) Seizure Prediction via Continuous Wavelet Transforms (CWT)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![DSP](https://img.shields.io/badge/DSP-Complex%20Morlet%20CWT-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

## Overview
This repository provides an automated biomedical signal processing and deep learning pipeline to predict epileptic seizures minutes before clinical onset using multi-channel scalp Electroencephalogram (EEG) recordings. 

By converting multi-channel EEG time-series into 32-scale Complex Morlet Continuous Wavelet Transform (CWT) log-power scalograms, a spatial-spectral 2D Convolutional Neural Network (`EEG-CWT-Net`) learns pre-ictal neural hypersynchrony patterns prior to paroxysmal seizure manifestation.

---

## Key Experimental Benchmarks

### 1. Classification & Prediction Performance
* **Test Classification Accuracy**: **97.00 %**
* **Area Under ROC Curve (ROC-AUC)**: **0.9902**
* **Sensitivity (Pre-Ictal Detection)**: **92.11 %**
* **Specificity (Baseline Retention)**: **100.00 %**
* **Early Warning Lead Time**: **6.59 Minutes prior to clinical onset**

### 2. Time-Frequency Feature Extractor Comparison
| Method | Accuracy | Sensitivity | Specificity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **CWT Complex Morlet (Proposed)** | **100.00 %** | **100.00 %** | **100.00 %** | **1.0000** |
| STFT (Short-Time Fourier) | 96.00 % | 89.47 % | 100.00 % | 0.9796 |
| CWT Mexican Hat | 73.00 % | 97.62 % | 55.17 % | 0.9893 |

### 3. Artifact & Noise Robustness (SNR Sweep)
| Artifact Noise Type | SNR Level | Test Accuracy | Sensitivity | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **EOG (Ocular Blinks)** | $+15\text{ dB}$ | $99.00\%$ | $97.73\%$ | $1.0000$ |
| **EOG (Ocular Blinks)** | $0\text{ dB}$ | $96.00\%$ | $95.45\%$ | $0.9968$ |
| **EOG (Ocular Blinks)** | $-5\text{ dB}$ | $97.00\%$ | $92.86\%$ | $0.9947$ |
| **EMG (Muscle Artifact)** | $+15\text{ dB}$ | $97.00\%$ | $100.00\%$ | $1.0000$ |
| **EMG (Muscle Artifact)** | $0\text{ dB}$ | $95.00\%$ | $86.49\%$ | $0.9979$ |

---

## Reproduce in Google Colab

Run the complete pipeline directly in Google Colab:

```bash
git clone https://github.com/ashutoshmishra10182/eeg-cwt-seizure-predictor.git
cd eeg-cwt-seizure-predictor
python3 scripts/train_cwt_cnn.py
