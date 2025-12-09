# AF-Detection-Single-Lead-ECG
Deep learning-based Atrial Fibrillation detection using single-lead ECG signals. Includes preprocessing (denoising, beat segmentation) and a CNN - BiLSTM model to capture spatial ECG features and temporal rhythm patterns. Designed for accurate beat-level AF classification toward wearable cardiac monitoring.

This project develops a deep learning-based approach to automatically detect Atrial Fibrillation (AF) using single-lead ECG recordings. The goal is to support early arrhythmia diagnosis for wearable and portable cardiac health monitoring.


Project Overview

ECG preprocessing: baseline wander removal, noise reduction, beat segmentation

Hybrid CNN + LSTM architecture for spatial + temporal feature learning

Beat-level classification for AF vs Normal Sinus Rhythm

Designed to be lightweight and real-time deployable
