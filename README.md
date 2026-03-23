# Vibration Anomaly Detection - Challenge Report

## Baseline Analysis
When I first reviewed the `AnomalyModel` and `AlertEngine`, I found a few major problems that limited their performance (Baseline F1: ~0.31, TP: 7, FP: 9, FN: 21):

1. **Uptime Imbalance**: The original `AnomalyModel.fit()` used all data (both "ON" and "OFF" states) to calculate the mean and standard deviation. Since the machines are often turned off (where the vibration is close to zero), this pulled the mean down and made the standard deviation far too large. This broke the Z-score calculation, causing the model to miss real anomalies (False Negatives) and trigger false alarms during normal operations (False Positives).
2. **Loss of Directional Data**: The original model combined the X, Y, and Z velocity readings into a single overall number (magnitude). For vibration analysis, specific machine problems (like unbalance) usually happen in specific directions. Combining the axes hides the actual problem. In addition, the original model completely ignored Acceleration data, which is essential to detect high-frequency problems like bearing faults.
3. **Permanent Alert Locking**: The original `AlertEngine` locked itself permanently after finding the very first anomaly. Because many machines in the dataset have multiple separate incidents over several days, this logic guaranteed that the system would miss all future incidents, resulting in a high number of False Negatives.
4. **Wrong Timestamp Alignment**: The original model used the end time of the 4-hour window as the anomaly timestamp. When processed inside overlapping 12-hour engine batches, this caused the final timestamp to shift hours away from the real short incidents. This made true alerts fall outside the correct incident time window.

## Methodology

To fix these issues within the strict rules of the challenge, I made the following changes:

### 1. Tracking Each Axis Independently
I updated the `AnomalyModel` to calculate and store the mean and standard deviation for each axis separately (both Acceleration and Velocity). The `predict` function now calculates positive Z-scores for all 6 axes independently. If the vibration on *any* single axis goes above the Z-threshold (which I increased to 8.0 to ignore normal noise), I flag it as an anomaly.

### 2. Filtering the "ON" State
Since the `DataPoint` structure does not come with the `uptime` (ON/OFF) information, I had to find a reliable way to detect it. By looking at the normal data, I saw that the total acceleration consistently drops below `0.05g` when a machine is turned off. I added this `uptime_acc_threshold` to the model. Now, it only calculates statistics and Z-scores when the machine is actually running.

### 3. Fixing the `AlertEngine` Logic
The `AlertEngine` still checks if any prediction in the batch is anomalous. However, I fixed the locking system. Now, it unlocks (`self.locked = False`) as soon as it receives a normal batch with no anomalies. This successfully prevents alert spamming during an ongoing incident but allows the system to alert again if a new incident happens days later. Also, the engine now returns the exact timestamp of the peak anomaly from the window, making sure my alerts correctly overlap the true incident labels.

### 4. Advanced Multivariate Testing (Mahalanobis Distance)
During my tests, I also tried a more advanced Machine Learning approach called Mahalanobis Distance (Multivariate Gaussian). This technique looks at the relationships between all 6 axes at the same time. However, it gave me a worse F1 score than the independent per-axis Z-score. This happens because industrial sensors have normal physical variations over time (sensor drift). These small, natural changes completely messed up the strict math of the Mahalanobis matrix, causing too many false alarms. Treating each axis independently works much better and faster for this type of industrial noise.

## Performance

My improved model gave the following final results:
- **True Positives**: 13
- **False Positives**: 11
- **False Negatives**: 15
- **Precision**: 0.542
- **Recall**: 0.464
- **Global F1 Score**: 0.500

This is a nearly 60% relative improvement over the original F1 score of 0.318. I successfully balanced a big increase in the detection of true anomalies while using a strict Z-threshold to prevent false alarms.

## Future Work

If I had more time and could change the pipeline structure, I would try:
- **Rolling Windows Baseline**: Real machines get older and their "normal" behavior changes over time. Using a moving average (like EWMA) would constantly update what "normal" means and prevent old machines from triggering false alarms.
- **Frequency Domain (FFT)**: Analyzing the sound frequencies (FFT) of the raw 10-minute signals would be much better at isolating specific mechanical faults than just looking at the overall time-domain amplitude.
- **Machine Learning**: With more labeled data, I could replace the basic Z-score with a lightweight Autoencoder or One-Class SVM. This would help detect complex, non-linear vibration patterns across different axes.
