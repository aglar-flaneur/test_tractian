# Vibration Anomaly Detection - Challenge Report

## What went wrong in the baseline
So, taking a look at the given `AnomalyModel` and `AlertEngine`, the baseline F1 score was sitting around 0.31 (7 TP, 9 FP, 21 FN). Basically, it was missing almost everything. 
Here is a quick breakdown of why it was failing:

- **The "OFF" state was ruining the math.** The original `fit()` function just averaged out every single timestamp. Since industrial machines spend half their time turned off, this dragged the standard deviation through the roof and the mean down to near zero. So the Z-scores were totally useless.
- **Directional data was getting thrown away.** It took X, Y, and Z velocity and just crushed them into one single magnitude. Real faults (like bearing defects) are usually directional. Plus, it just completely ignored acceleration data—which is crazy because acceleration is literally how you detect high-frequency issues.
- **Permanent locking.** The `AlertEngine` would spot one anomaly and then permanently lock itself. Forever. Since the data has multiple events spread across different days in the same scenario, it just blatantly ignored all the later ones.
- **Bad timestamps.** Because of the rolling windows, the model kept tagging the *end* of a 4-hour block as the anomaly time. In a 12-hour batch, this pushed alerts hours past the actual event. So they wouldn't overlap the true labels properly.

## How I fixed it

I didn't want to break the pipeline rules, so I had to be creative with the model internals. 

### 1. Splitting up the axes
I completely rewrote the `AnomalyModel` to track the mean and std deviation for all 6 axes independently (both Accel and Velocity). When it predicts, it checks the Z-score for each axis on its own. If any single axis spikes past a threshold of 8.0 (which I set really high to filter out natural machine noise), it gets flagged.

### 2. Spotting when the machine is actually running
Since there is no `uptime` flag in the `DataPoint` object, I had to build a workaround. If you look at the raw data, total acceleration drops below `0.05g` whenever the machine stops. I just added an `uptime_acc_threshold` so the model literally ignores data unless the machine is awake. This instantly fixed the math.

### 3. Fixing the Alert Engine
I kept the batch checking, but I fixed the lockdown issue. Now, the engine unlocks (`self.locked = False`) the moment it sees a clean, healthy batch. This stops alert fatigue during an ongoing breakdown but still lets it catch a new event a few days later. I also made it trace back and return the exact timestamp of the worst local spike, making the labels line up perfectly.

### * Side note on Mahalanobis Distance
I actually spent some time trying to implement a Multivariate Gaussian Envelope (Mahalanobis Distance) to map the correlations between all the axes. Believe it or not, it performed worse. Industrial sensors drift over time due to wear and tear. That natural drift completely skewed the strict covariance matrix, causing a ton of false alarms. Treating the axes independently was way lighter on the CPU and drastically more reliable.

## Final Results

Here are the hard numbers after the changes:
- **True Positives:** 13
- **False Positives:** 11
- **False Negatives:** 15
- **Precision:** 0.542
- **Recall:** 0.464
- **Global F1 Score:** 0.500

That is pretty much a 60% relative improvement from the old 0.318 score. 

## If I had more time...
Assuming I could change the actual data pipeline, I'd probably add:
- **Rolling Baselines:** Using an EWMA to slowly update the baseline over time. Machines wear out, so their "normal" noise changes. Static Z-scores eventually fail on aging equipment.
- **FFT Analysis:** Time-domain data is tricky. Crunching the raw signals through a Fast Fourier Transform would highlight the exact fault frequencies much better.
- **Autoencoders:** If we had more labeled data, swapping this out for an Autoencoder neural net would easily catch non-linear vibration patterns that basic Z-scores miss.
