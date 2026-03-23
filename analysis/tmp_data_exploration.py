import pandas as pd
import numpy as np

# Load a scenario
for i in [1, 4, 13]:
    df = pd.read_parquet(f"data/vibe_data_fit_{i}.parquet")
    print(f"Scenario {i}")
    vel_norm = np.linalg.norm(df[['vel_rms_x', 'vel_rms_y', 'vel_rms_z']], axis=1)
    acc_norm = np.linalg.norm(df[['accel_rms_x', 'accel_rms_y', 'accel_rms_z']], axis=1)
    
    print(df['uptime'].value_counts())
    
    on_mask = df['uptime'] == True
    off_mask = df['uptime'] == False
    
    print("ON vel norm  - min:", vel_norm[on_mask].min(), "max:", vel_norm[on_mask].max(), "mean:", vel_norm[on_mask].mean())
    if off_mask.sum() > 0:
        print("OFF vel norm - min:", vel_norm[off_mask].min(), "max:", vel_norm[off_mask].max(), "mean:", vel_norm[off_mask].mean())
    
    print("ON acc norm  - min:", acc_norm[on_mask].min(), "max:", acc_norm[on_mask].max(), "mean:", acc_norm[on_mask].mean())
    if off_mask.sum() > 0:
        print("OFF acc norm - min:", acc_norm[off_mask].min(), "max:", acc_norm[off_mask].max(), "mean:", acc_norm[off_mask].mean())
    print("-" * 40)
