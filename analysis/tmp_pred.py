import pandas as pd
import numpy as np

for i in [1, 4, 13]:
    df = pd.read_parquet(f"data/vibe_data_pred_{i}.parquet")
    incidents = []
    # Read true incidents quickly by hardcoding since I only want a rough idea
    # Scenario 1: 13:00 to 22:00
    if i == 1:
        mask = (df['sampled_at'] >= '2026-02-26T13:00:00+00:00') & (df['sampled_at'] <= '2026-02-26T22:00:00+00:00')
    elif i == 4:
        mask = ((df['sampled_at'] >= '2026-02-04T16:00:00+00:00') & (df['sampled_at'] <= '2026-02-05T04:00:00+00:00')) | \
               ((df['sampled_at'] >= '2026-02-27T00:00:00+00:00') & (df['sampled_at'] <= '2026-02-27T08:00:00+00:00'))
    elif i == 13:
        mask = ((df['sampled_at'] >= '2026-02-21T16:00:00+00:00') & (df['sampled_at'] <= '2026-02-21T19:00:00+00:00')) | \
               ((df['sampled_at'] >= '2026-02-22T15:00:00+00:00') & (df['sampled_at'] <= '2026-02-22T20:00:00+00:00'))
    
    inc_df = df[mask]
    norm_df = df[~mask]
    
    df_fit = pd.read_parquet(f"data/vibe_data_fit_{i}.parquet")
    fit_vel_norm = np.linalg.norm(df_fit[['vel_rms_x', 'vel_rms_y', 'vel_rms_z']], axis=1)
    fit_acc_norm = np.linalg.norm(df_fit[['accel_rms_x', 'accel_rms_y', 'accel_rms_z']], axis=1)
    
    is_on = fit_acc_norm > 0.05
    on_vel = fit_vel_norm[is_on]
    on_acc = fit_acc_norm[is_on]
    vel_mean, vel_std = on_vel.mean(), on_vel.std()
    acc_mean, acc_std = on_acc.mean(), on_acc.std()
    
    inc_acc = np.linalg.norm(inc_df[['accel_rms_x', 'accel_rms_y', 'accel_rms_z']], axis=1)
    inc_vel = np.linalg.norm(inc_df[['vel_rms_x', 'vel_rms_y', 'vel_rms_z']], axis=1)
    norm_acc = np.linalg.norm(norm_df[['accel_rms_x', 'accel_rms_y', 'accel_rms_z']], axis=1)
    norm_vel = np.linalg.norm(norm_df[['vel_rms_x', 'vel_rms_y', 'vel_rms_z']], axis=1)

    inc_on = inc_acc > 0.05
    inc_acc_z = np.abs((inc_acc[inc_on] - acc_mean) / acc_std)
    inc_vel_z = np.abs((inc_vel[inc_on] - vel_mean) / vel_std)
    inc_anom = (inc_acc_z > 3.0) | (inc_vel_z > 3.0)
    inc_anom_5 = (inc_acc_z > 5.0) | (inc_vel_z > 5.0)

    norm_on = norm_acc > 0.05
    norm_acc_z = np.abs((norm_acc[norm_on] - acc_mean) / acc_std)
    norm_vel_z = np.abs((norm_vel[norm_on] - vel_mean) / vel_std)
    norm_anom = (norm_acc_z > 3.0) | (norm_vel_z > 3.0)
    norm_anom_5 = (norm_acc_z > 5.0) | (norm_vel_z > 5.0)

    print(f"Scenario {i}:")
    print(f" INC Z>3 ratio: {inc_anom.mean():.3f}, Z>5 ratio: {inc_anom_5.mean():.3f}")
    if len(norm_on) > 0 and norm_on.sum() > 0:
        print(f" NRM Z>3 ratio: {norm_anom.mean():.3f}, Z>5 ratio: {norm_anom_5.mean():.3f}")
