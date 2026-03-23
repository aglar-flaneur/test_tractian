import yaml
import subprocess
import json

def run_experiment(z, r):
    params = {
        'z_threshold': z,
        'window_anomaly_ratio': r,
        'uptime_acc_threshold': 0.05
    }
    with open('hyperparameters/model_hyperparams.yaml', 'w') as f:
        yaml.dump(params, f)
    
    subprocess.run(["uv", "run", "main.py", "--no-plot"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    with open('experiment_outputs/metrics_all_scenarios.json', 'r') as f:
        metrics = json.load(f)
        
    return metrics

best_f1 = 0
best_params = {}

for z in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
    for r in [0.05, 0.1]:
        metrics = run_experiment(z, r)
        f1 = metrics['f1_global']
        print(f"z={z}, r={r} -> F1: {f1:.3f} (TP:{metrics['TP_total']}, FP:{metrics['FP_total']}, FN:{metrics['FN_total']})")
        if f1 > best_f1:
            best_f1 = f1
            best_params = {'z': z, 'r': r, 'metrics': metrics}

print("BEST:", best_params)
