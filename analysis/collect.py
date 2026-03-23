import json, glob, os

files = sorted(glob.glob("experiment_outputs/metrics_scenario_*.json"), key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]))
print("| Cenário | TP | FP | FN | Precisão | Recall | F1 |")
print("|---|---|---|---|---|---|---|")
for f in files:
    with open(f) as fl:
        m = json.load(fl)
    scen = os.path.basename(f).split("_")[-1].split(".")[0]
    p = f"{m.get('precision', 0):.2f}"
    r = f"{m.get('recall', 0):.2f}"
    f1 = f"{m.get('f1', 0):.2f}"
    print(f"| {scen} | {m.get('TP',0)} | {m.get('FP',0)} | {m.get('FN',0)} | {p} | {r} | {f1} |")
