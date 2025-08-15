import json, statistics
from pathlib import Path
import pyedflib
import pandas as pd

ROOT = Path("/home/zhuoying/projects/def-xilinliu/data/UPenn_data_bids")
RUN_LEN = 3600

def edf_info(p: Path):
    f = pyedflib.EdfReader(str(p))
    try:
        labels = f.getSignalLabels()
        sfs = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]
    finally:
        f.close()
    return labels, sfs

def write_json(json_path: Path, sf_mode: int):
    if json_path.exists():
        try: j = json.loads(json_path.read_text())
        except: j = {}
        j.setdefault("SamplingFrequency", int(sf_mode))
        json_path.write_text(json.dumps(j, indent=2)); return
    j = {"TaskName":"monitor","SamplingFrequency":int(sf_mode),"PowerLineFrequency":60,
         "EEGReference":"Pz","RecordingType":"continuous","Manufacturer":"Natus",
         "EEGPlacementScheme":"10-20","SourceFormat":"EDF"}
    json_path.write_text(json.dumps(j, indent=2))

def write_channels(tsv_path: Path, labels, sfs, reference="Pz"):
    rows=[]
    for nm, sf in zip(labels, sfs):
        u=nm.upper()
        ch_type = "ECG" if u in {"EKG","ECG"} else ("EOG" if u in {"LOC","ROC"} else "EEG")
        rows.append({"name":nm,"type":ch_type,"units":"uV","sampling_frequency":sf,
                     "reference":reference,"status":"good","status_description":""})
    pd.DataFrame(rows, columns=["name","type","units","sampling_frequency",
                                "reference","status","status_description"]).to_csv(tsv_path, sep="\t", index=False)

def ensure_events_header(tsv_path: Path):
    if not tsv_path.exists():
        pd.DataFrame(columns=["onset","duration","trial_type","annotation","notes"]).to_csv(tsv_path, sep="\t", index=False)

print(f"[INFO] ROOT={ROOT}")
subs = sorted(ROOT.glob("sub-*"))
print(f"[INFO] Found {len(subs)} subjects: {[p.name for p in subs]}")

for sub in subs:
    ses_dir = sub / "ses-phaseII" / "eeg"
    if not ses_dir.exists():
        print(f"[WARN] {ses_dir} missing; skipping {sub.name}")
        continue
    runs = sorted(ses_dir.glob(f"{sub.name}_ses-phaseII_*_run-*_eeg.edf"))
    print(f"[INFO] {sub.name}: {len(runs)} EDF runs found")
    scans_rows = []

    for edf in runs:
        print(f"[INFO] Processing {edf.name}")
        try:
            labels, sfs = edf_info(edf)
            sf_mode = int(statistics.mode(sfs))
        except Exception as e:
            print(f"[ERROR] Cannot read {edf}: {e}"); continue

        stem = edf.stem
        json_path = edf.with_suffix(".json")
        ch_tsv = edf.with_name(stem + "_channels.tsv")
        ev_tsv = edf.with_name(stem + "_events.tsv")

        write_json(json_path, sf_mode)
        write_channels(ch_tsv, labels, sfs, reference="Pz")
        ensure_events_header(ev_tsv)

        run_part = next((p for p in stem.split("_") if p.startswith("run-")), None)
        if run_part:
            r = int(run_part.split("-")[1])
            scans_rows.append([f"eeg/{edf.name}", (r-1)*RUN_LEN])

    if scans_rows:
        scans = pd.DataFrame(scans_rows, columns=["filename","relative_start_sec"])
        scans_path = sub / "ses-phaseII" / f"{sub.name}_ses-phaseII_scans.tsv"
        scans.to_csv(scans_path, sep="\t", index=False)
        print(f"[INFO] Wrote {scans_path}")
