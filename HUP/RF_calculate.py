import joblib
import numpy as np

rf = joblib.load("./models/RF_v4.joblib")
total_nodes = sum(t.tree_.node_count for t in rf.estimators_)
total_values = sum(t.tree_.value.size     for t in rf.estimators_)
total_floats = total_nodes + total_values
# 每个浮点数 8 字节（float64），若用 float32 则 4 字节
bytes_needed = total_floats * 8
print(f"Total floats: {total_floats}, RAM ≃ {bytes_needed/1024**2:.2f} MB")

