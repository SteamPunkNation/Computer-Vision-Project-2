import pandas as pd

def calculate_metrics(file_path):
    df = pd.read_csv(file_path)
    # The columns are: Timestamp, FPS, Delay_ms, TrackedFeatures, Pose_X, Pose_Z, Motion, Rotation
    avg_fps = df['FPS'].mean()
    avg_delay = df['Delay_ms'].mean()
    avg_features = df['TrackedFeatures'].mean()
    print(f"File: {file_path}")
    print(f"Avg FPS: {avg_fps:.2f}")
    print(f"Avg Delay: {avg_delay:.2f} ms")
    print(f"Avg Features: {avg_features:.2f}")
    print("-" * 30)

calculate_metrics("generated/logs/metrics_log_20260725_224151.csv")
calculate_metrics("generated/logs/metrics_log_20260725_224248.csv")
