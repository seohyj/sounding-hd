import h5py
import numpy as np
import os
from tqdm import tqdm

data_root = os.path.join(os.getenv('DATA_ROOT', './data'), 'mrhisum')
h5_path = os.path.join(data_root, "mrhisum.h5")
output_dir = os.path.join(data_root, "gtscore")
os.makedirs(output_dir, exist_ok=True)

with h5py.File(h5_path, 'r') as f:
    for video_id in tqdm(f.keys(), desc="Saving gtscore"):
        try:
            gtscore = f[f"{video_id}/gtscore"][:]
            np.save(os.path.join(output_dir, f"{video_id}.npy"), gtscore)
        except Exception as e:
            print(f"[{video_id}] Failed to save: {e}")

print("Done!")
print(f"Number of files saved: {len(os.listdir(output_dir))}")
