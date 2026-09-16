import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_tvsum_annotations(tsv_path, metadata_path, output_dir):
    """
    Processes TVSum annotations to generate score files that are:
    1. Averaged over 20 annotators.
    2. Normalized to a 0-1 range.
    3. Precisely resampled to a 1-FPS timeline using metadata.
    """
    print(f"Reading annotations from: {tsv_path}")
    print(f"Reading metadata from: {metadata_path}")

    try:
        df = pd.read_csv(tsv_path, sep='\t', header=None, names=['video_id', 'category', 'scores'])
        meta_df = pd.read_csv(metadata_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Required file not found: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    video_groups = df.groupby('video_id')

    for video_id, group in tqdm(video_groups, desc="Processing videos"):
        
        meta_row = meta_df[meta_df["video_id"] == video_id]
        if meta_row.empty:
            print(f"\n[Warning] Metadata for {video_id} not found. Skipping.")
            continue
        
        length_str = meta_row["length"].values[0]
        minutes, seconds = map(int, length_str.split(':'))
        video_duration_seconds = minutes * 60 + seconds
        target_length = video_duration_seconds

        if target_length == 0:
            print(f"\n[Warning] Video {video_id} has zero length. Skipping.")
            continue

        all_scores_for_video = [
            [int(s) for s in scores_str.split(',')] 
            for scores_str in group['scores']
        ]
        
        scores_array = np.array(all_scores_for_video)
        avg_scores = scores_array.mean(axis=0)
        normalized_scores = (avg_scores - 1) / 4.0

        original_time_points = np.linspace(0, target_length - 1, num=len(normalized_scores))
        
        target_time_points = np.arange(target_length)
        
        final_scores = np.interp(target_time_points, original_time_points, normalized_scores)

        output_path = os.path.join(output_dir, f"{video_id}.npy")
        np.save(output_path, final_scores.astype(np.float32))

    print(f"\nSuccessfully processed {len(video_groups)} videos.")
    print(f"Averaged and resampled ground truth scores saved in: {output_dir}")

def main():
    data_root = os.path.join(os.getenv('DATA_ROOT', './data'), 'tvsum')
    tsv_path = os.path.join(data_root, 'data/ydata-tvsum50-anno.tsv')
    metadata_path = os.path.join(data_root, 'tvsum_metadata.csv')
    output_dir = os.path.join(data_root, 'tvsum_gtscore')

    process_tvsum_annotations(tsv_path, metadata_path, output_dir)

if __name__ == "__main__":
    main()
