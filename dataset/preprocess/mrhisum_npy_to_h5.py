import os
import h5py
import numpy as np
from tqdm import tqdm
import json
import time
import argparse
from pathlib import Path

def get_video_list(split_file_path):
    with open(split_file_path, 'r') as f:
        splits = json.load(f)
    
    all_videos = []
    for split in ['train_keys', 'val_keys', 'test_keys']:
        all_videos.extend(splits[split])
    
    return sorted(all_videos)

def convert_feature_type(input_dir, output_hdf5_path, video_list, feature_name):
    print(f"\n🔄 Converting {feature_name}...")
    
    existing_videos = []
    missing_videos = []
    
    for vid in video_list:
        file_path = os.path.join(input_dir, f"{vid}.npy")
        if os.path.exists(file_path):
            existing_videos.append(vid)
        else:
            missing_videos.append(vid)
    
    print(f"✅ Found: {len(existing_videos)} videos")
    if missing_videos:
        print(f"⚠️  Missing: {len(missing_videos)} videos")
    
    with h5py.File(output_hdf5_path, 'w') as hf:
        hf.attrs['feature_type'] = feature_name
        hf.attrs['total_videos'] = len(existing_videos)
        hf.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        hf.attrs['missing_videos'] = str(missing_videos) if missing_videos else "None"
        
        for vid in tqdm(existing_videos, desc=f"Converting {feature_name}"):
            try:
                file_path = os.path.join(input_dir, f"{vid}.npy")
                data = np.load(file_path)
                
                hf.create_dataset(
                    vid, 
                    data=data, 
                    compression='gzip', 
                    compression_opts=6,
                    shuffle=True
                )
                
            except Exception as e:
                print(f"❌ Error processing {vid}: {str(e)}")
                continue
    
    print(f"✅ {feature_name} conversion completed: {output_hdf5_path}")
    
    file_size = os.path.getsize(output_hdf5_path) / (1024**3)
    print(f"📁 File size: {file_size:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='Convert MR.HiSum .npy files to HDF5 format')
    parser.add_argument('--audio', action='store_true', help='Convert audio features only')
    parser.add_argument('--visual', action='store_true', help='Convert visual features only')
    parser.add_argument('--melspec', action='store_true', help='Convert melspec features only')
    parser.add_argument('--gtscore', action='store_true', help='Convert gtscore features only')
    parser.add_argument('--all', action='store_true', help='Convert all features (default if no specific flag)')
    
    args = parser.parse_args()
    
    data_root = os.path.join(os.getenv('DATA_ROOT', './data'), 'mrhisum')
    split_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mrhisum_split.json'))
    
    all_conversions = {
        "audio_features": {
            "input_dir": os.path.join(data_root, "audio_emb"),
            "output_file": os.path.join(data_root, "audio_features.h5")
        },
        "visual_features": {
            "input_dir": os.path.join(data_root, "visual_emb"),
            "output_file": os.path.join(data_root, "visual_features.h5")
        },
        "melspec_features": {
            "input_dir": os.path.join(data_root, "audio", "melspec"),
            "output_file": os.path.join(data_root, "melspec_features.h5")
        },
        "gtscore": {
            "input_dir": os.path.join(data_root, "gtscore"),
            "output_file": os.path.join(data_root, "gtscore.h5")
        }
    }
    
    conversions_to_run = {}
    
    if args.audio:
        conversions_to_run["audio_features"] = all_conversions["audio_features"]
    if args.visual:
        conversions_to_run["visual_features"] = all_conversions["visual_features"]
    if args.melspec:
        conversions_to_run["melspec_features"] = all_conversions["melspec_features"]
    if args.gtscore:
        conversions_to_run["gtscore"] = all_conversions["gtscore"]
    
    if args.all or not conversions_to_run:
        conversions_to_run = all_conversions
    
    print("🚀 Starting HDF5 conversion...")
    print(f"📂 Data root: {data_root}")
    print(f"📋 Split file: {split_file}")
    print(f"🎯 Converting: {list(conversions_to_run.keys())}")
    
    try:
        video_list = get_video_list(split_file)
        print(f"📊 Total videos in split file: {len(video_list)}")
    except Exception as e:
        print(f"❌ Error reading split file: {str(e)}")
        return
    
    for name, config in conversions_to_run.items():
        output_path = config["output_file"]
        if os.path.exists(output_path):
            response = input(f"⚠️  {output_path} already exists. Overwrite? (y/N): ")
            if response.lower() != 'y':
                print(f"❌ Skipping {name} conversion.")
                del conversions_to_run[name]
    
    if not conversions_to_run:
        print("❌ No conversions to run.")
        return
    
    start_time = time.time()
    
    for feature_name, config in conversions_to_run.items():
        input_dir = config["input_dir"]
        output_path = config["output_file"]
        
        if not os.path.exists(input_dir):
            print(f"❌ Input directory not found: {input_dir}")
            continue
            
        convert_feature_type(input_dir, output_path, video_list, feature_name)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Selected conversions completed in {total_time/60:.1f} minutes!")
    
    print("\n📊 Converted file sizes:")
    total_size = 0
    for name, config in conversions_to_run.items():
        output_path = config["output_file"]
        if os.path.exists(output_path):
            size = os.path.getsize(output_path) / (1024**3)
            total_size += size
            print(f"  {name}: {size:.2f} GB")
    
    print(f"📁 Total converted size: {total_size:.2f} GB")

if __name__ == "__main__":
    main()