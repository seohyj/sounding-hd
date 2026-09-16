import tensorflow as tf
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils import load_img, img_to_array
from tqdm import tqdm
import re
import pickle
import cv2

_DATA_ROOT = os.path.join(os.getenv('DATA_ROOT', './data'), 'mrhisum')
VIDEO_DIR = os.path.join(_DATA_ROOT, 'video')
OUTPUT_DIR = os.path.join(_DATA_ROOT, 'visual_emb')
PCA_MATRIX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_matrix.npy')

def create_feature_extractor():
    """
    Creates an InceptionV3 model pre-trained on ImageNet and modifies it
    to output features from the 'avg_pool' layer.

    Returns:
        A Keras Model object that takes an image as input and outputs a 2048-dim vector.
    """
    base_model = InceptionV3(weights='imagenet', include_top=True)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
    return model

def extract_features(video_path, output_path, model, pca_matrix):
    """
    Extracts frames from a video at 1 FPS, processes them through the InceptionV3
    model, applies PCA, and saves the resulting features to a .npy file.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to save the output .npy file.
        model (keras.Model): The feature extraction model.
        pca_matrix (np.array): The PCA projection matrix (2048x1024).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video file, skipping: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if video_fps == 0:
        print(f"Warning: Video has 0 FPS, skipping: {video_path}")
        cap.release()
        return
        
    frame_interval = int(round(video_fps))
    
    final_features = []
    
    print(f"\nProcessing video: {os.path.basename(video_path)}")
    print(f"Original FPS: {video_fps:.2f}, Capturing 1 frame every {frame_interval} frames.")

    with tqdm(total=total_frames, unit='frames', desc=os.path.basename(video_path)) as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                img_resized = cv2.resize(frame, (299, 299))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_expanded = np.expand_dims(img_rgb, axis=0)
                img_preprocessed = preprocess_input(img_expanded)
                
                features_2048d = model.predict(img_preprocessed, verbose=0)
                
                features_1024d = np.dot(features_2048d, pca_matrix)
                
                final_features.append(features_1024d.squeeze())
            
            pbar.update(1)
            frame_idx += 1
            
    cap.release()

    final_features_array = np.array(final_features)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final_features_array)
    
    print("-" * 50)
    print("Feature extraction complete.")
    print(f"Shape of saved features: {final_features_array.shape}")
    print(f"Saved to: {output_path}")

def main():
    print("Loading InceptionV3 model...")
    model = create_feature_extractor()

    try:
        print("Loading PCA matrix...")
        pca_matrix = np.load(PCA_MATRIX_PATH)
        if pca_matrix.shape != (2048, 1024):
             print(f"Error: PCA matrix shape is {pca_matrix.shape}, but expected (2048, 1024).")
             return
    except FileNotFoundError:
        print(f"Error: PCA matrix file not found at {PCA_MATRIX_PATH}")
        return

    try:
        video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
        if not video_files:
            print(f"No .mp4 files found in the specified directory: {VIDEO_DIR}")
            return
    except FileNotFoundError:
        print(f"Error: Video directory not found at {VIDEO_DIR}")
        return

    print(f"\nFound {len(video_files)} video(s) to process in '{VIDEO_DIR}'")
    
    for video_filename in video_files:
        video_path = os.path.join(VIDEO_DIR, video_filename)
        
        output_filename = os.path.splitext(video_filename)[0] + '.npy'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        extract_features(video_path, output_path, model, pca_matrix)

if __name__ == '__main__':
    main()