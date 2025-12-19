"""Badminton Dataset Preprocessor

Usage:
    python preprocess.py --config config.yaml
"""

import gc
import os
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.stats import multivariate_normal
from tqdm import tqdm

IGNORED_FILES = {'.DS_Store', 'Thumbs.db', '.gitignore', '.gitkeep'}
IGNORED_DIRS = {'.git', '__pycache__', '.vscode', '.idea', 'node_modules'}
TARGET_WIDTH = 512
TARGET_HEIGHT = 288
JPEG_QUALITY = 95


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)['preprocess']


def is_valid_path(name):
    if name.startswith('.') and name not in {'.', '..'}:
        return False
    return name not in IGNORED_FILES and name not in IGNORED_DIRS


def generate_heatmap(center_x, center_y, width=TARGET_WIDTH, height=TARGET_HEIGHT, sigma=3):
    x_coords = np.arange(0, width)
    y_coords = np.arange(0, height)
    mesh_x, mesh_y = np.meshgrid(x_coords, y_coords)
    coordinates = np.dstack((mesh_x, mesh_y))
    gaussian_mean = [center_x, center_y]
    covariance_matrix = [[sigma ** 2, 0], [0, sigma ** 2]]
    distribution = multivariate_normal(gaussian_mean, covariance_matrix)
    heatmap = distribution.pdf(coordinates)
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
    return heatmap_uint8


def resize_with_aspect_ratio(image, target_w=TARGET_WIDTH, target_h=TARGET_HEIGHT):
    original_h, original_w = image.shape[:2]
    scale_width = target_w / original_w
    scale_height = target_h / original_h
    scale_factor = min(scale_width, scale_height)
    new_width = int(original_w * scale_factor)
    new_height = int(original_h * scale_factor)
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    offset_x = (target_w - new_width) // 2
    offset_y = (target_h - new_height) // 2
    canvas[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = resized_img
    return canvas, scale_factor, offset_x, offset_y


def transform_annotation_coords(x, y, scale, offset_x, offset_y):
    transformed_x = x * scale + offset_x
    transformed_y = y * scale + offset_y
    return transformed_x, transformed_y


def validate_dataset_structure(source_path):
    if not os.path.exists(source_path):
        return False, f"Source path does not exist: {source_path}"
    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    match_dirs = [
        item for item in entries
        if item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]
    if not match_dirs:
        return False, "No match directories found"
    valid_matches = 0
    video_count = 0
    annotation_count = 0
    for match_dir in match_dirs:
        match_path = os.path.join(source_path, match_dir)
        annotations_path = os.path.join(match_path, "csv")
        videos_path = os.path.join(match_path, "video")
        if os.path.exists(annotations_path) and os.path.exists(videos_path):
            valid_matches += 1
            csv_files = [f for f in os.listdir(annotations_path)
                         if f.endswith('_ball.csv') and is_valid_path(f)]
            annotation_count += len(csv_files)
            mp4_files = [f for f in os.listdir(videos_path)
                         if f.endswith('.mp4') and is_valid_path(f)]
            video_count += len(mp4_files)
    if valid_matches == 0:
        return False, "No valid match directories found (must contain both csv and video subdirectories)"
    summary = f"Found {valid_matches} match directories, {video_count} videos, {annotation_count} annotation files"
    return True, summary


def estimate_total_frames(source_path):
    frame_total = 0
    match_dirs = [
        item for item in os.listdir(source_path) if is_valid_path(item)
        and item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]
    for match_dir in match_dirs:
        match_path = os.path.join(source_path, match_dir)
        videos_path = os.path.join(match_path, "video")
        if os.path.exists(videos_path):
            mp4_files = [f for f in os.listdir(videos_path)
                         if f.endswith('.mp4') and is_valid_path(f)]
            for mp4_file in mp4_files:
                video_path = os.path.join(videos_path, mp4_file)
                video_capture = cv2.VideoCapture(video_path)
                if video_capture.isOpened():
                    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_total += total_frames
                video_capture.release()
    return frame_total


def process_video_sequence(video_path, annotation_path, inputs_output_dir, heatmaps_output_dir,
                           sequence_name, sigma_value, progress_bar):
    sequence_inputs_dir = os.path.join(inputs_output_dir, sequence_name)
    sequence_heatmaps_dir = os.path.join(heatmaps_output_dir, sequence_name)
    os.makedirs(sequence_inputs_dir, exist_ok=True)
    os.makedirs(sequence_heatmaps_dir, exist_ok=True)
    if not os.path.exists(annotation_path):
        tqdm.write(f"Warning: Annotation file not found {annotation_path}")
        return 0
    try:
        annotations_df = pd.read_csv(annotation_path)
    except Exception as e:
        tqdm.write(f"Error: Cannot read annotation file {annotation_path}: {e}")
        return 0
    video_stream = cv2.VideoCapture(video_path)
    if not video_stream.isOpened():
        tqdm.write(f"Error: Cannot open video {video_path}")
        return 0
    frames_processed = 0
    current_frame = 0
    encoding_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    annotation_lookup = {}
    for _, row in annotations_df.iterrows():
        annotation_lookup[row['Frame']] = row
    try:
        while True:
            frame_available, frame_data = video_stream.read()
            if not frame_available:
                break
            progress_bar.update(1)
            if current_frame in annotation_lookup:
                annotation_row = annotation_lookup[current_frame]
                processed_frame, scale_factor, x_offset, y_offset = resize_with_aspect_ratio(frame_data)
                if annotation_row['Visibility'] == 1:
                    original_x = annotation_row['X']
                    original_y = annotation_row['Y']
                    if pd.isna(original_x) or pd.isna(original_y):
                        heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
                    else:
                        transformed_x, transformed_y = transform_annotation_coords(
                            original_x, original_y, scale_factor, x_offset, y_offset
                        )
                        transformed_x = max(0, min(TARGET_WIDTH - 1, transformed_x))
                        transformed_y = max(0, min(TARGET_HEIGHT - 1, transformed_y))
                        heatmap = generate_heatmap(transformed_x, transformed_y, sigma=sigma_value)
                else:
                    heatmap = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
                frame_output_path = os.path.join(sequence_inputs_dir, f"{current_frame}.jpg")
                heatmap_output_path = os.path.join(sequence_heatmaps_dir, f"{current_frame}.jpg")
                cv2.imwrite(frame_output_path, processed_frame, encoding_params)
                cv2.imwrite(heatmap_output_path, heatmap)
                frames_processed += 1
                del processed_frame, heatmap
            current_frame += 1
            if current_frame % 100 == 0:
                gc.collect()
    finally:
        video_stream.release()
    return frames_processed


def process_match_directory(match_path, output_base_dir, sigma_value, progress_bar):
    match_name = os.path.basename(match_path)
    match_output_dir = os.path.join(output_base_dir, match_name)
    inputs_output_dir = os.path.join(match_output_dir, "inputs")
    heatmaps_output_dir = os.path.join(match_output_dir, "heatmaps")
    os.makedirs(inputs_output_dir, exist_ok=True)
    os.makedirs(heatmaps_output_dir, exist_ok=True)
    videos_dir = os.path.join(match_path, "video")
    annotations_dir = os.path.join(match_path, "csv")
    if not os.path.exists(videos_dir) or not os.path.exists(annotations_dir):
        tqdm.write(f"Warning: Missing video or csv directory in {match_name}")
        return
    mp4_files = [f for f in os.listdir(videos_dir)
                 if f.endswith('.mp4') and is_valid_path(f)]
    sequences_processed = 0
    total_frames_processed = 0
    for mp4_file in mp4_files:
        video_path = os.path.join(videos_dir, mp4_file)
        sequence_name = Path(mp4_file).stem
        annotation_filename = f"{sequence_name}_ball.csv"
        annotation_path = os.path.join(annotations_dir, annotation_filename)
        if os.path.exists(annotation_path):
            frames_count = process_video_sequence(
                video_path, annotation_path, inputs_output_dir, heatmaps_output_dir,
                sequence_name, sigma_value, progress_bar
            )
            if frames_count > 0:
                sequences_processed += 1
                total_frames_processed += frames_count
        else:
            tqdm.write(f"Warning: Annotation file not found for {mp4_file}")
    tqdm.write(f"Completed {match_name}: {sequences_processed} sequences, {total_frames_processed} frames")


def preprocess_dataset(cfg):
    source_path = cfg['source']
    output_path = cfg['output']
    sigma_value = cfg['sigma']
    force_overwrite = cfg['force']
    
    structure_valid, validation_message = validate_dataset_structure(source_path)
    if not structure_valid:
        print(f"Error: {validation_message}")
        return False
    print(f"Validated: {validation_message}")
    if os.path.exists(output_path):
        if force_overwrite:
            print(f"Removing existing directory: {output_path}")
            shutil.rmtree(output_path)
        else:
            user_input = input(f"Output directory exists: {output_path}\nDelete and rebuild? (y/n): ")
            if user_input.lower() != 'y':
                print("Operation cancelled")
                return False
            shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)
    entries = [item for item in os.listdir(source_path) if is_valid_path(item)]
    match_dirs = [
        item for item in entries
        if item.startswith("match") and os.path.isdir(os.path.join(source_path, item))
    ]
    valid_match_dirs = []
    for match_dir_name in match_dirs:
        match_dir_path = os.path.join(source_path, match_dir_name)
        has_csv = os.path.exists(os.path.join(match_dir_path, "csv"))
        has_video = os.path.exists(os.path.join(match_dir_path, "video"))
        if has_csv and has_video:
            valid_match_dirs.append(match_dir_name)
        else:
            print(f"Skipping {match_dir_name}: missing csv or video directory")
    if not valid_match_dirs:
        print("Error: No valid match directories found")
        return False
    print(f"Processing {len(valid_match_dirs)} match directories...")
    total_frame_count = estimate_total_frames(source_path)
    print(f"Total frames to process: {total_frame_count}")
    with tqdm(total=total_frame_count, desc="Processing frames", unit="frame") as progress_bar:
        for match_dir_name in valid_match_dirs:
            match_dir_path = os.path.join(source_path, match_dir_name)
            process_match_directory(match_dir_path, output_path, sigma_value, progress_bar)
            gc.collect()
    print(f"Preprocessing completed!")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print(f"Heatmap sigma: {sigma_value}")
    return True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    success = preprocess_dataset(config)
    sys.exit(0 if success else 1)
