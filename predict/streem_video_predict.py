import argparse
import cv2
import numpy as np
import pandas as pd
from collections import deque
import os
import time
from tqdm import tqdm
import torch

from model.tracknet_v4 import TrackNet

def parse_args():
    parser = argparse.ArgumentParser(description="Volleyball ball detection and tracking")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video file")
    parser.add_argument("--track_length", type=int, default=8, help="Length of the ball track (default: 8 frames)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save output video and CSV")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights file (e.g., outputs/exp_20250801_110350/checkpoints/best_model.pth)")
    parser.add_argument("--visualize", action="store_true", default=False, help="Enable visualization on display using cv2")
    parser.add_argument("--only_csv", action="store_true", default=False, help="Save only CSV, skip video output")
    return parser.parse_args()

def load_model(model_path, input_height=288, input_width=512):
    if not os.path.exists(model_path):
        raise ValueError(f"Model weights file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TrackNet().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def initialize_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, frame_width, frame_height, fps, total_frames

def setup_output_writer(video_basename, output_dir, frame_width, frame_height, fps, only_csv):
    if output_dir is None or only_csv:
        return None, None

    output_path = os.path.join(output_dir, f'{video_basename}_predict.mp4')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return out_writer, output_path

def setup_csv_file(video_basename, output_dir):
    if output_dir is None:
        return None
    csv_path = os.path.join(output_dir, f'{video_basename}_predict_ball.csv')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Initialize CSV with headers
    pd.DataFrame(columns=['Frame', 'Visibility', 'X', 'Y']).to_csv(csv_path, index=False)
    return csv_path

def append_to_csv(result, csv_path):
    if csv_path is None:
        return
    # Append single result to CSV
    pd.DataFrame([result]).to_csv(csv_path, mode='a', header=False, index=False)

def preprocess_frame(frame, input_height=288, input_width=512):
    frame = cv2.resize(frame, (input_width, input_height))
    frame = frame.astype(np.float32) / 255.0
    return frame

def preprocess_input(frame_buffer, input_height=288, input_width=512):          
    input_tensor = np.concatenate(frame_buffer, axis=2)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2))
    # Convert numpy array to PyTorch tensor
    input_tensor = torch.from_numpy(input_tensor).float()
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = input_tensor.to(device)
    return input_tensor 

def postprocess_output(output, threshold=0.5, input_height=288, input_width=512):
    # Now returns: (visibility, cx, cy, bbox_x, bbox_y, bbox_w, bbox_h)
    results = []
    for frame_idx in range(3):
        heatmap = output[frame_idx, :, :]
        _, binary = cv2.threshold(heatmap, threshold, 1.0, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours((binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(largest_contour)
                results.append((1, cx, cy, x, y, w, h))
            else:
                results.append((0, 0, 0, 0, 0, 0, 0))
        else:
            results.append((0, 0, 0, 0, 0, 0, 0))
    return results

def visualize_heatmaps(output, frame_index, input_height=288, input_width=512):
    for frame_idx in range(3):
        heatmap = output[frame_idx, :, :]
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        cv2.imshow(f'Heatmap Frame {frame_idx}', heatmap_color)
    cv2.waitKey(1)

def draw_track(frame, track_points, current_color=(0, 0, 255), history_color=(255, 0, 0), current_ball_bbox=None):
    for point in list(track_points)[:-1]:
        if point is not None:
            cv2.circle(frame, point, 5, history_color, -1)
    if track_points and track_points[-1] is not None:
        cv2.circle(frame, track_points[-1], 5, current_color, -1)
    # Draw green bounding box for current ball if provided
    # if current_ball_bbox is not None:
    #     x, y, box_w, box_h = current_ball_bbox
    #     cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
    return frame

  

def main():
    args = parse_args()
    input_width, input_height = 512, 288

    model = load_model(args.model_path, input_height, input_width)
    model.eval()
    
    cap, frame_width, frame_height, fps, total_frames = initialize_video(args.video_path)

    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    out_writer, _ = setup_output_writer(video_basename, args.output_dir, frame_width, frame_height, fps, args.only_csv)
    csv_path = setup_csv_file(video_basename, args.output_dir)

    processed_frame_buffer = deque(maxlen=3)
    frame_buffer = deque(maxlen=3)
    track_points = deque(maxlen=args.track_length)
    prediction_buffer = {}
    frame_index = 0

    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")
    stop = False
    while cap.isOpened() and stop == False:
        start_time = time.time()  # Start time for FPS calculation

        ret = None
        ret, frame = cap.read()
        # ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = preprocess_frame(frame_rgb)
        frame_buffer.append(frame)
        processed_frame_buffer.append(processed_frame)


        if len(processed_frame_buffer) < 3:
            continue

        if len(processed_frame_buffer) == 3:
            input_tensor = preprocess_input(processed_frame_buffer)
            # Set model to evaluation mode and make prediction
            
            with torch.no_grad():
                output = model(input_tensor)
            output = output.squeeze(0).cpu().numpy()

            predictions = postprocess_output(output, input_height=input_height, input_width=input_width)

            # Process all predictions in the list
            for idx, pred in enumerate(predictions):
                processed_frame_buffer.popleft()
                frame = frame_buffer.popleft()

                frame_index += 1
                visibility, x, y, bbox_x, bbox_y, bbox_w, bbox_h = pred
                current_ball_bbox = None
                if visibility == 0:
                    x_orig, y_orig = -1, -1
                    if len(track_points) > 0:
                        track_points.popleft()
                else:
                    x_orig = x * frame_width / input_width
                    y_orig = y * frame_height / input_height
                    track_points.append((int(x_orig), int(y_orig)))
                    # Scale the bounding box from input size to original frame size
                    bbox_x_orig = int(bbox_x * frame_width / input_width)
                    bbox_y_orig = int(bbox_y * frame_height / input_height)
                    bbox_w_orig = int(bbox_w * frame_width / input_width)
                    bbox_h_orig = int(bbox_h * frame_height / input_height)
                    current_ball_bbox = (bbox_x_orig, bbox_y_orig, bbox_w_orig, bbox_h_orig)

                result = {
                    'Frame': frame_index,
                    'Visibility': visibility,
                    'X': int(x_orig),
                    'Y': int(y_orig)
                }
                append_to_csv(result, csv_path)

                if args.visualize or out_writer is not None:
                    vis_frame = frame.copy()
                    vis_frame = draw_track(vis_frame, track_points, current_ball_bbox=current_ball_bbox)
                    if args.visualize:
                        #visualize_heatmaps(output, frame_index, input_height, input_width)
                        cv2.namedWindow(
                            "Tracking", cv2.WINDOW_NORMAL
                        )  # Create window with freedom of dimensions

                        cv2.imshow('Tracking', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            stop = True
                            break
                    if out_writer is not None:
                        out_writer.write(vis_frame)


            # Remove one element from frame_buffer after each iteration
            
        # Calculate and print FPS
        end_time = time.time()
        batch_time = end_time - start_time
        batch_fps = 1 / batch_time if batch_time > 0 else 0

        # Update progress bar (increment by 3 since we process 3 frames at a time)
        pbar.update(1)

    # Close progress bar
    pbar.close()

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
