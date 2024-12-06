import cProfile
import pstats
from pstats import SortKey
import argparse
import os
from functools import lru_cache
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip


def profile_visualization(func):
    """
    Decorator to profile the visualization functions.
    Profiles execution time and displays the top 20 time-consuming functions.
    """
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(20)  # Top 20 functions by cumulative time
        return result

    return wrapper


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_model_path", type=str, help="Path to face model file.")
    parser.add_argument("--lp_model_path", type=str, help="Path to license plate model file.")
    parser.add_argument("--face_model_score_threshold", type=float, default=0.9, help="Confidence threshold for face detections.")
    parser.add_argument("--lp_model_score_threshold", type=float, default=0.9, help="Confidence threshold for license plate detections.")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.3, help="NMS IoU threshold.")
    parser.add_argument("--scale_factor_detections", type=float, default=1.0, help="Scale factor for detection boxes.")
    parser.add_argument("--input_image_path", type=str, help="Path to input image.")
    parser.add_argument("--output_image_path", type=str, help="Path to save the output image.")
    parser.add_argument("--input_video_path", type=str, help="Path to input video.")
    parser.add_argument("--output_video_path", type=str, help="Path to save the output video.")
    parser.add_argument("--output_video_fps", type=int, default=30, help="FPS for the output video.")
    return parser.parse_args()


def create_directory_if_not_exists(file_path: str) -> None:
    """Create directory for the given file path if it does not exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def validate_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Validate input arguments."""
    if args.face_model_path and not os.path.exists(args.face_model_path):
        raise FileNotFoundError(f"Face model path does not exist: {args.face_model_path}")
    if args.lp_model_path and not os.path.exists(args.lp_model_path):
        raise FileNotFoundError(f"License plate model path does not exist: {args.lp_model_path}")
    if args.input_image_path and not os.path.exists(args.input_image_path):
        raise FileNotFoundError(f"Input image path does not exist: {args.input_image_path}")
    if args.input_video_path and not os.path.exists(args.input_video_path):
        raise FileNotFoundError(f"Input video path does not exist: {args.input_video_path}")
    if args.output_image_path:
        create_directory_if_not_exists(args.output_image_path)
    if args.output_video_path:
        create_directory_if_not_exists(args.output_video_path)
    return args


@lru_cache
def get_device() -> str:
    """Return the preferred device (CPU or GPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_detector(model_path: Optional[str]) -> Optional[torch.jit.ScriptModule]:
    """Load and return a TorchScript model from the given path."""
    if model_path:
        model = torch.jit.load(model_path, map_location=get_device())
        model.eval()
        return model
    return None


def get_image_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an image to a Torch tensor."""
    tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
    return tensor.to(get_device())


def get_detections(
    model: torch.jit.ScriptModule,
    image_tensor: torch.Tensor,
    score_threshold: float,
    nms_threshold: float,
) -> List[List[float]]:
    """Perform detections using the specified model."""
    with torch.no_grad():
        boxes, _, scores, _ = model(image_tensor)
        nms_indices = torchvision.ops.nms(boxes, scores, nms_threshold)
        filtered_boxes = boxes[nms_indices][scores[nms_indices] > score_threshold]
    return filtered_boxes.cpu().numpy().tolist()


def visualize(image: np.ndarray, detections: List[List[float]], scale: float) -> np.ndarray:
    """Visualize detections on an image."""
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (51, 51), 0)
    return image


@profile_visualization
def visualize_video(
    input_video_path: str,
    face_detector: Optional[torch.jit.ScriptModule],
    lp_detector: Optional[torch.jit.ScriptModule],
    face_model_score_threshold: float,
    lp_model_score_threshold: float,
    nms_iou_threshold: float,
    output_video_path: str,
    scale_factor_detections: float,
    output_video_fps: int,
):
    """Process and visualize a video."""
    clip = VideoFileClip(input_video_path)
    frames = []
    for frame in clip.iter_frames(fps=output_video_fps, dtype="uint8"):
        tensor = get_image_tensor(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        detections = []
        if face_detector:
            detections += get_detections(face_detector, tensor, face_model_score_threshold, nms_iou_threshold)
        if lp_detector:
            detections += get_detections(lp_detector, tensor, lp_model_score_threshold, nms_iou_threshold)
        frames.append(visualize(frame, detections, scale_factor_detections))
    ImageSequenceClip(frames, fps=output_video_fps).write_videofile(output_video_path)


if __name__ == "__main__":
    args = validate_inputs(parse_args())
    face_detector = load_detector(args.face_model_path)
    lp_detector = load_detector(args.lp_model_path)

    if args.input_image_path:
        # Add functionality for image visualization if needed
        pass

    if args.input_video_path:
        visualize_video(
            args.input_video_path,
            face_detector,
            lp_detector,
            args.face_model_score_threshold,
            args.lp_model_score_threshold,
            args.nms_iou_threshold,
            args.output_video_path,
            args.scale_factor_detections,
            args.output_video_fps,
        )
