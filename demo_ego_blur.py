import argparse
import os
from functools import lru_cache
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip


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
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    """Validate and preprocess the input arguments."""
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


def read_image(image_path: str) -> np.ndarray:
    """Read and return an image in BGR format."""
    return cv2.imread(image_path, cv2.IMREAD_COLOR)


def write_image(image: np.ndarray, image_path: str) -> None:
    """Write an image to the specified path."""
    cv2.imwrite(image_path, image)


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


def scale_box(box: List[float], image_width: int, image_height: int, scale: float) -> List[float]:
    """Scale a detection box by the specified factor."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xc, yc = x1 + w / 2, y1 + h / 2
    w, h = scale * w, scale * h
    return [
        max(0, xc - w / 2),
        max(0, yc - h / 2),
        min(image_width, xc + w / 2),
        min(image_height, yc + h / 2),
    ]


def visualize(image: np.ndarray, detections: List[List[float]], scale: float) -> np.ndarray:
    """Visualize detections on an image."""
    for box in detections:
        x1, y1, x2, y2 = map(int, scale_box(box, image.shape[1], image.shape[0], scale))
        image[y1:y2, x1:x2] = cv2.GaussianBlur(image[y1:y2, x1:x2], (51, 51), 0)
    return image


def process_image(args: argparse.Namespace, face_model, lp_model):
    """Process and visualize an image."""
    image = read_image(args.input_image_path)
    tensor = get_image_tensor(image)
    detections = []
    if face_model:
        detections += get_detections(face_model, tensor, args.face_model_score_threshold, args.nms_iou_threshold)
    if lp_model:
        detections += get_detections(lp_model, tensor, args.lp_model_score_threshold, args.nms_iou_threshold)
    result = visualize(image, detections, args.scale_factor_detections)
    write_image(result, args.output_image_path)


def process_video(args: argparse.Namespace, face_model, lp_model):
    """Process and visualize a video."""
    clip = VideoFileClip(args.input_video_path)
    frames = []
    for frame in clip.iter_frames(fps=args.output_video_fps, dtype="uint8"):
        tensor = get_image_tensor(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        detections = []
        if face_model:
            detections += get_detections(face_model, tensor, args.face_model_score_threshold, args.nms_iou_threshold)
        if lp_model:
            detections += get_detections(lp_model, tensor, args.lp_model_score_threshold, args.nms_iou_threshold)
        frames.append(visualize(frame, detections, args.scale_factor_detections))
    ImageSequenceClip(frames, fps=args.output_video_fps).write_videofile(args.output_video_path)


if __name__ == "__main__":
    args = parse_args()
    validate_args(args)
    face_model = load_detector(args.face_model_path)
    lp_model = load_detector(args.lp_model_path)
    if args.input_image_path:
        process_image(args, face_model, lp_model)
    if args.input_video_path:
        process_video(args, face_model, lp_model)
