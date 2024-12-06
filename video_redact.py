import cProfile
import pstats
from pstats import SortKey
import argparse
import os
from functools import lru_cache
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

def profile_visualization(func):
    """
    Decorator to profile the visualization functions
    """

    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        # Print the stats sorted by cumulative time
        stats = pstats.Stats(profiler)
        stats.sort_stats(SortKey.CUMULATIVE)
        stats.print_stats(20)  # Print top 20 time-consuming functions

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


@profile_visualization
def visualize_video(
        input_video_path: str,
        face_detector: torch.jit._script.RecursiveScriptModule,
        lp_detector: torch.jit._script.RecursiveScriptModule,
        face_model_score_threshold: float,
        lp_model_score_threshold: float,
        nms_iou_threshold: float,
        output_video_path: str,
        scale_factor_detections: float,
        output_video_fps: int,
):
    video_reader_clip = VideoFileClip(input_video_path)
    print(f"Video Resolution: {video_reader_clip.size}")
    print(f"Video Duration: {video_reader_clip.duration}s")
    print(f"Video FPS: {video_reader_clip.fps}")
    n_frames = int(video_reader_clip.duration * video_reader_clip.fps)
    print(f"Total Frames: {n_frames}")

    # Initialize video writer first
    height, width = video_reader_clip.size[::-1]  # Reverse to get height, width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, video_reader_clip.fps, (width, height), isColor=True)

    frame_counter = 0
    # Adaptive frame skip based on resolution
    if width * height > 1920 * 1080:  # 4K or higher
        frame_skip = 4  # Process every 4th frame
    else:
        frame_skip = 3  # Standard HD
    prev_processed_frame = None

    device = get_device()
    with torch.cuda.amp.autocast(), torch.no_grad():
        for frame in video_reader_clip.iter_frames():
            frame_counter += 1

            if frame_counter % frame_skip != 0:
                if prev_processed_frame is not None:
                    out.write(prev_processed_frame)
                continue

            # Process frame with pre-allocated memory
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            with torch.cuda.stream(torch.cuda.Stream()):  # Use separate CUDA stream
                image_tensor = torch.from_numpy(np.transpose(bgr_frame, (2, 0, 1))).to(device, non_blocking=True)

                detections = []
                if face_detector is not None:
                    face_detections = get_detections(face_detector, image_tensor,
                                                     face_model_score_threshold, nms_iou_threshold)
                    detections.extend(face_detections)

                if lp_detector is not None:
                    detections.extend(get_detections(lp_detector, image_tensor,
                                                     lp_model_score_threshold, nms_iou_threshold))

                # Process immediately and free memory
                processed_frame = visualize(bgr_frame, detections, scale_factor_detections)
                out.write(processed_frame)
                prev_processed_frame = processed_frame

                # Aggressive memory cleanup
                del image_tensor
                del detections
                torch.cuda.synchronize()  # Ensure GPU operations complete

            if frame_counter % 20 == 0:  # Reduced frequency of cache clearing
                torch.cuda.empty_cache()

    video_reader_clip.close()
    out.release()


if __name__ == "__main__":
    args = validate_inputs(parse_args())

    torch.cuda.set_per_process_memory_fraction(0.8)
    torch.backends.cudnn.benchmark = True

    if args.face_model_path is not None:
        face_detector = torch.jit.load(args.face_model_path, map_location="cpu").to(
            get_device()
        )
        face_detector.eval()
    else:
        face_detector = None

    if args.lp_model_path is not None:
        lp_detector = torch.jit.load(args.lp_model_path, map_location="cpu").to(
            get_device()
        )
        lp_detector.eval()
    else:
        lp_detector = None

    if args.input_image_path is not None:
        image = visualize_image(
            args.input_image_path,
            face_detector,
            lp_detector,
            args.face_model_score_threshold,
            args.lp_model_score_threshold,
            args.nms_iou_threshold,
            args.output_image_path,
            args.scale_factor_detections,
        )

    if args.input_video_path is not None:
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
