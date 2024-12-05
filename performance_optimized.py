# performance_optimized.py
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
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn


class DetectionProcessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_model = self._load_model(args.face_model_path)
        self.lp_model = self._load_model(args.lp_model_path)
        # Pre-allocate tensors for batch processing
        self.batch_size = 32
        self.processing_pool = ThreadPoolExecutor(max_workers=4)

    @torch.no_grad()  # Disable gradient computation
    def _load_model(self, model_path: Optional[str]) -> Optional[nn.Module]:
        if model_path:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            if self.device == "cuda":
                model = model.half()  # Use FP16 for faster inference
            return model
        return None

    def process_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        # Convert batch to tensor
        batch_tensor = torch.stack([
            torch.from_numpy(np.transpose(frame, (2, 0, 1))).float().to(self.device)
            for frame in frames
        ])
        if self.device == "cuda":
            batch_tensor = batch_tensor.half()

        # Process batch
        all_detections = []
        for model, threshold in [(self.face_model, self.args.face_model_score_threshold),
                                 (self.lp_model, self.args.lp_model_score_threshold)]:
            if model:
                boxes, _, scores, _ = model(batch_tensor)
                nms_indices = torchvision.ops.batched_nms(
                    boxes, scores, torch.zeros_like(scores), self.args.nms_iou_threshold)
                filtered_boxes = boxes[nms_indices][scores[nms_indices] > threshold]
                all_detections.extend(filtered_boxes.cpu().numpy().tolist())

        # Process results in parallel
        return list(self.processing_pool.map(
            lambda frame_dets: self._visualize(frame_dets[0], frame_dets[1]),
            zip(frames, [all_detections] * len(frames))
        ))

    def _visualize(self, image: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        result = image.copy()
        for box in detections:
            x1, y1, x2, y2 = map(int, self._scale_box(box, image.shape[1], image.shape[0]))
            # Use faster blur method
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                roi = cv2.blur(roi, (31, 31))
                result[y1:y2, x1:x2] = roi
        return result

    def process_video(self):
        clip = VideoFileClip(self.args.input_video_path)
        frames = []
        batch = []

        for frame in clip.iter_frames(fps=self.args.output_video_fps):
            batch.append(frame)
            if len(batch) == self.batch_size:
                frames.extend(self.process_batch(batch))
                batch = []

        if batch:
            frames.extend(self.process_batch(batch))

        ImageSequenceClip(frames, fps=self.args.output_video_fps).write_videofile(
            self.args.output_video_path,
            codec='libx264',
            threads=4
        )

    @staticmethod
    def _scale_box(box: List[float], width: int, height: int) -> List[float]:
        x1, y1, x2, y2 = box
        return [
            max(0, x1),
            max(0, y1),
            min(width, x2),
            min(height, y2)
        ]