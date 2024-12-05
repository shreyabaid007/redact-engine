# memory_optimized.py
import argparse
import os
from functools import lru_cache
from typing import List, Optional, Generator
import cv2
import numpy as np
import torch
import torchvision
from moviepy.editor import VideoFileClip
import gc


class MemoryEfficientProcessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_model = self._load_model(args.face_model_path)
        self.lp_model = self._load_model(args.lp_model_path)

    def _load_model(self, model_path: Optional[str]) -> Optional[torch.jit.ScriptModule]:
        if model_path:
            model = torch.jit.load(model_path, map_location=self.device)
            model.eval()
            return model
        return None

    def process_frame_generator(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """Memory efficient frame generator"""
        cap = cv2.VideoCapture(video_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process single frame
                tensor = self._prepare_tensor(frame)
                detections = self._get_detections(tensor)
                processed_frame = self._apply_blur(frame, detections)

                yield processed_frame

                # Clean up
                del tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

        finally:
            cap.release()

    def _prepare_tensor(self, frame: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.transpose(frame, (2, 0, 1))).float()
        return tensor.to(self.device).unsqueeze(0)

    def _get_detections(self, tensor: torch.Tensor) -> List[List[float]]:
        detections = []
        with torch.no_grad():
            for model, threshold in [(self.face_model, self.args.face_model_score_threshold),
                                     (self.lp_model, self.args.lp_model_score_threshold)]:
                if model:
                    boxes, _, scores, _ = model(tensor)
                    nms_indices = torchvision.ops.nms(boxes[0], scores[0], self.args.nms_iou_threshold)
                    filtered_boxes = boxes[0][nms_indices][scores[0][nms_indices] > threshold]
                    detections.extend(filtered_boxes.cpu().numpy().tolist())
        return detections

    def _apply_blur(self, frame: np.ndarray, detections: List[List[float]]) -> np.ndarray:
        for box in detections:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frame[y1:y2, x1:x2] = cv2.blur(roi, (31, 31))
        return frame

    def process_video(self):
        # Get video properties
        cap = cv2.VideoCapture(self.args.input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.args.output_video_path, fourcc, self.args.output_video_fps, (width, height))

        # Process frames
        try:
            for processed_frame in self.process_frame_generator(self.args.input_video_path):
                out.write(processed_frame)
        finally:
            out.release()