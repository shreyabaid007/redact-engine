# modular_optimized.py
from dataclasses import dataclass
from typing import List, Optional, Protocol
import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class DetectionConfig:
    model_path: str
    score_threshold: float
    nms_threshold: float
    scale_factor: float


class DetectorProtocol(Protocol):
    def detect(self, image: np.ndarray) -> List[List[float]]:
        ...


class BlurProcessor(Protocol):
    def process(self, image: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        ...


class BaseDetector(ABC):
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> torch.jit.ScriptModule:
        pass

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[List[float]]:
        tensor = self._prepare_tensor(image)
        boxes, _, scores, _ = self.model(tensor)
        nms_indices = torchvision.ops.nms(
            boxes[0], scores[0], self.config.nms_threshold
        )
        filtered_boxes = boxes[0][nms_indices][scores[0][nms_indices] > self.config.score_threshold]
        return filtered_boxes.cpu().numpy().tolist()

    def _prepare_tensor(self, image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        return tensor.to(self.device).unsqueeze(0)


class FaceDetector(BaseDetector):
    def _load_model(self) -> torch.jit.ScriptModule:
        model = torch.jit.load(self.config.model_path, map_location=self.device)
        model.eval()
        return model


class LicensePlateDetector(BaseDetector):
    def _load_model(self) -> torch.jit.ScriptModule:
        model = torch.jit.load(self.config.model_path, map_location=self.device)
        model.eval()
        return model


class GaussianBlurProcessor:
    def process(self, image: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
        result = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
        return result


class VideoProcessor:
    def __init__(
            self,
            detectors: List[DetectorProtocol],
            blur_processor: BlurProcessor,
            input_path: str,
            output_path: str,
            fps: int
    ):
        self.detectors = detectors
        self.blur_processor = blur_processor
        self.input_path = input_path
        self.output_path = output_path
        self.fps = fps

    def process(self):
        with VideoFileClip(self.input_path) as clip:
            processed_frames = []
            for frame in clip.iter_frames(fps=self.fps):
                all_detections = []
                for detector in self.detectors:
                    all_detections.extend(detector.detect(frame))
                processed_frame = self.blur_processor.process(frame, all_detections)
                processed_frames.append(processed_frame)

            ImageSequenceClip(processed_frames, fps=self.fps).write_videofile(
                self.output_path,
                codec='libx264',
                threads=4
            )