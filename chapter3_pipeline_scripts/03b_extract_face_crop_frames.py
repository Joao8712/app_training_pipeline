#!/usr/bin/env python3
"""
Extract deterministic face-crop frames from videos and create an image manifest.

This script is intended as a face-crop ablation companion to the full-frame
frame-extraction pipeline. It uses the same segment-level label manifest and the
same deterministic frame sampling strategy, but it detects a face on the original
video frame before resizing the output image.

Input:
  --video_root: directory containing extracted .mp4 files, recursively; or
  --archives_root: directory containing ZIP archives with nested videos.
  --label_manifest: labels_manifest.csv or final_split_manifest.csv.
  --out_dir: output directory for cropped images and image_manifest.csv.

Main design choices:
  - The sampled frame indices are deterministic.
  - Face detection is applied to the original decoded frame before resizing.
  - If no face is detected, the full frame is kept as a fallback and
    face_crop_status is set to "no_face_fallback_full_frame".
  - All columns from the input label manifest are preserved.
  - Additional face-crop metadata is written for traceability.
  - The produced image_manifest.csv keeps the same core fields used by the
    existing training pipeline: image_path, final_split, video_name, labels,
    frame_index, extraction_status, and related metadata.

Notes:
  - The default detector uses OpenCV's Haar cascade because it is available in
    standard OpenCV installations. Use --detector mediapipe if mediapipe is
    installed and preferred.
  - This script does not recalculate labels. Each extracted image inherits the
    official segment-level label of its source video segment.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import pandas as pd
from PIL import Image
from tqdm import tqdm


TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


@dataclass
class FaceDetection:
    """Pixel-space face detection result."""

    x: int
    y: int
    w: int
    h: int
    confidence: Optional[float]
    detector: str

    @property
    def area(self) -> int:
        return int(self.w * self.h)


def find_video_files(video_root: Path) -> dict[str, Path]:
    """
    Find .mp4 files recursively.

    The label files refer to videos by filename. Duplicate basenames are
    ambiguous and are treated as an error rather than silently overwritten.
    """
    files: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in video_root.rglob("*.mp4"):
        if path.name in files:
            duplicates.setdefault(path.name, [files[path.name]]).append(path)
        else:
            files[path.name] = path

    if duplicates:
        duplicate_msg = "\n".join(
            f"{name}: " + ", ".join(str(p) for p in paths)
            for name, paths in duplicates.items()
        )
        raise ValueError(
            "Duplicate video filenames were found. Because the label manifest "
            "uses video basenames, these duplicates are ambiguous:\n"
            f"{duplicate_msg}"
        )

    return files


def safe_extract_zip(zip_path: Path, dest_dir: Path, password: bytes | None = None) -> None:
    """Extract a ZIP file into dest_dir."""
    with zipfile.ZipFile(zip_path, "r") as z:
        if password is not None:
            z.extractall(path=str(dest_dir), pwd=password)
        else:
            z.extractall(path=str(dest_dir))


def recursively_extract_inner_zips(
    base_dir: Path,
    max_depth: int = 5,
    password: bytes | None = None,
) -> None:
    """
    Recursively extract ZIP files under base_dir up to max_depth levels.

    Extracted content is placed in a sibling directory named after the zip file.
    The original zip files are not deleted.
    """
    for _depth in range(max_depth):
        zips = list(base_dir.rglob("*.zip"))
        if not zips:
            break

        extracted_any = False
        for zp in zips:
            dest = zp.with_suffix("")
            if dest.exists():
                continue
            try:
                dest.mkdir(parents=True, exist_ok=True)
                safe_extract_zip(zp, dest, password=password)
                extracted_any = True
            except Exception as exc:
                print(f"Warning: failed extracting inner zip {zp}: {exc}")
                continue

        if not extracted_any:
            break


def sample_indices(n_frames: int, n_samples: int, strategy: str = "uniform") -> list[int]:
    """
    Return deterministic frame indices.

    uniform:
      Selects indices distributed across the decoded frame sequence and avoids
      the exact first and last frames when possible.

    center:
      Selects a small window around the center frame.
    """
    if n_frames <= 0 or n_samples <= 0:
        return []

    if n_samples == 1:
        return [n_frames // 2]

    if strategy == "uniform":
        indices = [
            int(round((i + 1) * (n_frames - 1) / (n_samples + 1)))
            for i in range(n_samples)
        ]
    elif strategy == "center":
        center = n_frames // 2
        offsets = list(range(-(n_samples // 2), n_samples - n_samples // 2))
        indices = [center + offset for offset in offsets]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    clipped = [min(max(idx, 0), n_frames - 1) for idx in indices]
    return sorted(set(clipped))


def read_frame_bgr(cap: cv2.VideoCapture, frame_index: int) -> Any | None:
    """Read one frame from an OpenCV VideoCapture object."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    return frame_bgr if ok else None


class FaceDetector:
    """Face detector wrapper supporting OpenCV Haar and MediaPipe."""

    def __init__(self, args: argparse.Namespace):
        self.detector_name = args.detector
        self.args = args
        self._haar = None
        self._mp_face_detection = None
        self._mp_detector = None

        if self.detector_name == "opencv_haar":
            cascade_path = Path(args.haar_cascade_path) if args.haar_cascade_path else None
            if cascade_path is None:
                cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if not cascade_path.exists():
                raise FileNotFoundError(f"OpenCV Haar cascade file not found: {cascade_path}")
            self._haar = cv2.CascadeClassifier(str(cascade_path))
            if self._haar.empty():
                raise RuntimeError(f"Failed to load OpenCV Haar cascade: {cascade_path}")

        elif self.detector_name == "mediapipe":
            try:
                import mediapipe as mp  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "MediaPipe is not installed. Install it with `pip install mediapipe` "
                    "or use `--detector opencv_haar`."
                ) from exc
            self._mp_face_detection = mp.solutions.face_detection
            self._mp_detector = self._mp_face_detection.FaceDetection(
                model_selection=args.mediapipe_model_selection,
                min_detection_confidence=args.mediapipe_min_detection_confidence,
            )
        else:
            raise ValueError(f"Unsupported detector: {self.detector_name}")

    def close(self) -> None:
        if self._mp_detector is not None:
            self._mp_detector.close()

    def detect_largest_face(self, frame_bgr: Any) -> FaceDetection | None:
        """Detect faces and return the largest valid detection in pixel coordinates."""
        if self.detector_name == "opencv_haar":
            return self._detect_largest_face_haar(frame_bgr)
        if self.detector_name == "mediapipe":
            return self._detect_largest_face_mediapipe(frame_bgr)
        return None

    def _detect_largest_face_haar(self, frame_bgr: Any) -> FaceDetection | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=self.args.haar_scale_factor,
            minNeighbors=self.args.haar_min_neighbors,
            minSize=(self.args.min_face_size, self.args.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda box: int(box[2] * box[3]))
        return FaceDetection(
            x=int(x),
            y=int(y),
            w=int(w),
            h=int(h),
            confidence=None,
            detector="opencv_haar",
        )

    def _detect_largest_face_mediapipe(self, frame_bgr: Any) -> FaceDetection | None:
        if self._mp_detector is None:
            return None
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mp_detector.process(frame_rgb)
        detections = result.detections or []
        if not detections:
            return None

        parsed: list[FaceDetection] = []
        for det in detections:
            rel = det.location_data.relative_bounding_box
            x = int(round(rel.xmin * width))
            y = int(round(rel.ymin * height))
            w = int(round(rel.width * width))
            h = int(round(rel.height * height))
            x, y, w, h = clip_box(x, y, w, h, width, height)
            if w <= 0 or h <= 0:
                continue
            confidence = float(det.score[0]) if det.score else None
            parsed.append(
                FaceDetection(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    confidence=confidence,
                    detector="mediapipe",
                )
            )

        if not parsed:
            return None
        return max(parsed, key=lambda face: face.area)


def clip_box(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    """Clip a bounding box to image boundaries."""
    x1 = max(0, min(int(x), image_width - 1))
    y1 = max(0, min(int(y), image_height - 1))
    x2 = max(0, min(int(x + w), image_width))
    y2 = max(0, min(int(y + h), image_height))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def make_margin_crop_box(
    face: FaceDetection,
    image_width: int,
    image_height: int,
    crop_margin: float,
    square_crop: bool,
) -> tuple[int, int, int, int]:
    """
    Create a crop box around a detected face.

    If square_crop is True, the crop side is based on the larger face-box
    dimension plus margin. The box is clipped to image boundaries. If clipping
    near an image boundary prevents a perfectly square crop, the remaining box is
    still used and its dimensions are recorded in the manifest.
    """
    cx = face.x + face.w / 2.0
    cy = face.y + face.h / 2.0

    if square_crop:
        side = max(face.w, face.h) * (1.0 + 2.0 * crop_margin)
        x1 = int(round(cx - side / 2.0))
        y1 = int(round(cy - side / 2.0))
        x2 = int(round(cx + side / 2.0))
        y2 = int(round(cy + side / 2.0))
    else:
        x1 = int(round(face.x - crop_margin * face.w))
        y1 = int(round(face.y - crop_margin * face.h))
        x2 = int(round(face.x + face.w + crop_margin * face.w))
        y2 = int(round(face.y + face.h + crop_margin * face.h))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))
    return x1, y1, x2 - x1, y2 - y1


def save_bgr_as_rgb_image(
    image_bgr: Any,
    out_path: Path,
    image_size: int,
    image_ext: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    """Resize a BGR image if requested, convert to RGB, and save with Pillow."""
    pre_resize_height, pre_resize_width = image_bgr.shape[:2]

    if image_size and image_size > 0:
        image_bgr = cv2.resize(
            image_bgr,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

    output_height, output_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    if image_ext == "jpg":
        image.save(
            out_path,
            format="JPEG",
            quality=jpeg_quality,
            subsampling=0,
            optimize=True,
        )
    elif image_ext == "png":
        image.save(out_path, format="PNG")
    else:
        raise ValueError(f"Unsupported image extension: {image_ext}")

    return {
        "crop_pre_resize_width": pre_resize_width,
        "crop_pre_resize_height": pre_resize_height,
        "output_width": output_width,
        "output_height": output_height,
        "color_mode": "RGB",
    }


def base_manifest_fields(
    row: pd.Series,
    video_path: Path | None,
    n_frames: int | None,
    fps: float | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Create manifest fields shared by all rows for a video segment."""
    return {
        **row.to_dict(),
        "video_path": str(video_path) if video_path is not None else None,
        "video_frame_count": n_frames,
        "video_fps": fps,
        "sampling_strategy": args.sampling_strategy,
        "target_frames_per_segment": args.n_frames_per_segment,
        "output_image_size": args.image_size,
        "image_ext": args.image_ext,
        "jpeg_quality": args.jpeg_quality if args.image_ext == "jpg" else None,
        "face_detector": args.detector,
        "crop_margin": args.crop_margin,
        "square_crop": args.square_crop,
        "fallback_policy": "full_frame_if_no_face" if args.fallback_full_frame else "mark_failure_if_no_face",
    }


def failure_face_fields() -> dict[str, Any]:
    """Default face-crop fields for extraction failures."""
    return {
        "face_crop_status": None,
        "face_detected": False,
        "face_box_x": None,
        "face_box_y": None,
        "face_box_w": None,
        "face_box_h": None,
        "face_box_area": None,
        "face_confidence": None,
        "crop_source": None,
        "crop_box_x": None,
        "crop_box_y": None,
        "crop_box_w": None,
        "crop_box_h": None,
        "crop_pre_resize_width": None,
        "crop_pre_resize_height": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deterministic face-crop RGB frames from videos."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_root", type=Path, help="Directory with extracted .mp4 files (recursive search)")
    group.add_argument("--archives_root", type=Path, help="Directory containing ZIP archives; nested zips are extracted temporarily")
    parser.add_argument("--label_manifest", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--n_frames_per_segment", type=int, default=5)
    parser.add_argument(
        "--sampling_strategy",
        choices=["uniform", "center"],
        default="uniform",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Resize output to image_size x image_size. Use 0 to keep crop/full-frame size.",
    )
    parser.add_argument("--image_ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--zip_password", type=str, default=None, help="Optional password for encrypted ZIPs")
    parser.add_argument("--max_zip_depth", type=int, default=5, help="Maximum nested zip extraction depth")
    parser.add_argument("--cleanup_extracted", action="store_true", help="Delete temporary extracted folders after processing")

    parser.add_argument(
        "--detector",
        choices=["opencv_haar", "mediapipe"],
        default="opencv_haar",
        help="Face detector used before resizing the output image.",
    )
    parser.add_argument("--haar_cascade_path", type=str, default=None, help="Optional path to a Haar cascade XML file")
    parser.add_argument("--haar_scale_factor", type=float, default=1.1)
    parser.add_argument("--haar_min_neighbors", type=int, default=5)
    parser.add_argument("--min_face_size", type=int, default=24)
    parser.add_argument("--mediapipe_model_selection", type=int, default=1, choices=[0, 1])
    parser.add_argument("--mediapipe_min_detection_confidence", type=float, default=0.5)
    parser.add_argument("--crop_margin", type=float, default=0.35, help="Margin around detected face box, expressed as a fraction of face size")
    parser.add_argument("--square_crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback_full_frame", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(exist_ok=True)

    labels = pd.read_csv(args.label_manifest)
    if "video_name" not in labels.columns:
        raise ValueError("The label manifest must contain a 'video_name' column.")

    rows: list[dict[str, Any]] = []
    processed_videos: set[str] = set()
    zip_password_bytes = args.zip_password.encode("utf-8") if args.zip_password else None
    detector = FaceDetector(args)

    def process_label_rows_for_videos(label_subset: pd.DataFrame, video_files_map: dict[str, Path]) -> None:
        if len(label_subset) == 0:
            return
        for _, row in tqdm(label_subset.iterrows(), total=len(label_subset), desc="Processing videos", unit="video"):
            video_name = row["video_name"]
            video_path = video_files_map.get(video_name)

            if video_path is None:
                base = base_manifest_fields(row, None, None, None, args)
                rows.append(
                    {
                        **base,
                        "frame_index": None,
                        "timestamp_sec": None,
                        "image_path": None,
                        "original_frame_width": None,
                        "original_frame_height": None,
                        "output_width": None,
                        "output_height": None,
                        "color_mode": None,
                        **failure_face_fields(),
                        "extraction_status": "video_missing",
                        "error_message": None,
                    }
                )
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                base = base_manifest_fields(row, video_path, None, None, args)
                rows.append(
                    {
                        **base,
                        "frame_index": None,
                        "timestamp_sec": None,
                        "image_path": None,
                        "original_frame_width": None,
                        "original_frame_height": None,
                        "output_width": None,
                        "output_height": None,
                        "color_mode": None,
                        **failure_face_fields(),
                        "extraction_status": "video_open_failed",
                        "error_message": None,
                    }
                )
                continue

            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
            indices = sample_indices(
                n_frames=n_frames,
                n_samples=args.n_frames_per_segment,
                strategy=args.sampling_strategy,
            )
            base = base_manifest_fields(row, video_path, n_frames, fps, args)

            if not indices:
                rows.append(
                    {
                        **base,
                        "frame_index": None,
                        "timestamp_sec": None,
                        "image_path": None,
                        "original_frame_width": None,
                        "original_frame_height": None,
                        "output_width": None,
                        "output_height": None,
                        "color_mode": None,
                        **failure_face_fields(),
                        "extraction_status": "no_frames",
                        "error_message": None,
                    }
                )
                cap.release()
                continue

            stem = Path(video_name).stem

            for frame_index in indices:
                timestamp_sec = frame_index / fps if fps > 0 else None
                frame_bgr = read_frame_bgr(cap, frame_index)

                if frame_bgr is None:
                    rows.append(
                        {
                            **base,
                            "frame_index": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "image_path": None,
                            "original_frame_width": None,
                            "original_frame_height": None,
                            "output_width": None,
                            "output_height": None,
                            "color_mode": None,
                            **failure_face_fields(),
                            "extraction_status": "frame_read_failed",
                            "error_message": None,
                        }
                    )
                    continue

                original_height, original_width = frame_bgr.shape[:2]
                detection = detector.detect_largest_face(frame_bgr)

                face_fields: dict[str, Any]
                if detection is not None:
                    crop_x, crop_y, crop_w, crop_h = make_margin_crop_box(
                        detection,
                        image_width=original_width,
                        image_height=original_height,
                        crop_margin=args.crop_margin,
                        square_crop=args.square_crop,
                    )
                    crop_bgr = frame_bgr[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    face_fields = {
                        "face_crop_status": "face_detected",
                        "face_detected": True,
                        "face_box_x": detection.x,
                        "face_box_y": detection.y,
                        "face_box_w": detection.w,
                        "face_box_h": detection.h,
                        "face_box_area": detection.area,
                        "face_confidence": detection.confidence,
                        "crop_source": "detected_face_margin_crop",
                        "crop_box_x": crop_x,
                        "crop_box_y": crop_y,
                        "crop_box_w": crop_w,
                        "crop_box_h": crop_h,
                    }
                else:
                    if not args.fallback_full_frame:
                        rows.append(
                            {
                                **base,
                                "frame_index": frame_index,
                                "timestamp_sec": timestamp_sec,
                                "image_path": None,
                                "original_frame_width": original_width,
                                "original_frame_height": original_height,
                                "output_width": None,
                                "output_height": None,
                                "color_mode": None,
                                **{
                                    **failure_face_fields(),
                                    "face_crop_status": "no_face_detected",
                                    "crop_source": None,
                                },
                                "extraction_status": "face_missing",
                                "error_message": None,
                            }
                        )
                        continue

                    crop_x, crop_y, crop_w, crop_h = 0, 0, original_width, original_height
                    crop_bgr = frame_bgr
                    face_fields = {
                        "face_crop_status": "no_face_fallback_full_frame",
                        "face_detected": False,
                        "face_box_x": None,
                        "face_box_y": None,
                        "face_box_w": None,
                        "face_box_h": None,
                        "face_box_area": None,
                        "face_confidence": None,
                        "crop_source": "full_frame_fallback",
                        "crop_box_x": crop_x,
                        "crop_box_y": crop_y,
                        "crop_box_w": crop_w,
                        "crop_box_h": crop_h,
                    }

                out_name = f"{stem}_f{frame_index:06d}.{args.image_ext}"
                out_path = image_dir / out_name

                try:
                    image_metadata = save_bgr_as_rgb_image(
                        image_bgr=crop_bgr,
                        out_path=out_path,
                        image_size=args.image_size,
                        image_ext=args.image_ext,
                        jpeg_quality=args.jpeg_quality,
                    )
                    rows.append(
                        {
                            **base,
                            "frame_index": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "image_path": str(out_path.relative_to(args.out_dir)),
                            "original_frame_width": original_width,
                            "original_frame_height": original_height,
                            **face_fields,
                            **image_metadata,
                            "extraction_status": "ok",
                            "error_message": None,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            **base,
                            "frame_index": frame_index,
                            "timestamp_sec": timestamp_sec,
                            "image_path": None,
                            "original_frame_width": original_width,
                            "original_frame_height": original_height,
                            "output_width": None,
                            "output_height": None,
                            "color_mode": None,
                            **face_fields,
                            "crop_pre_resize_width": None,
                            "crop_pre_resize_height": None,
                            "extraction_status": "image_write_failed",
                            "error_message": str(exc),
                        }
                    )

            cap.release()

    try:
        if args.video_root is not None:
            video_files = find_video_files(args.video_root)
            process_label_rows_for_videos(labels, video_files)

        if args.archives_root is not None:
            archives = sorted(args.archives_root.rglob("*.zip"))
            tmp_base = args.out_dir / "tmp_extracted"
            tmp_base.mkdir(parents=True, exist_ok=True)
            if not archives:
                print(f"No ZIP archives found under {args.archives_root}")

            for archive in tqdm(archives, desc="Processing archives", unit="archive"):
                tmp_dir = Path(tempfile.mkdtemp(prefix=f"{archive.stem}_", dir=str(tmp_base)))
                try:
                    safe_extract_zip(archive, tmp_dir, password=zip_password_bytes)
                    recursively_extract_inner_zips(
                        tmp_dir,
                        max_depth=args.max_zip_depth,
                        password=zip_password_bytes,
                    )
                    video_files = find_video_files(tmp_dir)
                    # Avoid processing duplicate video basenames across archives.
                    available_new = {k: v for k, v in video_files.items() if k not in processed_videos}
                    subset = labels[labels["video_name"].isin(available_new.keys())]
                    if not subset.empty:
                        process_label_rows_for_videos(subset, available_new)
                        processed_videos.update(subset["video_name"].astype(str).tolist())
                except Exception as exc:
                    print(f"Warning: failed processing archive {archive}: {exc}")
                finally:
                    if args.cleanup_extracted:
                        try:
                            shutil.rmtree(tmp_dir)
                        except Exception as exc:
                            print(f"Warning: failed to remove temporary directory {tmp_dir}: {exc}")
    finally:
        detector.close()

    image_manifest = pd.DataFrame(rows)
    image_manifest.to_csv(args.out_dir / "image_manifest.csv", index=False)

    extraction_summary = (
        image_manifest.groupby("extraction_status", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values("extraction_status")
    )
    extraction_summary.to_csv(args.out_dir / "image_extraction_summary.csv", index=False)

    if "face_crop_status" in image_manifest.columns:
        face_summary = (
            image_manifest.groupby(["extraction_status", "face_crop_status"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["extraction_status", "face_crop_status"])
        )
        face_summary.to_csv(args.out_dir / "face_crop_summary.csv", index=False)

    print(f"Wrote image manifest to {args.out_dir / 'image_manifest.csv'}")
    print(f"Wrote extraction summary to {args.out_dir / 'image_extraction_summary.csv'}")
    print(f"Wrote face-crop summary to {args.out_dir / 'face_crop_summary.csv'}")


if __name__ == "__main__":
    main()
