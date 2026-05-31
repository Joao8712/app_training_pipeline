#!/usr/bin/env python3
"""
Extract deterministic frames from videos and create an image manifest.

Input:
  --video_root: directory containing extracted .mp4 files, recursively.
  --label_manifest: labels_manifest.csv or final_split_manifest.csv.
  --out_dir: directory for extracted images and image_manifest.csv.

This script does not recalculate labels. Each extracted image inherits the
official segment-level label of its source video segment.

Main design choices:
  - Deterministic frame-index sampling.
  - RGB output images saved through Pillow.
  - Explicit manifest metadata for reproducibility.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

import cv2
import pandas as pd
from PIL import Image


TRAITS = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
]


def find_video_files(video_root: Path) -> dict[str, Path]:
    """
    Find .mp4 files recursively.

    The label files refer to videos by filename. Therefore, duplicate basenames
    are ambiguous and should be treated as an error rather than silently
    overwritten.
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
    """Extract a ZIP file into dest_dir. If the zip is encrypted, pass a password as bytes."""
    with zipfile.ZipFile(zip_path, "r") as z:
        if password is not None:
            try:
                z.extractall(path=str(dest_dir), pwd=password)
            except RuntimeError as exc:
                # Some zip files raise a RuntimeError if the password is wrong
                raise
        else:
            z.extractall(path=str(dest_dir))


def recursively_extract_inner_zips(base_dir: Path, max_depth: int = 5, password: bytes | None = None) -> None:
    """Recursively extract any .zip files found under base_dir up to max_depth levels.

    Extracted content is placed in a sibling directory named after the zip (without deleting the zip).
    """
    for depth in range(max_depth):
        zips = list(base_dir.rglob("*.zip"))
        # stop if no more zips or all zips are already top-level outer archives
        if not zips:
            break
        extracted_any = False
        for zp in zips:
            # skip zips that are already inside an extraction folder we created
            dest = zp.with_suffix("")
            if dest.exists():
                continue
            try:
                dest.mkdir(parents=True, exist_ok=True)
                safe_extract_zip(zp, dest, password=password)
                extracted_any = True
            except Exception:
                # ignore problematic inner zips but continue
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

    For very short videos, duplicate indices may occur; duplicates are removed.
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


def save_rgb_image(
    frame_bgr: Any,
    out_path: Path,
    image_size: int,
    image_ext: str,
    jpeg_quality: int,
) -> dict[str, Any]:
    """
    Convert an OpenCV BGR frame to RGB and save with Pillow.

    Returns dimension metadata.
    """
    original_height, original_width = frame_bgr.shape[:2]

    if image_size and image_size > 0:
        frame_bgr = cv2.resize(
            frame_bgr,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )

    output_height, output_width = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

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
        "original_frame_width": original_width,
        "original_frame_height": original_height,
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract deterministic RGB frames from videos."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video_root", type=Path, help="Directory with extracted .mp4 files (recursive search)")
    group.add_argument("--archives_root", type=Path, help="Directory containing outer ZIP archives; script will extract nested zips, process mp4s, and remove extracted temp files")
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
        help="Resize square output to image_size x image_size. Use 0 to keep original size.",
    )
    parser.add_argument("--image_ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--zip_password", type=str, default=None, help="Optional password for encrypted ZIPs")
    parser.add_argument("--max_zip_depth", type=int, default=5, help="Maximum nested zip extraction depth")
    parser.add_argument("--cleanup_extracted", action="store_true", help="Delete temporary extracted folders after processing to save disk space")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(exist_ok=True)

    labels = pd.read_csv(args.label_manifest)

    if "video_name" not in labels.columns:
        raise ValueError("The label manifest must contain a 'video_name' column.")

    rows: list[dict[str, Any]] = []

    zip_password_bytes = args.zip_password.encode("utf-8") if args.zip_password else None

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
                            "extraction_status": "frame_read_failed",
                            "error_message": None,
                        }
                    )
                    continue

                out_name = f"{stem}_f{frame_index:06d}.{args.image_ext}"
                out_path = image_dir / out_name

                try:
                    image_metadata = save_rgb_image(
                        frame_bgr=frame_bgr,
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
                            "original_frame_width": None,
                            "original_frame_height": None,
                            "output_width": None,
                            "output_height": None,
                            "color_mode": None,
                            "extraction_status": "image_write_failed",
                            "error_message": str(exc),
                        }
                    )

            cap.release()

    # Mode: single video_root directory
    if args.video_root is not None:
        video_files = find_video_files(args.video_root)
        process_label_rows_for_videos(labels, video_files)

    # Mode: archives_root — extract each outer archive, recursively unpack inner zips,
    # process mp4 files found inside, then optionally remove the temporary extraction folder.
    if args.archives_root is not None:
        archives = sorted(args.archives_root.rglob("*.zip"))
        tmp_base = args.out_dir / "tmp_extracted"
        tmp_base.mkdir(parents=True, exist_ok=True)
        if not archives:
            print(f"No ZIP archives found under {args.archives_root}")
        for archive in tqdm(archives, desc="Processing archives", unit="archive"):
            # create a unique temp dir for this archive
            tmp_dir = Path(tempfile.mkdtemp(prefix=f"{archive.stem}_", dir=str(tmp_base)))
            try:
                safe_extract_zip(archive, tmp_dir, password=zip_password_bytes)
                recursively_extract_inner_zips(tmp_dir, max_depth=args.max_zip_depth, password=zip_password_bytes)
                # process videos found under tmp_dir
                video_files = find_video_files(tmp_dir)
                subset = labels[labels["video_name"].isin(video_files.keys())]
                if not subset.empty:
                    process_label_rows_for_videos(subset, video_files)
            except Exception as exc:
                # record failures for videos in this archive if necessary
                print(f"Warning: failed processing archive {archive}: {exc}")
            finally:
                if args.cleanup_extracted:
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass

    image_manifest = pd.DataFrame(rows)
    image_manifest.to_csv(args.out_dir / "image_manifest.csv", index=False)

    summary = (
        image_manifest.groupby("extraction_status")
        .size()
        .reset_index(name="n")
        .sort_values("extraction_status")
    )
    summary.to_csv(args.out_dir / "image_extraction_summary.csv", index=False)

    print(f"Wrote image manifest to {args.out_dir / 'image_manifest.csv'}")
    print(f"Wrote extraction summary to {args.out_dir / 'image_extraction_summary.csv'}")


if __name__ == "__main__":
    main()
