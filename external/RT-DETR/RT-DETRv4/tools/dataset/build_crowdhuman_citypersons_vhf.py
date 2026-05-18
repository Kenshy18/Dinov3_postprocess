#!/usr/bin/env python3
"""Build a 3-class VisibleBody/Head/Face dataset from CrowdHuman + CityPersons.

The output is COCO detection format with contiguous class ids:
  0: VisibleBody
  1: Head
  2: Face

Images are not duplicated for CrowdHuman. The builder creates symlinks to the
existing CrowdHuman image folders and extracts only the CityPersons images that
are referenced by the new BHF annotations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RTDETR_ROOT = REPO_ROOT / "RT-DETRv4"

DEFAULT_CROWDHUMAN_ZIP = REPO_ROOT / "Data/CrowdHuman-20260517T012519Z-3-001.zip"
DEFAULT_CITYPERSONS_ZIP = REPO_ROOT / "Data/CityPersons-20260517T012605Z-3-001.zip"
DEFAULT_CITYSCAPES_ZIP = REPO_ROOT / "Data/leftImg8bit_trainvaltest.zip"
DEFAULT_CROWDHUMAN_IMAGES = RTDETR_ROOT / "data/CrowdHuman/images"
DEFAULT_OUTPUT_DIR = RTDETR_ROOT / "data/CrowdHumanCityPersons_VHF"

CATEGORIES = [
    {"id": 0, "name": "VisibleBody", "supercategory": "person"},
    {"id": 1, "name": "Head", "supercategory": "person"},
    {"id": 2, "name": "Face", "supercategory": "person"},
]

CLASS_FIELDS = [
    (0, "VisibleBody", "vbox"),
    (1, "Head", "h_bbox"),
    (2, "Face", "f_bbox"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crowdhuman-zip", type=Path, default=DEFAULT_CROWDHUMAN_ZIP)
    parser.add_argument("--citypersons-zip", type=Path, default=DEFAULT_CITYPERSONS_ZIP)
    parser.add_argument("--cityscapes-zip", type=Path, default=DEFAULT_CITYSCAPES_ZIP)
    parser.add_argument("--crowdhuman-images", type=Path, default=DEFAULT_CROWDHUMAN_IMAGES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-box-size", type=float, default=1.0)
    return parser.parse_args()


def load_json_from_zip(zip_path: Path, member_name: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            return json.load(f)


def find_zip_member(zip_path: Path, suffix: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [name for name in zf.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one member ending with {suffix!r}, found {matches}")
    return matches[0]


def ensure_clean_dirs(output_dir: Path) -> None:
    for rel in [
        "annotations",
        "images/citypersons/train",
        "images/citypersons/val",
        "metadata",
    ]:
        output_dir.joinpath(rel).mkdir(parents=True, exist_ok=True)


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        current = os.readlink(link_path)
        expected = os.path.relpath(target_path, link_path.parent)
        if current == expected:
            return
        link_path.unlink()
    elif link_path.exists():
        raise RuntimeError(f"refusing to replace non-symlink path: {link_path}")

    relative_target = os.path.relpath(target_path, link_path.parent)
    link_path.symlink_to(relative_target)


def valid_bbox(raw_bbox, width: int, height: int, min_box_size: float) -> list[float] | None:
    if not isinstance(raw_bbox, list) or len(raw_bbox) < 4:
        return None
    x, y, w, h = [float(v) for v in raw_bbox[:4]]
    if w <= min_box_size or h <= min_box_size:
        return None

    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    w2 = x2 - x1
    h2 = y2 - y1
    if w2 <= min_box_size or h2 <= min_box_size:
        return None
    return [round(x1, 3), round(y1, 3), round(w2, 3), round(h2, 3)]


def valid_image_ids(coco: dict, min_box_size: float) -> set[int]:
    images_by_id = {image["id"]: image for image in coco["images"]}
    keep: set[int] = set()
    for ann in coco["annotations"]:
        if ann.get("ignore", 0) != 0 or ann.get("iscrowd", 0) != 0:
            continue
        image = images_by_id[ann["image_id"]]
        width = int(image["width"])
        height = int(image["height"])
        for _, _, field_name in CLASS_FIELDS:
            if valid_bbox(ann.get(field_name), width, height, min_box_size) is not None:
                keep.add(ann["image_id"])
                break
    return keep


def build_cityscapes_member_index(cityscapes_zip: Path) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    with zipfile.ZipFile(cityscapes_zip) as zf:
        for name in zf.namelist():
            if not name.endswith("_leftImg8bit.png"):
                continue
            parts = name.split("/")
            if len(parts) < 4 or parts[0] != "leftImg8bit":
                continue
            split = parts[1]
            file_name = parts[-1]
            key = (split, file_name)
            if key in index:
                raise RuntimeError(f"duplicate Cityscapes file name for {key}: {name}, {index[key]}")
            index[key] = name
    return index


def extract_citypersons_images(
    cityscapes_zip: Path,
    member_index: dict[tuple[str, str], str],
    image_records_by_split: dict[str, list[dict]],
    output_dir: Path,
) -> dict[str, int]:
    extracted = Counter()
    with zipfile.ZipFile(cityscapes_zip) as zf:
        for split, images in image_records_by_split.items():
            for image in images:
                member = member_index.get((split, image["file_name"]))
                if member is None:
                    raise FileNotFoundError(f"Cityscapes image not found in zip: {split}/{image['file_name']}")
                city = member.split("/")[-2]
                destination = output_dir / "images/citypersons" / split / city / image["file_name"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and destination.stat().st_size > 0:
                    continue
                tmp = destination.with_suffix(destination.suffix + ".tmp")
                with zf.open(member) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.replace(tmp, destination)
                extracted[split] += 1
    return dict(extracted)


class CocoBuilder:
    def __init__(self, min_box_size: float):
        self.min_box_size = min_box_size
        self.next_image_id = 1
        self.next_annotation_id = 1
        self.images: list[dict] = []
        self.annotations: list[dict] = []
        self.stats = {
            "images": Counter(),
            "source_records": Counter(),
            "skipped_records": Counter(),
            "skipped_images_without_valid_boxes": Counter(),
            "boxes": defaultdict(Counter),
        }

    def add_image(self, source_dataset: str, split: str, image: dict, file_name: str) -> int:
        image_id = self.next_image_id
        self.next_image_id += 1
        self.images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "source_dataset": source_dataset,
                "source_split": split,
                "source_image_id": image["id"],
                "source_file_name": image["file_name"],
            }
        )
        self.stats["images"][source_dataset] += 1
        return image_id

    def add_annotations(
        self,
        source_dataset: str,
        image_id_map: dict[int, int],
        images_by_id: dict[int, dict],
        annotations: list[dict],
    ) -> None:
        for ann in annotations:
            self.stats["source_records"][source_dataset] += 1
            if ann["image_id"] not in image_id_map:
                continue
            if ann.get("ignore", 0) != 0 or ann.get("iscrowd", 0) != 0:
                self.stats["skipped_records"][source_dataset] += 1
                continue

            source_image = images_by_id[ann["image_id"]]
            width = int(source_image["width"])
            height = int(source_image["height"])
            target_image_id = image_id_map[ann["image_id"]]
            for category_id, class_name, field_name in CLASS_FIELDS:
                bbox = valid_bbox(ann.get(field_name), width, height, self.min_box_size)
                if bbox is None:
                    continue
                self.annotations.append(
                    {
                        "id": self.next_annotation_id,
                        "image_id": target_image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": round(bbox[2] * bbox[3], 3),
                        "iscrowd": 0,
                        "segmentation": [],
                        "source_dataset": source_dataset,
                        "source_annotation_id": ann.get("id"),
                        "source_box_field": field_name,
                    }
                )
                self.next_annotation_id += 1
                self.stats["boxes"][source_dataset][class_name] += 1

    def to_coco(self) -> dict:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": CATEGORIES,
        }

    def summary(self) -> dict:
        return {
            "images": dict(self.stats["images"]),
            "source_records": dict(self.stats["source_records"]),
            "skipped_records_ignore_or_crowd": dict(self.stats["skipped_records"]),
            "skipped_images_without_valid_boxes": dict(self.stats["skipped_images_without_valid_boxes"]),
            "boxes": {
                source: dict(counter)
                for source, counter in self.stats["boxes"].items()
            },
            "total_images": len(self.images),
            "total_annotations": len(self.annotations),
        }


def build_split(
    split: str,
    crowdhuman: dict,
    citypersons: dict,
    output_dir: Path,
    min_box_size: float,
) -> tuple[dict, dict]:
    builder = CocoBuilder(min_box_size=min_box_size)

    ch_image_map = {}
    ch_images_by_id = {image["id"]: image for image in crowdhuman["images"]}
    ch_valid_image_ids = valid_image_ids(crowdhuman, min_box_size)
    for image in crowdhuman["images"]:
        if image["id"] not in ch_valid_image_ids:
            builder.stats["skipped_images_without_valid_boxes"]["CrowdHuman"] += 1
            continue
        merged_id = builder.add_image(
            "CrowdHuman",
            split,
            image,
            f"crowdhuman_{split}/{image['file_name']}",
        )
        ch_image_map[image["id"]] = merged_id
    builder.add_annotations("CrowdHuman", ch_image_map, ch_images_by_id, crowdhuman["annotations"])

    cp_image_map = {}
    cp_images_by_id = {image["id"]: image for image in citypersons["images"]}
    cp_valid_image_ids = valid_image_ids(citypersons, min_box_size)
    for image in citypersons["images"]:
        if image["id"] not in cp_valid_image_ids:
            builder.stats["skipped_images_without_valid_boxes"]["CityPersons"] += 1
            continue
        city = image["file_name"].split("_", 1)[0]
        merged_id = builder.add_image(
            "CityPersons",
            split,
            image,
            f"citypersons/{split}/{city}/{image['file_name']}",
        )
        cp_image_map[image["id"]] = merged_id
    builder.add_annotations("CityPersons", cp_image_map, cp_images_by_id, citypersons["annotations"])

    coco = builder.to_coco()
    summary = builder.summary()
    annotation_path = output_dir / "annotations" / f"instances_{split}_visiblebody_head_face.json"
    with annotation_path.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False)
    return coco, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ensure_clean_dirs(output_dir)

    crowdhuman_train = load_json_from_zip(
        args.crowdhuman_zip,
        find_zip_member(args.crowdhuman_zip, "instances_train_full_bhf_new.json"),
    )
    crowdhuman_val = load_json_from_zip(
        args.crowdhuman_zip,
        find_zip_member(args.crowdhuman_zip, "instances_val_full_bhf_new.json"),
    )
    citypersons_train = load_json_from_zip(
        args.citypersons_zip,
        find_zip_member(args.citypersons_zip, "instances_train_bhfmatch_new.json"),
    )
    citypersons_val = load_json_from_zip(
        args.citypersons_zip,
        find_zip_member(args.citypersons_zip, "instances_val_bhfmatch_new.json"),
    )

    ensure_symlink(output_dir / "images/crowdhuman_train", args.crowdhuman_images / "train")
    ensure_symlink(output_dir / "images/crowdhuman_val", args.crowdhuman_images / "val")

    member_index = build_cityscapes_member_index(args.cityscapes_zip)
    extracted = extract_citypersons_images(
        args.cityscapes_zip,
        member_index,
        {"train": citypersons_train["images"], "val": citypersons_val["images"]},
        output_dir,
    )

    _, train_summary = build_split("train", crowdhuman_train, citypersons_train, output_dir, args.min_box_size)
    _, val_summary = build_split("val", crowdhuman_val, citypersons_val, output_dir, args.min_box_size)

    summary = {
        "name": "CrowdHumanCityPersons_VHF",
        "classes": CATEGORIES,
        "min_box_size": args.min_box_size,
        "sources": {
            "crowdhuman_zip": str(args.crowdhuman_zip.resolve()),
            "citypersons_zip": str(args.citypersons_zip.resolve()),
            "cityscapes_zip": str(args.cityscapes_zip.resolve()),
            "crowdhuman_images": str(args.crowdhuman_images.resolve()),
        },
        "output_dir": str(output_dir),
        "extracted_citypersons_images": extracted,
        "splits": {
            "train": train_summary,
            "val": val_summary,
        },
    }
    summary_path = output_dir / "metadata/summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
