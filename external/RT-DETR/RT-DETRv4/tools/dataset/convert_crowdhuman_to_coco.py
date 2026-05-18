import argparse
import json
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Convert CrowdHuman ODGT annotations to COCO format.')
    parser.add_argument('--input-odgt', required=True, help='Path to CrowdHuman .odgt file')
    parser.add_argument('--images-dir', required=True, help='Directory containing CrowdHuman images')
    parser.add_argument('--output-json', required=True, help='Output COCO JSON path')
    parser.add_argument(
        '--box-field',
        default='fbox',
        choices=['fbox', 'vbox', 'hbox'],
        help='CrowdHuman box field to export as COCO bbox',
    )
    parser.add_argument(
        '--category-id',
        type=int,
        default=0,
        help='COCO category id for person annotations',
    )
    parser.add_argument(
        '--keep-ignore',
        action='store_true',
        help='Keep ignored regions as iscrowd=1 annotations for evaluation',
    )
    return parser.parse_args()


def load_odgt(path: Path):
    records = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clip_box(box, width, height):
    x, y, w, h = box
    x1 = max(0.0, float(x))
    y1 = max(0.0, float(y))
    x2 = min(float(width), float(x) + float(w))
    y2 = min(float(height), float(y) + float(h))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None
    return [x1, y1, clipped_w, clipped_h]


def is_ignored(gtbox):
    extra = gtbox.get('extra', {})
    head_attr = gtbox.get('head_attr', {})
    if extra.get('ignore', 0):
        return True
    if head_attr.get('ignore', 0):
        return True
    if gtbox.get('tag') != 'person':
        return True
    return False


def convert(records, images_dir: Path, box_field: str, category_id: int, keep_ignore: bool):
    images = []
    annotations = []
    missing_images = []
    ann_id = 1

    for image_id, record in enumerate(records, start=1):
        file_name = f"{record['ID']}.jpg"
        image_path = images_dir / file_name
        if not image_path.exists():
            missing_images.append(file_name)
            continue

        with Image.open(image_path) as image:
            width, height = image.size

        images.append(
            {
                'id': image_id,
                'file_name': file_name,
                'width': width,
                'height': height,
            }
        )

        for gtbox in record.get('gtboxes', []):
            if box_field not in gtbox:
                continue

            ignore = is_ignored(gtbox)
            if ignore and not keep_ignore:
                continue

            bbox = clip_box(gtbox[box_field], width, height)
            if bbox is None:
                continue

            annotations.append(
                {
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': bbox,
                    'area': bbox[2] * bbox[3],
                    'iscrowd': 1 if ignore else 0,
                }
            )
            ann_id += 1

    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': category_id, 'name': 'person'}],
    }
    return coco, missing_images


def main():
    args = parse_args()
    input_odgt = Path(args.input_odgt)
    images_dir = Path(args.images_dir)
    output_json = Path(args.output_json)

    records = load_odgt(input_odgt)
    coco, missing_images = convert(
        records=records,
        images_dir=images_dir,
        box_field=args.box_field,
        category_id=args.category_id,
        keep_ignore=args.keep_ignore,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open('w', encoding='utf-8') as handle:
        json.dump(coco, handle)

    print(
        json.dumps(
            {
                'images': len(coco['images']),
                'annotations': len(coco['annotations']),
                'missing_images': len(missing_images),
                'box_field': args.box_field,
                'keep_ignore': args.keep_ignore,
                'output_json': str(output_json),
            },
            indent=2,
        )
    )

    if missing_images:
        print('Missing image examples:')
        for file_name in missing_images[:20]:
            print(file_name)


if __name__ == '__main__':
    main()
