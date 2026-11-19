"""Audit local stage-specific data without loading diffusion models."""
import argparse
import json
from pathlib import Path
from PIL import Image
from data_utils.landscape_dataset import ROOT, load_samples, validation_group
from utils.data_processor import validate_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset_root', default=str(ROOT / 'dataset'))
    parser.add_argument('--check_images', action='store_true', help='Check every image header and inscription box')
    args = parser.parse_args()
    failed = False
    for stage in (1, 2):
        samples, report = load_samples(args.dataset_root, stage)
        report['validation'] = sum(validation_group(s['url'], 42, 0.05) for s in samples)
        report['train'] = len(samples) - report['validation']
        if args.check_images:
            errors = []
            for sample in samples:
                try:
                    with Image.open(sample['path']) as image:
                        validate_texts(sample['texts'], *image.size)
                        image.verify()
                except Exception as error:
                    errors.append({'path': sample['path'], 'error': str(error)})
            report['image_errors'] = len(errors)
            report['error_examples'] = errors[:5]
            failed |= bool(errors)
        print(json.dumps({'stage': stage, **report}, ensure_ascii=False, indent=2))
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
