"""Stage-specific local datasets with grouped, reproducible validation splits."""
import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.text_embedder import TextEmbedder
from utils.data_processor import validate_texts
from utils.utils import cal_resize_and_padding, pad_image_to_shape

ROOT = Path(__file__).resolve().parents[1]


def description_four(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('empty description')
    value = value.replace('\\r\\n', '\n').replace('\\n', '\n').replace('\r\n', '\n')
    match = re.search(r'Description\s*4\s*[:：]\s*(.*?)(?=\n\s*Description\s*\d+\s*[:：]|\Z)', value, re.I | re.S)
    if match:
        result = match.group(1).strip()
    else:
        parts = [p.strip() for p in re.split(r'\n\s*\n', value) if p.strip()]
        if len(parts) != 4:
            raise ValueError(f'expected four description paragraphs, got {len(parts)}')
        result = parts[3]
    if not result:
        raise ValueError('empty Description 4')
    return result


def validation_group(filename, seed, fraction):
    # All font variants sharing a source filename stay in the same partition.
    digest = hashlib.sha256(f'{seed}:{Path(filename).name}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64 < fraction


def load_samples(root, stage):
    root = Path(root)
    samples, report = [], Counter()
    if stage == 1:
        folders = sorted(root.glob('dataset1_manu_made_*'))
        if not folders:
            raise FileNotFoundError(f'No inscription datasets in {root}')
        for folder in folders:
            with (folder / 'training_data.json').open(encoding='utf-8') as handle:
                rows = json.load(handle)
            for row in rows:
                report['rows'] += 1
                path = folder / 'images' / row['url']
                if not path.is_file():
                    report['missing_images'] += 1
                    continue
                if not row.get('texts'):
                    report['empty_inscriptions'] += 1
                    continue
                if not isinstance(row.get('caption'), str) or not row['caption'].strip():
                    report['invalid_captions'] += 1
                    continue
                texts = []
                for text in row['texts']:
                    pos = text.get('pos', [])
                    if (len(pos) != 4 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in pos)
                            or pos[0] >= pos[2] or pos[1] >= pos[3]):
                        report['invalid_boxes'] += 1
                        continue
                    texts.append(text)
                if not texts:
                    report['empty_inscriptions'] += 1
                    continue
                samples.append(dict(url=row['url'], path=str(path), caption=row['caption'], texts=texts))
    elif stage == 2:
        from openpyxl import load_workbook
        folder = root / 'dataset2_only_landscape'
        workbook = load_workbook(folder / 'data1_9000data_nowords_COMBINED.xlsx', read_only=True, data_only=True)
        try:
            rows = workbook.active.iter_rows(values_only=True)
            header = list(next(rows))
            name_index, caption_index = header.index('image_name'), header.index('文字描述')
            seen = set()
            for row in rows:
                report['rows'] += 1
                name = row[name_index]
                if not isinstance(name, str) or not name.strip():
                    report['invalid_names'] += 1
                    continue
                name = name.strip()
                if name in seen:
                    raise ValueError(f'Duplicate image_name in workbook: {name}')
                seen.add(name)
                path = folder / 'images' / name
                if not path.is_file():
                    report['missing_images'] += 1
                    continue
                try:
                    caption = description_four(row[caption_index])
                except ValueError:
                    report['invalid_descriptions'] += 1
                    continue
                samples.append(dict(url=name, path=str(path), caption=caption, texts=[]))
        finally:
            workbook.close()
    else:
        raise ValueError('training_stage must be 1 or 2')
    report['accepted'] = len(samples)
    if not samples:
        raise ValueError(f'No usable stage {stage} samples: {dict(report)}')
    return samples, dict(report)


class LandscapeDataset(Dataset):
    def __init__(self, args, split='train'):
        if split not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        self.stage = getattr(args, 'training_stage', 1)
        self.input_size = (args.resolution_h, args.resolution_w)
        if any(v <= 0 or v % 16 for v in self.input_size):
            raise ValueError('resolution must be positive multiples of 16')
        self.max_texts = getattr(args, 'max_num_texts', 7)
        self.char_length = getattr(args, 'char_padding_to_len', 16)
        if (self.max_texts, self.char_length) != (7, 16):
            raise ValueError('Current pretrained text encoder requires 7 boxes × 16 characters')
        fraction = getattr(args, 'validation_fraction', 0.05)
        if not 0 < fraction < 1:
            raise ValueError('validation_fraction must be between 0 and 1')
        seed = getattr(args, 'dataset_split_seed', 42)
        all_samples, self.report = load_samples(getattr(args, 'dataset_root', ROOT / 'dataset'), self.stage)
        self.samples = [s for s in all_samples if validation_group(s['url'], seed, fraction) == (split == 'validation')]
        if not self.samples:
            raise ValueError(f'Empty {split} partition for stage {self.stage}')
        self.report.update(split=split, selected=len(self.samples))
        self.embedder = TextEmbedder(str(ROOT / 'assets/char2feat_ppocr_neck64_avg.pth'))
        self.feature_drop = (getattr(args, 'text_feature_drop', 0) or 0) if split == 'train' else 0
        print(f'Stage {self.stage} dataset: {self.report}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = cv2.imread(sample['path'], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Unreadable image: {sample["path"]}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        texts = validate_texts(copy.deepcopy(sample['texts']), w, h)
        new_h, new_w, scale = cal_resize_and_padding((h, w), self.input_size)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        image = pad_image_to_shape(image, self.input_size, pad_value=255)
        scaled = [{'content': t['content'], 'pos': [v * scale for v in t['pos']]} for t in texts]
        # TextEmbedder's position coordinate system matches inference: 1024 × 1024.
        dimensions = (self.input_size[1], self.input_size[0]) * 2
        encoded = [{'content': t['content'],
                    'pos': [v * 1024 / dimensions[i] for i, v in enumerate(t['pos'])]}
                   for t in scaled]
        embeds = self.embedder(encoded)
        if texts and self.feature_drop:
            for i in range(len(texts)):
                if torch.rand(()).item() < self.feature_drop ** (1 / len(texts)):
                    embeds[i * 16:(i + 1) * 16, :64] = 0
        padded_texts = scaled + [{'content': '', 'pos': [0., 0., 0., 0.]} for _ in range(7 - len(texts))]
        return dict(url=sample['url'], texts=padded_texts,
                    gt_im=torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 127.5 - 1,
                    caption=sample['caption'], num_texts=len(texts), text_embeds=embeds,
                    target_size=(new_h, new_w), original_size=(h, w))
