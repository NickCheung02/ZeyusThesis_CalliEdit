import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from openpyxl import Workbook
from torch.utils.data import DataLoader

from data_utils.landscape_dataset import LandscapeDataset, description_four, load_samples
from utils.data_processor import UserInputProcessor


class LocalDatasetTests(unittest.TestCase):
    def test_description_four_formats(self):
        for value in ['Description 1: a\\n\\nDescription 2: b\\n\\nDescription 3: c\\n\\nDescription 4: d',
                      'a\n\nb\n\nc\n\nd', 'a\\n\\nb\\n\\nc\\n\\nd']:
            self.assertEqual(description_four(value), 'd')
        for value in ['', None, 'only one paragraph']:
            with self.assertRaises(ValueError):
                description_four(value)

    def test_two_stages_grouped_splits_and_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{'url': f'{i}.png', 'caption': f'paint mountains {i}',
                     'texts': [{'content': '山', 'pos': [2, 3, 10, 25]}, {'content': '蘋', 'pos': [2, 25, 10, 25]}]} for i in range(40)]
            for variant in (1, 2):
                folder = root / f'dataset1_manu_made_{variant}'
                (folder / 'images').mkdir(parents=True)
                (folder / 'training_data.json').write_text(json.dumps(rows), encoding='utf-8')
                for row in rows:
                    Image.new('RGB', (32, 64), 'white').save(folder / 'images' / row['url'])
            folder = root / 'dataset2_only_landscape'
            (folder / 'images').mkdir(parents=True)
            workbook = Workbook()
            workbook.active.append(['image_name', '文字描述'])
            for row in rows:
                workbook.active.append([row['url'], 'first\\n\\nsecond\\n\\nthird\\n\\nfourth'])
                Image.new('RGB', (32, 64), 'white').save(folder / 'images' / row['url'])
            workbook.save(folder / 'data1_9000data_nowords_COMBINED.xlsx')
            workbook.close()
            for stage in (1, 2):
                args = SimpleNamespace(dataset_root=tmp, training_stage=stage,
                                       resolution_h=64, resolution_w=64, validation_fraction=0.25)
                train = LandscapeDataset(args)
                validation = LandscapeDataset(args, 'validation')
                self.assertFalse({s['url'] for s in train.samples} & {s['url'] for s in validation.samples})
                self.assertEqual(len(train) + len(validation), 80 if stage == 1 else 40)
                batch = next(iter(DataLoader(train, batch_size=2)))
                self.assertEqual(batch['gt_im'].shape, (2, 3, 64, 64))
                self.assertEqual(batch['text_embeds'].shape, (2, 112, 128))
                if stage == 2:
                    self.assertEqual(batch['caption'], ['fourth', 'fourth'])
                    self.assertEqual(torch.count_nonzero(batch['text_embeds']).item(), 0)
                    self.assertEqual(batch['num_texts'].tolist(), [0, 0])
                else:
                    item = train[0]
                    inferred = UserInputProcessor()([rows[0]['texts'][0]], '', width=64, height=64)
                    torch.testing.assert_close(item['text_embeds'], inferred['text_embeds'][0])
                    self.assertEqual(item['num_texts'], 1)
                    self.assertEqual(train.report['invalid_boxes'], 80)
                    # No loading or resizing mutates stored source coordinates.
                    self.assertEqual(train.samples[0]['texts'][0]['pos'], [2, 3, 10, 25])
