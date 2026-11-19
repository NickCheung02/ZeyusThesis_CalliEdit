"""Compatibility import for stage-specific held-out validation."""
from data_utils.landscape_dataset import LandscapeDataset


class Poster_Dataset(LandscapeDataset):
    def __init__(self, args, **kwargs):
        super().__init__(args, split='validation')
