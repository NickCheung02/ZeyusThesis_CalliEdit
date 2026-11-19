"""Run offline CPU smoke tests: python smoke_test.py (no model downloads)."""
import os
from pathlib import Path
import sys
import tempfile
import unittest


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    # Set these before importing torch/transformers; use a writable disposable cache.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['USE_TF'] = '0'
    with tempfile.TemporaryDirectory(prefix='calliedit-smoke-') as cache:
        os.environ['HF_HOME'] = cache
        import torch
        torch.set_default_device('cpu')
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(2)
        torch.manual_seed(42)
        print(f'CPU smoke test | PyTorch {torch.__version__} | offline | float32', flush=True)
        suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_*.py')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
