__all__ = ['ROOT_DIR', 'DATA_DIR', 'NTBK_DIR', 'IMGS_DIR', 'RES_DIR',]
import os, inspect
__file = inspect.getfile(lambda: None)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file), '..'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
NTBK_DIR = os.path.join(ROOT_DIR, 'notebooks')
IMGS_DIR = os.path.join(ROOT_DIR, 'imgs')
RES_DIR = os.path.join(ROOT_DIR, 'results')
