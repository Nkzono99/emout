from pathlib import Path
from typing import List

from .emout import Emout


def search_dirs(filename_patterns: List[str]) -> List[Path]:
    dirs: List[Path] = []
    for d in filename_patterns:
        dirs += Path().cwd().glob(d)
    return list(filter(lambda d: d.is_dir(), dirs))


def load_emouts(filename_patterns: List[str]) -> List[Emout]:
    dirs = search_dirs(filename_patterns)
    datas = []
    for d in dirs:
        data = Emout(d)
        datas.append(data)
    return datas
