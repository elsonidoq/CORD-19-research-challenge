from pathlib import Path
from tqdm import tqdm
import json


def iter_papers(data_path, only_some=False):
    """
    Iterate over all directories and yields all papers
    """
    dirs = 'comm_use_subset noncomm_use_subset pmc_custom_license biorxiv_medrxiv'.split()
    data_path = Path(data_path)
    n = 0
    for dir in dirs:
        fnames = (data_path / dir / dir).glob('*')
        for fname in fnames:
            with fname.open() as f:
                content = json.load(f)
            yield content
            n += 1
            if only_some and n == 500: break

        if only_some and n == 500: break


def iter_fields(paper, flat=False):
    def get(k):
        if flat:
            return paper[k]
        else:
            res = paper
            for kk in k.split('.'): res = res.get(kk)
            return res

    for key in 'abstract body_text metadata.title'.split():
        try:
            val = get(key)
        except KeyError:
            continue

        if isinstance(val, str): val = [{'text': val}]

        for p in val:
            yield key, p
