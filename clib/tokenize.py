from collections import defaultdict
from tqdm.auto import tqdm
import re
import nltk

# nltk.download('stopwords')
import json

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

from . import data


def iter_tokenized(data_path, only_some=False):
    for i, paper in enumerate(tqdm(data.iter_papers(data_path, only_some=only_some))):
        tok_paper = defaultdict(list)
        for field, value in data.iter_fields(paper):
            tok_paper[field].append({
                'text': value['text'],
                'tokens': [
                    tok.lower()
                    for tok in nltk.word_tokenize(value['text'])
                    if tok.isalnum() and tok not in stopwords
                ]
            })

        yield tok_paper

def cache_tokenized_papers(data_path, fname, only_some=False):
    with open(fname, 'w') as f:
        for i, tok_paper in enumerate(iter_tokenized(data_path, only_some=only_some)):
            if i > 0: f.write('\n')
            f.write(json.dumps(dict(tok_paper)))

def load_tokenized_papers(fname):
    papers = []
    with open(fname) as f:
        for line in tqdm(f):
            papers.append(json.loads(line))

    for paper in papers:
        for field, paragraphs in paper.items():
            for paragraph in paragraphs:
                paragraph['tokens'] = [t.lower() for t in paragraph['tokens'] if t.lower() not in stopwords]

    return papers
