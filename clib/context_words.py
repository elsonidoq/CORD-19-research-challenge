import numpy as np
import json
from pathlib import Path
from collections import namedtuple
from itertools import chain
from tqdm.auto import tqdm

ContextStats = namedtuple('ContextStats', ['unigrams', 'bigrams'])


def get_context_stats(papers, window_size=50):
    bigrams = {}
    unigrams = {}

    for paper in tqdm(papers):
        for paragraphs in paper.values():
            tokens = list(chain(*[e['tokens'] for e in paragraphs]))

            for i, t in enumerate(tokens):
                if t not in unigrams: unigrams[t] = 0

                unigrams[t] += 1

                lb = max(0, i - window_size)
                ub = min(len(tokens) - 1, i + window_size)

                if t not in bigrams: bigrams[t] = {}
                t_bigrams = bigrams[t]
                for j in range(lb, ub):
                    t2 = tokens[j]
                    if t2 not in t_bigrams: t_bigrams[t2] = 0
                    bigrams[t][t2] += 1

    return ContextStats(dict(unigrams), dict(bigrams))


def save_stats(stats, path, min_bigram_cnt=5, max_related=500):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    print('saving unigrams...')
    with (path / 'unigrams.jl').open('w') as f:
        for i, (tok, cnt) in enumerate(tqdm(stats.unigrams.items())):
            if i > 0: f.write('\n')
            f.write(json.dumps({'t': tok, 'c': cnt}))

    print('saving bigrams...')
    with (path / 'bigrams.jl').open('w') as f:
        for i, (tok, related) in enumerate(tqdm(stats.bigrams.items())):
            if i > 0: f.write('\n')

            p_related = {t: cnt for t, cnt in related.items() if cnt >= min_bigram_cnt}
            if len(p_related) > max_related:
                p_related = dict(sorted(p_related.items(), key=lambda x: -x[1])[:max_related])

            f.write(json.dumps({'t': tok, 'r': p_related}))


def load_stats(path):
    path = Path(path)

    unigrams = {}
    with (path / 'unigrams.jl').open() as f:
        for line in tqdm(f, desc='loading unigrams'):
            d = json.loads(line)
            unigrams[d['t']] = d['c']

    bigrams = {}
    with (path / 'bigrams.jl').open() as f:
        for line in tqdm(f, desc='loading bigrams'):
            d = json.loads(line)
            bigrams[d['t']] = d['r']

    return ContextStats(unigrams, bigrams)


def get_scores(full_stats, min_count=5):
    scores = {}
    len_vocab = len(full_stats.unigrams)

    for tok1, related in tqdm(full_stats.bigrams.items()):
        scores[tok1] = tok1_scores = {}
        for tok2, bigram_count in related.items():
            tok1_scores[tok2] = (
                    (bigram_count - min_count) / full_stats.unigrams[tok1] / full_stats.unigrams[tok2] * len_vocab
            )
    return scores


def get_comparative_scores(stats, ref_stats, min_count=5):
    scores = {}
    out_ref = []

    for tok1, related in tqdm(stats.bigrams.items()):
        scores[tok1] = tok1_scores = {}

        if tok1 not in ref_stats.unigrams:
            out_ref.append(tok1)
            continue

        for tok2, bigram_count in related.items():
            if bigram_count < min_count: continue

            covid_prob = bigram_count / stats.unigrams[tok1]
            general_prob = ref_stats.bigrams.get(tok1, {}).get(tok2, 0.1) / ref_stats.unigrams[tok1]

            tok1_scores[tok2] = covid_prob * np.log(covid_prob / general_prob)

    return scores, out_ref
