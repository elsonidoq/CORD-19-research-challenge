from tqdm.auto import tqdm
from IPython.display import display, Markdown
from nltk import sent_tokenize
import re

from . import data

dm = lambda x: display(Markdown(x))


def bold(sent, part):
    part = part.strip()
    return sent.replace(part, f'**{part}**')


def contains(i1, i2):
    return i1[0] <= i2[0] and i1[1] >= i2[1]


def intersects(i1, i2):
    if contains(i1, i2) or contains(i2, i1): return True

    if i2[0] <= i1[0] <= i2[1] or i2[0] <= i1[1] <= i2[1]: return True
    return False


def get_snippets(papers, tokens):
    seeds_pat = '|'.join(tokens)
    pat = re.compile(f'(?P<p>\W|^)(?P<w>{seeds_pat})(?P<n>\W|$)', re.I)
    
    s_tokens = set(tokens)
    res = []
    for pid, paper in enumerate(tqdm(papers)):
        for field, value in data.iter_fields(paper, flat=True):
            if len(s_tokens.intersection(value['tokens'])) == 0: continue
            if len(value['tokens']) < 10: continue

            text = value['text']
            n = 0
            ws = []
            snippet = pat.sub('\g<p>**\g<w>**\g<n>', text)
            for match in pat.finditer(text):
                w = match.groupdict()['w'].lower()
                n += tokens[w]
                ws.append(w)
                
            res.append({
                'n': n,
                'pid': pid,
                'field': field,
                'snippet': snippet,
                'queries': set(ws),
                'score': len(ws) / len(value['tokens'])
            })

    return res


def add_snippets_bigrams(papers, seeds, word_related):
    seeds_pat = '|'.join(w for w in seeds)
    related_words_pat = '|'.join(w for w in word_related)
    rpat = re.compile(f"(\W|^)(?P<seed>{seeds_pat})\W.*?\W(?P<word>{related_words_pat})(\W|$)")
    lpat = re.compile(f"(\W|^)(?P<word>{related_words_pat})\W.*?\W(?P<seed>{seeds_pat})(\W|$)")

    for paper in tqdm(papers):
        paper['snippets'] = paper_snippets = []
        for field, value in data.iter_fields(paper, flat=True):
            for sent in sent_tokenize(value):
                lmatch = lpat.search(sent)
                rmatch = rpat.search(sent)

                show = True
                lgd = lmatch.groupdict() if lmatch else None
                rgd = rmatch.groupdict() if rmatch else None

                if lmatch is not None and rmatch is not None:
                    if intersects(lmatch, rmatch):
                        start = min(lmatch.start(), rmatch.start())
                        end = max(lmatch.end(), rmatch.end())
                        sent = bold(sent, sent[start:end])
                    else:
                        sent = bold(sent, sent[lmatch.start():lmatch.end()])
                        sent = bold(sent, sent[rmatch.start():rmatch.end()])
                elif lmatch is not None:
                    sent = bold(sent, sent[lmatch.start():lmatch.end()])
                elif rmatch is not None:
                    sent = bold(sent, sent[rmatch.start():rmatch.end()])
                else:
                    show = False

                if show:
                    paper_snippets.append(dict(sent=sent, lgd=lgd, rgd=rgd))
