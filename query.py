# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "onnxruntime>=1.20", "tokenizers>=0.22"]
# ///
"""Search an index index.py wrote, as the web page does, and print the passages it would show.

README.md describes the method. The arithmetic follows the page's worker (rounding, integer sums, the order of ties),
so both rank alike.

    uv run query.py "Can virtue be taught?"
    uv run query.py --index path/to/index --groups 2 --whole "Is free will compatible with determinism?"
"""

import argparse
import json
import math
import re
import sys
import textwrap
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np

from common import ENCODER, HERE, TABLE, query_encoder

CANDIDATES = 100  # passages the first ranking passes on
GROUP = 3  # results at first and per Show more
# Thresholds on the first result's score, set on 60 judged evaluation questions (README.md): below NO_MATCH nothing
# is shown, below LOOSE the results are said to match only loosely.
NO_MATCH, LOOSE = 0.30, 0.40
SOURCES = {'sep': 'Stanford Encyclopedia of Philosophy', 'iep': 'Internet Encyclopedia of Philosophy'}


class Index:
    """The files of an index folder (index.py describes them), read as the page reads them."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text(encoding='utf-8'))
        file = lambda key: self.folder / self.manifest[key]
        self.documents = json.loads(file('documents').read_text(encoding='utf-8'))
        self.table = np.fromfile(file('table'), dtype=TABLE)
        count, dimensions = self.manifest['count'], self.manifest['dimensions']
        self.bits = np.fromfile(file('bits'), dtype=np.uint8).reshape(count, dimensions // 8)
        self.vectors = np.memmap(file('vectors'), dtype=np.uint8, mode='r').reshape(-1, 4 + dimensions)
        self.text_path = file('text')
        # vectors.bin holds a row per passage, then each passage's paragraphs' rows in turn.
        self.first_paragraph = count + np.concatenate([[0], np.cumsum(self.table['paragraphs'].astype(np.int64))])

    def encoder(self):
        """encode(text): its normalized vector, as float32 values. The model files are checked against the manifest
        by size only: hashing 106 MB would slow every question, and model.py checks the hashes when it writes them."""
        encoder = self.manifest['encoder']
        for path, file in encoder['files'].items():
            if (self.folder / path).stat().st_size != file['bytes']:
                raise SystemExit(f'{self.folder / path} does not have the size the manifest gives')
        return query_encoder(self.folder / encoder['graph'], self.folder / encoder['tokenizer'])

    def passage(self, i):
        """The passage's document (an index into documents), first and last block, top-level section and paragraph
        count, from table.bin."""
        row = self.table[i]
        sections = self.documents[row['document']].get('sections', [])
        return {'document': int(row['document']), 'start': int(row['start']), 'end': int(row['end']),
                'section': sum(s <= row['start'] for s in sections),  # how many sections start at or before it
                'paragraphs': int(row['paragraphs'])}

    def record(self, i):
        """The passage's text, section and place in the reading page."""
        start = int(self.table['text'][i])
        end = int(self.table['text'][i + 1]) if i + 1 < len(self.table) else self.manifest['text_bytes']
        with open(self.text_path, 'rb') as f:
            f.seek(start)
            return json.loads(f.read(end - start).decode('utf-8'))


def sign_scores(query, bits):
    """Each passage's score by signs, in the page's integers: per byte of signs, the sum of its eight dimensions'
    values with their signs, times 2^16 and rounded half up."""
    width = bits.shape[1]
    q = query.astype(np.float64).reshape(width, 8)
    v = np.arange(256)
    sums = np.zeros((width, 256))
    for k in range(8):  # added in the page's order, so the sums are the same doubles
        sums += np.where((v >> k) & 1, q[:, k:k + 1], -q[:, k:k + 1])
    scaled = sums * 65536
    floor = np.floor(scaled)
    table = (floor + (scaled - floor >= 0.5)).astype(np.int64)  # JavaScript's Math.round
    scores = np.zeros(len(bits), np.int64)
    for b in range(width):
        scores += table[b][bits[:, b]]
    return scores.astype(np.int32)


def best(scores, k):
    """The indexes of the `k` highest scores, best first, in the page's order for equal scores: a heap holding the best
    so far with the lowest at its root, then a stable sort."""
    scores, heap = scores.tolist(), []
    for i, s in enumerate(scores):
        if len(heap) < k:
            heap.append(i)
            c = len(heap) - 1
            while c > 0 and scores[heap[c]] < scores[heap[(c - 1) >> 1]]:
                p = (c - 1) >> 1
                heap[c], heap[p] = heap[p], heap[c]
                c = p
        elif s > scores[heap[0]]:
            heap[0], c = i, 0
            while True:
                left, right, m = 2 * c + 1, 2 * c + 2, c
                if left < k and scores[heap[left]] < scores[heap[m]]:
                    m = left
                if right < k and scores[heap[right]] < scores[heap[m]]:
                    m = right
                if m == c:
                    break
                heap[c], heap[m] = heap[m], heap[c]
                c = m
    return sorted(heap, key=lambda i: -scores[i])


def dot(query, row):
    """A question's dot product with a row of vectors.bin, summed in the page's order."""
    scale = float(row[:4].view('<f4')[0])
    s = 0.0
    for q, v in zip(query.tolist(), row[4:].view(np.int8).tolist()):
        s += q * v
    return s * scale


def rank(index, query):
    """The best candidates, best first, and the first one's score."""
    candidates = best(sign_scores(query, index.bits), CANDIDATES)
    exact = {i: dot(query, index.vectors[i]) for i in candidates}
    order = sorted(candidates, key=lambda i: -exact[i])
    return order, exact[order[0]]


def arrange(ranking, passage):
    """The order the page shows a ranking in: overlapping passages left out, then groups of GROUP from different
    sections, at most two from one document. passage(i) gives document, section, start and end."""
    of = {i: passage(i) for i in ranking}
    pending = []
    for i in ranking:
        p = of[i]
        if not any(q['document'] == p['document'] and q['start'] <= p['end'] and p['start'] <= q['end']
                   for q in map(of.get, pending)):
            pending.append(i)
    order = []
    while pending:
        group, rest = [], []
        for i in pending:
            p = of[i]
            siblings = [of[j] for j in group if of[j]['document'] == p['document']]
            taken = len(group) < GROUP and len(siblings) < 2 and not any(q['section'] == p['section'] for q in siblings)
            (group if taken else rest).append(i)
        order += group
        pending = rest
    return order


def lead(index, query, i):
    """The passage's paragraph closest to the question."""
    first, count = index.first_paragraph[i], index.passage(i)['paragraphs']
    closest, top = 0, -math.inf
    for p in range(count):
        s = dot(query, index.vectors[first + p])
        if s > top:
            closest, top = p, s
    return closest


def keyword_ranker(texts, k1=1.2, b=0.75):
    """BM25 over lowercased words, the page's ranking on a device that cannot run the model."""
    words = lambda text: re.findall(r'\w+', text.lower())
    docs = [Counter(words(t)) for t in texts]
    lengths = [sum(d.values()) for d in docs]
    average = sum(lengths) / len(docs)
    frequency = Counter(w for d in docs for w in d)
    idf = {w: math.log(1 + (len(docs) - n + 0.5) / (n + 0.5)) for w, n in frequency.items()}

    def rank(query):  # indices of the texts sharing a word with the query, best first
        terms = [w for w in words(query) if w in idf]
        scores = []
        for d, length in zip(docs, lengths):
            s = 0
            for w in terms:
                f = d.get(w, 0)
                s += idf[w] * f * (k1 + 1) / (f + k1 * (1 - b + b * length / average))
            scores.append(s)
        return sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
    return rank


def link(document, record):
    """The original at this passage: its section's address where the index records one, otherwise its opening
    words as a text fragment."""
    url = record.get('url') or document['reading_url']
    if '#' in url:
        return url
    opening = ' '.join(record['text'].split()[:6])
    return f'{url}#:~:text=' + urllib.parse.quote(opening, safe="!*'()").replace('-', '%2D')


def source_line(document, record):
    title = document['title']
    line = f'{title}, Wikipedia' if document['source'] == 'wikipedia' else f'{document["author"]}, {title}'
    if document['source'] in SOURCES:
        line += f', {SOURCES[document["source"]]}'
    if document.get('translator'):
        line += f', translated by {document["translator"]}'
    return ', '.join([line, *record['section']])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('question')
    parser.add_argument('--index', type=Path, default=HERE / 'data' / 'index', help="index.py's output")
    parser.add_argument('--groups', type=int, default=1, help='groups of three to show (the page shows one, then '
                                                              'one more per Show more)')
    parser.add_argument('--whole', action='store_true', help='print each passage from its start')
    parser.add_argument('--keyword', action='store_true', help='rank the reviewed passages by BM25 instead')
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')  # passages hold dashes and accents, whatever the console's code page
    index = Index(args.index)
    query, note = None, ''
    if args.keyword:
        reviewed = json.loads((index.folder / index.manifest['reviewed']).read_text(encoding='utf-8'))
        ranking = [reviewed['rows'][k] for k in keyword_ranker(reviewed['texts'])(args.question)]
    else:
        query = index.encoder()(index.manifest['recipe']['query_prefix'] + args.question)
        ranking, score = rank(index, query)
        if score < NO_MATCH:
            ranking = []
        elif score < LOOSE:
            note = 'These passages match your question only loosely.'
    order = arrange(ranking, index.passage)[:GROUP * args.groups]
    if not order:
        print('No passage matches your question.')
    if note:
        print(note)
    for n, i in enumerate(order, 1):
        document, record = index.documents[index.table['document'][i]], index.record(i)
        paragraphs = record['text'].split(ENCODER['block_separator'])
        first = 0 if args.whole or query is None or len(paragraphs) == 1 else lead(index, query, i)
        print(f'\n{n}. {source_line(document, record)}\n')
        for k, paragraph in enumerate(paragraphs[first:]):
            for line in paragraph.split('\n'):  # verse keeps its lines
                print(textwrap.fill(('[…] ' if first and not k else '') + line, 100, initial_indent='   ',
                                    subsequent_indent='   '))
            print()
        print(f'   {link(document, record)}')


if __name__ == '__main__':
    main()
