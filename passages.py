# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.22"]
# ///
"""Cut every document into the passages the search returns.

The reviewed passages come first: reviewed.jsonl names each one's first and last block and the hash of its text,
which must match the blocks extract.py wrote. The rest of every text is cut without annotation. A passage is a run
of consecutive blocks of one section, at most 768 tokens with the encoder's passage prefix, cut into parts of about
equal length. A block over that budget alone, and a passage under MIN_WORDS words, are left out.

Reads <data>/documents.jsonl and <data>/blocks.jsonl, and writes <data>/passages.jsonl in the index's order (by
document, then by place in it): the passage's document, first and last block as places among the document's blocks
other than notes, section, text, run ("reviewed" or "auto") and, for a book from Standard Ebooks or Wikisource, the
address of its section.

    uv run passages.py
"""

import argparse
import hashlib
import json
from pathlib import Path

from common import ENCODER, HERE, pinned, read_jsonl, tokenizer, write_jsonl

MIN_WORDS = 30  # a heading-like line or a pointer to a supplement


def parts(sizes, budget):
    """Cuts consecutive items of these sizes into groups, each at most `budget` long: as few groups as greedy cutting
    allows, about equal in length. Returns each group's number of items.

    A known fault, kept so that the passages match the live index: the second pass can make more groups than greedy
    cutting, since a group closed early at the target length can leave the next one too little room. parts([12, 18,
    6, 7, 13, 19, 10], 25) gives 6 groups where greedy cutting gives 5."""
    n, size = 0, None
    for s in sizes:  # the fewest groups, cutting greedily
        if size is None or size + s > budget:
            n, size = n + 1, 0
        size += s
    target = sum(sizes) / n
    counts, count, size = [], 0, 0
    for s in sizes:
        if count and (size + s > budget or (size >= target and len(counts) < n - 1)):
            counts.append(count)
            count, size = 0, 0
        count += 1
        size += s
    return counts + [count]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--reviewed', type=Path, default=HERE / 'reviewed.jsonl')
    args = parser.parse_args()
    tok = tokenizer(pinned('tokenizer.json', args.data))
    budget, separator, prefix = ENCODER['max_passage_tokens'], ENCODER['block_separator'], ENCODER['passage_prefix']
    overhead = len(tok.encode(prefix).ids)  # the prefix and the closing token
    joint = len(tok.encode(separator, add_special_tokens=False).ids)

    documents = read_jsonl(args.data / 'documents.jsonl')
    blocks = {d['document_id']: [] for d in documents}  # each document's blocks other than notes, in order
    for b in read_jsonl(args.data / 'blocks.jsonl'):
        if b['kind'] != 'note':
            blocks[b['document_id']].append(b)
    place = {b['id']: n for doc in blocks.values() for n, b in enumerate(doc)}
    titles = {d['document_id']: d['title'] for d in documents}

    def passage(doc_id, start, end, run):
        chosen = blocks[doc_id][start:end + 1]
        # A book's or page's own title often heads its section path, sometimes twice (the Meno inside its volume).
        section = list(chosen[0]['section'])
        while section[:1] == [titles[doc_id]]:
            section.pop(0)
        return {'document_id': doc_id, 'start': start, 'end': end, 'section': section,
                'text': separator.join(b['text'] for b in chosen), 'run': run,
                **({'url': chosen[0]['url']} if chosen[0].get('url') else {})}

    passages, reviewed = [], set()
    for r in read_jsonl(args.reviewed):
        if r['document_id'] not in blocks:  # a document not extracted
            continue
        p = passage(r['document_id'], place[r['start_block']], place[r['end_block']], 'reviewed')
        if hashlib.sha256(p['text'].encode('utf-8')).hexdigest() != r['text_sha256']:
            raise SystemExit(f'{r["start_block"]}: the reviewed passage\'s text differs from the extracted blocks')
        passages.append(p)
        reviewed.update(b['id'] for b in blocks[r['document_id']][p['start']:p['end'] + 1])

    oversized = short = 0
    for doc_id, doc in blocks.items():
        counts = [len(c.ids) for c in tok.encode_batch([b['text'] for b in doc], add_special_tokens=False)]
        runs, current = [], []  # runs of free blocks within one section, as (place, tokens)
        for n, (b, c) in enumerate(zip(doc, counts)):
            free = b['id'] not in reviewed and c + overhead <= budget
            oversized += b['id'] not in reviewed and not free
            if current and (not free or b['section'] != doc[current[-1][0]]['section']):
                runs.append(current)
                current = []
            if free:
                current.append((n, c))
        runs.append(current)
        for run in filter(None, runs):
            # Each block costs its tokens and a separator; the first saves one separator and adds the overhead.
            at = 0
            for count in parts([c + joint for _, c in run], budget - overhead + joint):
                group, at = run[at:at + count], at + count
                p = passage(doc_id, group[0][0], group[-1][0], 'auto')
                if len(p['text'].split()) < MIN_WORDS:
                    short += 1
                    continue
                assert len(tok.encode(prefix + p['text']).ids) <= budget, (doc_id, p['start'])
                passages.append(p)

    order = {d['document_id']: n for n, d in enumerate(documents)}
    passages.sort(key=lambda p: (order[p['document_id']], p['start']))
    write_jsonl(args.data / 'passages.jsonl', passages)
    # A hash of every passage's text, in order, to compare with the live index's (README.md).
    corpus = hashlib.sha256(json.dumps([p['text'] for p in passages], ensure_ascii=False).encode('utf-8'))
    print(f'{len(passages):,} passages ({sum(p["run"] == "reviewed" for p in passages)} reviewed) from '
          f'{len({p["document_id"] for p in passages}):,} documents, text hash {corpus.hexdigest()[:12]}. '
          f'{oversized} blocks over the budget and {short} passages under {MIN_WORDS} words left out')


if __name__ == '__main__':
    main()
