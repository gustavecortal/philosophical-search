# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "sentence-transformers==6.2.0", "transformers==5.22.1", "torch>=2.8"]
# ///
"""Embed the passages and write the search index the web page and query.py read.

Each passage of <data>/passages.jsonl is embedded, and each of its paragraphs alone, with the encoder in encoder/config.json
through Sentence Transformers in float32, on the GPU when PyTorch finds one. Vectors are kept by text in
<data>/embeddings/, so an interrupted or repeated run embeds only new text.

Writes into <out>:

    manifest.json     encoder files and hashes, recipe, counts, file names
    bits.bin          each passage's vector as signs, one bit a dimension (dimension 8b+k is bit k of byte b), for a
                      first ranking
    table.bin         each passage's document (an index into documents.json), first and last block, paragraph count
                      (uint16 each) and the offset of its record in text.bin (uint32), little-endian
    documents.json    the documents' attribution, reading page and section starts (from <data>/sections.json)
    vectors.bin       rows of a float32 scale then int8 values (row = values * scale, 772 bytes at 768 dimensions):
                      one per passage, then one per paragraph of each passage in turn, for rescoring the first
                      ranking's best and for the preview to start at the paragraph closest to the question
    text.bin          each passage's record, UTF-8 JSON: text, section, and its place in the reading page when finer
                      than the page
    reviewed.json     the reviewed passages' rows and text, for keyword search on a device that cannot run the model
    model/            the query encoder and its tokenizer, from model.py

    uv run index.py
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import ENCODER, HERE, TABLE, int8_rows, read_jsonl, sha256, sign_bits

MODEL_FILES = ('model_q4_web.onnx', 'model_q4_web.onnx_data', 'tokenizer.json', 'tokenizer_config.json')
CHUNK = 4096  # texts embedded between two saves of the embedding store
ROWS = 65536  # rows of vectors.bin quantized at a time
RECIPE = ('repo', 'revision', 'passage_prefix', 'pooling', 'normalize')  # what the passage vectors depend on


def device():
    return 'cuda' if torch.cuda.is_available() else 'xpu' if torch.xpu.is_available() else (
        'mps' if torch.backends.mps.is_available() else 'cpu')


def embedder(data, device, batch_size):
    """embed(texts): their normalized float32 vectors as passages, from <data>/embeddings/ when embedded before. New
    vectors are saved there in chunks, so an interrupted run keeps what it embedded."""
    model = SentenceTransformer(ENCODER['repo'], revision=ENCODER['revision'], trust_remote_code=True,
                                device=device, model_kwargs={'dtype': torch.float32})
    # The recipe recorded in encoder/config.json must be the one the model's configuration applies.
    assert model.prompts == {'query': ENCODER['query_prefix'], 'document': ENCODER['passage_prefix']}, model.prompts
    recipe = {k: ENCODER[k] for k in RECIPE}
    store = data / 'embeddings' / hashlib.sha256(json.dumps(recipe).encode('utf-8')).hexdigest()[:12]
    store.mkdir(parents=True, exist_ok=True)
    known, numbers = {}, [-1]
    for f in sorted(store.glob('chunk-*.npz')):
        saved = np.load(f)
        known.update(zip(saved['keys'].tolist(), saved['vectors']))
        numbers.append(int(f.stem.split('-')[1]))
    chunk = max(numbers)  # each new chunk is numbered past the highest, so none is overwritten
    key = lambda t: hashlib.sha256(t.encode('utf-8')).hexdigest()

    def embed(texts):
        nonlocal chunk
        missing = list(dict.fromkeys(t for t in texts if key(t) not in known))
        print(f'{len(texts) - len(missing):,} of {len(texts):,} texts embedded before, {len(missing):,} to embed')
        for at in range(0, len(missing), CHUNK):
            part = missing[at:at + CHUNK]
            vectors = model.encode(part, prompt_name='document', batch_size=batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
            keys = [key(t) for t in part]
            chunk += 1
            np.savez(store / f'chunk-{chunk:05d}.npz', keys=np.array(keys), vectors=vectors)
            known.update(zip(keys, vectors))
        return np.stack([known[key(t)] for t in texts])
    return embed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--model', type=Path, default=HERE / 'data' / 'model', help="model.py's output")
    parser.add_argument('--out', type=Path, default=HERE / 'data' / 'index')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    documents = {d['document_id']: d for d in read_jsonl(args.data / 'documents.jsonl')}
    passages = read_jsonl(args.data / 'passages.jsonl')
    starts = json.loads((args.data / 'sections.json').read_text(encoding='utf-8'))
    paragraphs = [p['text'].split(ENCODER['block_separator']) for p in passages]  # the blocks the page shows

    on = device()
    embed = embedder(args.data, on, args.batch_size)
    vectors = embed([p['text'] for p in passages])
    paragraph_vectors = embed([t for texts in paragraphs for t in texts])

    used = list(dict.fromkeys(p['document_id'] for p in passages))
    index_of = {d: n for n, d in enumerate(used)}
    keys = ('title', 'author', 'translator', 'source', 'reading_url', 'tex_macros')
    document_list = [{'id': d, **{k: documents[d][k] for k in keys if documents[d].get(k)}, 'sections': starts[d]}
                     for d in used]

    records = [json.dumps({'text': p['text'], 'section': p['section'], **({'url': p['url']} if 'url' in p else {})},
                          ensure_ascii=False).encode('utf-8') for p in passages]
    offsets = np.cumsum([0] + [len(r) for r in records])
    assert offsets[-1] < 2 ** 32 and max(p['end'] for p in passages) < 2 ** 16 and max(map(len, paragraphs)) < 2 ** 16
    table = np.zeros(len(passages), dtype=TABLE)
    table['document'] = [index_of[p['document_id']] for p in passages]
    table['start'] = [p['start'] for p in passages]
    table['end'] = [p['end'] for p in passages]
    table['paragraphs'] = list(map(len, paragraphs))
    table['text'] = offsets[:-1]
    reviewed = [n for n, p in enumerate(passages) if p['run'] == 'reviewed']

    out = args.out
    (out / 'model').mkdir(parents=True, exist_ok=True)
    (out / 'bits.bin').write_bytes(sign_bits(vectors))
    (out / 'table.bin').write_bytes(table.tobytes())
    (out / 'documents.json').write_bytes(json.dumps(document_list, ensure_ascii=False).encode('utf-8'))
    with open(out / 'vectors.bin', 'wb') as f:  # quantized a slice at a time, to keep memory down
        for rows in (vectors, paragraph_vectors):
            for at in range(0, len(rows), ROWS):
                f.write(int8_rows(rows[at:at + ROWS]))
    (out / 'text.bin').write_bytes(b''.join(records))
    (out / 'reviewed.json').write_bytes(json.dumps({'rows': reviewed, 'texts': [passages[n]['text'] for n in reviewed]},
                                                   ensure_ascii=False).encode('utf-8'))
    files = {}
    for name in MODEL_FILES:
        shutil.copyfile(args.model / name, out / 'model' / name)
        files[f'model/{name}'] = {'bytes': (out / 'model' / name).stat().st_size, 'sha256': sha256(out / 'model' / name)}
    served = ('bits.bin', 'table.bin', 'documents.json', 'vectors.bin', 'text.bin', 'reviewed.json')
    manifest = {
        'count': len(passages), 'paragraph_count': len(paragraph_vectors), 'dimensions': int(vectors.shape[1]),
        'encoder': {'repo': ENCODER['repo'], 'revision': ENCODER['revision'], 'graph': f'model/{MODEL_FILES[0]}',
                    'weights': f'model/{MODEL_FILES[1]}', 'tokenizer': f'model/{MODEL_FILES[2]}',
                    'tokenizer_config': f'model/{MODEL_FILES[3]}', 'files': files,
                    'derived_from': 'onnx/model_q4.onnx; embedding lookup rewritten and vocabulary trimmed by model.py'},
        'recipe': {'query_prefix': ENCODER['query_prefix'], 'passage_prefix': ENCODER['passage_prefix'],
                   'pooling': ENCODER['pooling'], 'normalize': ENCODER['normalize'],
                   'passages_encoded_with': f'Sentence Transformers, float32, {on}'},
        'vector_format': 'rows of a float32 scale then int8 values', 'text_bytes': int(offsets[-1]),
        **{name.split('.')[0]: name for name in served},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    sizes = ', '.join(f'{name} {(out / name).stat().st_size / 1e6:.1f} MB' for name in served)
    print(f'{len(passages):,} passages ({len(reviewed)} reviewed) from {len(used):,} documents: {sizes}')


if __name__ == '__main__':
    main()
