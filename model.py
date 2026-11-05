# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "onnx==1.23.2", "onnxruntime==1.31.1", "tokenizers==0.23.3"]
# ///
"""Make the query encoder the web page runs: Jina's 4-bit ONNX file, made to run on ONNX Runtime Web's WebAssembly
build, with its vocabulary cut to the tokens English and French text use.

Two changes, neither of which changes a vector:
- The token embedding table is read by GatherBlockQuantized, an operator ONNX Runtime Web's WebAssembly build lacks.
  That node becomes standard operators computing the same thing: gather the token's packed row, its scales and its
  packed zero points, split each byte into two 4-bit values (low nibble first), then (value - zero point) * scale
  per block of 32.
- The tokenizer's 128,256 tokens cover many languages, and the embedding table is a third of the file. Only the
  tokens in encoder/kept-tokens.txt stay: those seen in the corpus and in samples of English and French Wikipedia, the byte
  tokens, the special tokens and every token the merges pass through to build a kept one. A text whose tokens are
  all kept is tokenized exactly as before. The download drops from 155 to 106 MB.

The result is checked against the original file on a few questions (same tokens, cosine above 0.99999), then
compared with the files the web page serves. The script exits with an error when either check fails.

    uv run model.py
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from common import ENCODER, HERE, pinned, query_encoder, sha256, tokenizer

GRAPH = 'model_q4_web.onnx'
EMBEDDING = ('model_embed_tokens_weight_quant', 'model_embed_tokens_weight_scales', 'model_embed_tokens_weight_zp')
# The files model.py writes, as https://gustavecortal.com/philosophical-search/ serves them.
LIVE = {
    'model_q4_web.onnx': '7ab9e675c7e4fd8792fcf222b66b4d03228ca6231e8cb848a99de011ca6678fb',
    'model_q4_web.onnx_data': 'aa9815841fec6f8cf84f0ac91c9fdba6302e055a949cdc0fe6938cc19c8d2e76',
    'tokenizer.json': 'e80330c2918364765bbe8472df65e15ef4c59ce18cc1d813866bff06cb6562d0',
    'tokenizer_config.json': '6c4640d432db970b2436a4386d3ee992b99e756b62c37446c3f581c8d09cbb05',
}
QUESTIONS = ['Is free will compatible with determinism?', 'Why does grief make everything feel pointless?',
             'Can virtue be taught?', 'Is reality made of things or of processes?',
             'How should I live knowing I will die?', 'Does matter exist outside the mind?',
             'La liberté consiste-t-elle à faire ce que l\'on veut ?']


def unpack(name, packed, out, nodes):
    """uint8 bytes [..., n] -> float 4-bit values [..., 2n], low nibble first."""
    nodes += [
        helper.make_node('BitwiseAnd', [packed, 'nibble_mask'], [f'{name}_low']),
        helper.make_node('BitShift', [packed, 'nibble_shift'], [f'{name}_high'], direction='RIGHT'),
        helper.make_node('Unsqueeze', [f'{name}_low', 'last_axis'], [f'{name}_low1']),
        helper.make_node('Unsqueeze', [f'{name}_high', 'last_axis'], [f'{name}_high1']),
        helper.make_node('Concat', [f'{name}_low1', f'{name}_high1'], [f'{name}_pairs'], axis=-1),
        helper.make_node('Cast', [f'{name}_pairs'], [f'{name}_float'], to=TensorProto.FLOAT),
        helper.make_node('Reshape', [f'{name}_float', f'{name}_shape'], [out]),
    ]


def rewrite(model):
    """Replaces the GatherBlockQuantized node with standard operators."""
    graph = model.graph
    [node] = [n for n in graph.node if n.op_type == 'GatherBlockQuantized']
    attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
    assert attrs == {'bits': 4, 'block_size': 32, 'gather_axis': 0, 'quantize_axis': 1}, attrs
    data, ids, scales, zero_points = node.input
    width = next(i.dims[1] for i in graph.initializer if i.name == data) * 2
    blocks = width // attrs['block_size']
    graph.initializer.extend([
        helper.make_tensor('nibble_mask', TensorProto.UINT8, [], [15]),
        helper.make_tensor('nibble_shift', TensorProto.UINT8, [], [4]),
        helper.make_tensor('last_axis', TensorProto.INT64, [1], [-1]),
        # Reshape's 0 keeps the batch and sequence dimensions.
        helper.make_tensor('values_shape', TensorProto.INT64, [4], [0, 0, blocks, attrs['block_size']]),
        helper.make_tensor('zero_points_shape', TensorProto.INT64, [4], [0, 0, blocks, 1]),
        helper.make_tensor('embedding_shape', TensorProto.INT64, [3], [0, 0, width]),
    ])
    nodes = [
        helper.make_node('Gather', [data, ids], ['row_packed'], axis=0),
        helper.make_node('Gather', [zero_points, ids], ['zero_points_packed'], axis=0),
        helper.make_node('Gather', [scales, ids], ['row_scales'], axis=0),
        helper.make_node('Unsqueeze', ['row_scales', 'last_axis'], ['row_scales1']),
    ]
    unpack('values', 'row_packed', 'row_values', nodes)
    unpack('zero_points', 'zero_points_packed', 'row_zero_points', nodes)
    nodes += [
        helper.make_node('Sub', ['row_values', 'row_zero_points'], ['row_centred']),
        helper.make_node('Mul', ['row_centred', 'row_scales1'], ['row_scaled']),
        helper.make_node('Reshape', ['row_scaled', 'embedding_shape'], [node.output[0]]),
    ]
    # First: the new nodes read only weights and input_ids. (The original file lists its node after the first layer
    # that reads it, which ONNX Runtime accepts but onnx.checker does not.)
    order = nodes + [n for n in graph.node if n is not node]
    del graph.node[:]
    graph.node.extend(order)


def kept_tokens(path):
    """Token IDs from lines such as "0-270" and "298"."""
    ids = []
    for line in Path(path).read_text(encoding='utf-8').split():
        first, _, last = line.partition('-')
        ids += range(int(first), int(last or first) + 1)
    return ids


def trim(model, spec, old):
    """Keeps the tokens `old` (sorted IDs) in the tokenizer `spec` and the model's embedding table, renumbered."""
    new_id = {o: n for n, o in enumerate(old)}
    added = {t['id'] for t in spec['added_tokens']}
    token_of = {i: t for t, i in spec['model']['vocab'].items()}
    out = copy.deepcopy(spec)
    out['model']['vocab'] = kept = {token_of[o]: new_id[o] for o in old if o not in added}
    merges = [tuple(m) if isinstance(m, list) else tuple(m.split(' ')) for m in spec['model']['merges']]
    out['model']['merges'] = [[a, b] for a, b in merges if a in kept and b in kept and a + b in kept]
    for t in out['added_tokens']:
        t['id'] = new_id[t['id']]
    if out.get('padding'):
        out['padding']['pad_id'] = new_id[out['padding']['pad_id']]
    for p in out['post_processor']['processors']:
        for special in p.get('special_tokens', {}).values():
            special['ids'] = [new_id[i] for i in special['ids']]
    for init in model.graph.initializer:
        if init.name in EMBEDDING:
            rows = numpy_helper.to_array(init)
            init.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(rows[old]), init.name))
    return out


def vectors(graph, tokenizer_path, texts):
    """Each text's vector by the web page's recipe."""
    encode = query_encoder(graph, tokenizer_path)
    return np.array([encode(t) for t in texts])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, default=HERE / 'data', help='where the original files are downloaded')
    parser.add_argument('--out', type=Path, default=HERE / 'data' / 'model')
    args = parser.parse_args()
    source = pinned('onnx/model_q4.onnx', args.data)
    pinned('onnx/model_q4.onnx_data', args.data)  # read by onnx.load beside the graph
    tokenizer_path = pinned('tokenizer.json', args.data)
    config = pinned('tokenizer_config.json', args.data)

    model = onnx.load(str(source))
    rewrite(model)
    old = kept_tokens(HERE / 'encoder' / 'kept-tokens.txt')
    spec = trim(model, json.loads(tokenizer_path.read_text(encoding='utf-8')), old)
    args.out.mkdir(parents=True, exist_ok=True)
    # onnx.save appends to an existing weights file, so a second run would write a larger, different one.
    for name in (GRAPH, GRAPH + '_data'):
        (args.out / name).unlink(missing_ok=True)
    (args.out / 'tokenizer.json').write_text(json.dumps(spec, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    (args.out / 'tokenizer_config.json').write_bytes(config.read_bytes())
    onnx.save(model, str(args.out / GRAPH), save_as_external_data=True, all_tensors_to_one_file=True,
              location=GRAPH + '_data', size_threshold=1024)

    texts = [ENCODER['query_prefix'] + q for q in QUESTIONS]
    original, trimmed = tokenizer(tokenizer_path), tokenizer(args.out / 'tokenizer.json')
    same = sum([old[i] for i in trimmed.encode(t).ids] == original.encode(t).ids for t in texts)
    cosine = np.sum(vectors(source, tokenizer_path, texts) * vectors(args.out / GRAPH, args.out / 'tokenizer.json', texts),
                    axis=1)
    print(f'{same} of {len(texts)} questions tokenize as before. Their cosine to the original model is at least '
          f'{cosine.min():.7f}')
    if same < len(texts) or cosine.min() < 0.99999:
        raise SystemExit('the new model differs from the original')
    different = [name for name, digest in LIVE.items() if sha256(args.out / name) != digest]
    print(f'Wrote {args.out}. Identical to the files the web page serves: {len(LIVE) - len(different)} of {len(LIVE)}')
    if different:
        raise SystemExit(f'different from the served files: {", ".join(different)}')


if __name__ == '__main__':
    main()
