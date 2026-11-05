"""What the pipeline's steps share: the encoder's recipe and pinned files, the query encoder, the index's formats,
and JSON Lines."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ENCODER = json.loads((HERE / 'encoder.json').read_text(encoding='utf-8'))
# A row of table.bin: the passage's document, first and last block, paragraph count, and the offset of its record in
# text.bin.
TABLE = [('document', '<u2'), ('start', '<u2'), ('end', '<u2'), ('paragraphs', '<u2'), ('text', '<u4')]


def standard_ebooks_url(repo):
    """A Standard Ebooks book's page, from its repository's name: john-stuart-mill_on-liberty is
    https://standardebooks.org/ebooks/john-stuart-mill/on-liberty."""
    return f'https://standardebooks.org/ebooks/{repo.replace("_", "/")}'


def section_without_title(section, title):
    """The section path without the document's own title, which often heads it, sometimes twice (the Meno inside its
    volume)."""
    section = list(section)
    while section[:1] == [title]:
        section.pop(0)
    return section


def int8_rows(rows):
    """Each row as its float32 scale then its int8 values (row = values * scale)."""
    import numpy as np
    scales = np.abs(rows).max(axis=1) / 127
    quantized = np.round(rows / scales[:, None]).astype(np.int8)
    out = np.zeros((len(rows), 4 + rows.shape[1]), np.uint8)
    out[:, :4] = scales.astype('<f4')[:, None].view(np.uint8)
    out[:, 4:] = quantized.view(np.uint8)
    return out.tobytes()


def sign_bits(rows):
    """Each row's signs, one bit a dimension: dimension 8b+k is bit k of byte b."""
    import numpy as np
    return np.packbits(rows > 0, axis=1, bitorder='little').tobytes()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def pinned(name, data):
    """A file of the encoder's repository at its pinned revision, downloaded once into <data>/models/ and checked
    against its hash in encoder.json."""
    path = Path(data) / 'models' / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/{ENCODER["repo"]}/resolve/{ENCODER["revision"]}/{name}'
        part = path.with_name(path.name + '.part')
        with urllib.request.urlopen(url, timeout=120) as response, open(part, 'wb') as f:
            shutil.copyfileobj(response, f)
        part.replace(path)
    if sha256(path) != ENCODER['files'][name]:
        raise SystemExit(f'{path} does not match its hash in encoder.json: delete it to download it again')
    return path


def tokenizer(path):
    from tokenizers import Tokenizer  # here, so that the steps without a tokenizer do not need the package
    tok = Tokenizer.from_file(str(path))
    tok.no_padding()  # the file pads batches to their longest text, which would inflate counts
    tok.no_truncation()
    return tok


def query_encoder(graph, tokenizer_path):
    """encode(text): the text's vector by the web page's recipe, as float32 values. One input at a time, so there is
    no padding, the last token's hidden state, normalized in double precision as the page's JavaScript does."""
    import math
    import numpy as np
    import onnxruntime as ort
    tok = tokenizer(tokenizer_path)
    session = ort.InferenceSession(str(graph), providers=['CPUExecutionProvider'])

    def encode(text):
        ids = np.array([tok.encode(text).ids], dtype=np.int64)
        [hidden] = session.run(['last_hidden_state'], {'input_ids': ids, 'attention_mask': np.ones_like(ids)})
        last = hidden[0, -1].astype(np.float64)
        return (last / math.hypot(*last)).astype(np.float32)
    return encode
