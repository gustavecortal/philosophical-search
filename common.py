"""What the pipeline's steps share: the encoder's recipe and pinned files, and JSON Lines."""

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ENCODER = json.loads((HERE / 'encoder.json').read_text(encoding='utf-8'))


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
