# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2", "onnxruntime>=1.20", "tokenizers>=0.22"]
# ///
"""Checks the passage cutting, and query.py's search on a toy index against the web page's own code: the expected
values below were computed by the page's JavaScript (worker.mjs, page.mjs and keyword.mjs) on the same toy data.

    uv run test.py   (exits 1 on a failure)
"""

import sys
from types import SimpleNamespace

import numpy as np

from passages import parts
from query import arrange, best, keyword_ranker, link, rank, sign_scores


def toy_index(count=400, dimensions=64, documents=12):
    """Random passages, 40 of them repeated (equal scores), overlapping one another within their documents."""
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(count, dimensions)).astype(np.float32)
    vectors[200:240] = vectors[:40]
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = rng.normal(size=dimensions).astype(np.float32)
    query /= np.linalg.norm(query)
    scales = np.abs(vectors).max(axis=1) / 127
    rows = np.zeros((count, 4 + dimensions), np.uint8)
    rows[:, :4] = scales.astype('<f4')[:, None].view(np.uint8)
    rows[:, 4:] = np.round(vectors / scales[:, None]).astype(np.int8).view(np.uint8)
    bits = np.packbits(vectors > 0, axis=1, bitorder='little')
    document = rng.integers(0, documents, count)
    start = rng.integers(0, 60, count)
    end = start + rng.integers(0, 4, count)
    sections = [sorted(rng.choice(np.arange(1, 60), 3, replace=False).tolist()) for _ in range(documents)]
    passage = lambda i: {'document': int(document[i]), 'start': int(start[i]), 'end': int(end[i]),
                         'section': int(sum(s <= start[i] for s in sections[document[i]]))}
    return SimpleNamespace(bits=bits, vectors=rows), query, passage


# The page's results on toy_index(): the best 10 by signs, the best 100 rescored, the first one's score, the order
# shown, then a keyword ranking and a link.
SIGNS = [288, 178, 223, 23, 109, 379, 312, 11, 211, 83]
RANKING = [
    299, 243, 211, 11, 149, 109, 178, 83, 301, 354, 210, 10, 379, 313, 201, 1, 30, 230, 31, 231, 162, 198, 35,
    235, 144, 370, 174, 131, 127, 303, 71, 186, 315, 365, 258, 384, 244, 223, 23, 73, 216, 16, 79, 312, 307,
    145, 74, 248, 102, 175, 197, 89, 288, 267, 376, 256, 176, 24, 224, 352, 147, 161, 380, 123, 240, 319, 271,
    78, 399, 81, 196, 221, 21, 195, 333, 276, 69, 326, 192, 220, 20, 364, 67, 340, 60, 191, 132, 397, 135, 76,
    111, 347, 251, 289, 72, 39, 239, 341, 242, 270,
]
BEST = 0.32822819310822
SHOWN = [
    299, 243, 211, 11, 149, 109, 178, 301, 354, 210, 10, 379, 313, 201, 1, 30, 230, 31, 231, 162, 144, 35, 370,
    174, 235, 131, 127, 303, 71, 186, 258, 384, 244, 223, 23, 216, 16, 79, 312, 307, 145, 74, 248, 175, 197, 89,
    267, 376, 224, 352, 147, 380, 123, 240, 319, 271, 78, 399, 81, 221, 21, 195, 333, 326, 220, 20, 364, 67,
    340, 135, 76, 111, 251, 289, 72, 341, 242, 270,
]
KEYWORD = [2, 0, 3]
LINK = "https://example.org/page#:~:text=It%20is%E2%80%94so%3F%20(Yes)%20l'%C3%A9t%C3%A9%2050%25%20done%2Dish"


def main():
    index, query, passage = toy_index()
    ranking, score = rank(index, query)
    texts = ['Virtue can be taught, says Protagoras.', 'Nothing here.', 'Can virtue be taught? Can it?',
             'Taught by whom: virtue, or knowledge?']
    checks = {
        'passages are cut into as few parts as fit the budget, about equal in length':
            parts([100] * 9, 450) == [3, 3, 3] and parts([300] * 4, 700) == [2, 2]
            and parts([200, 50, 50, 300, 100], 400) == [3, 2],
        "signs rank as the page's do": best(sign_scores(query, index.bits), 10) == SIGNS,
        "the best 100, rescored by their 8-bit vectors, come in the page's order, ties included": ranking == RANKING,
        "the first score is the page's to the last bit": score == BEST,
        'overlaps are left out and each group mixes sections and documents as on the page':
            arrange(ranking, passage) == SHOWN,
        "keyword search ranks as the page's does": keyword_ranker(texts)('can virtue be taught') == KEYWORD,
        "a link opens the original at the passage's first words, encoded as on the page":
            link({'reading_url': 'https://example.org/page'}, {'text': "It is—so? (Yes) l'été 50% done-ish too"}) == LINK,
    }
    for name, ok in checks.items():
        print(f'{"ok  " if ok else "FAIL"} {name}')
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
