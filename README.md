# philosophical search

[philosophical search](https://gustavecortal.com/philosophical-search/) is a semantic search over passages from philosophy books, encyclopedias and Wikipedia. This repository builds the corpus, encoder and index of the live demo, and searches the index from the command line as the page does.

<p align="center">
  <a href="https://gustavecortal.com/philosophical-search/"><img src="docs/demo.webp" width="720" alt="philosophical search: two questions typed in turn, each answered with three passages from different works"></a>
</p>

## Quick start

Build an index of three public-domain books and ask it a question:

```bash
git clone https://github.com/gustavecortal/philosophical-search.git
cd philosophical-search
uv run fetch.py --only se-hume-enquiry se-plato-meno se-epictetus-enchiridion
uv run extract.py && uv run passages.py && uv run model.py && uv run index.py
uv run query.py "Can virtue be taught?"
```

This takes about three minutes on a CPU and downloads the encoder (560 MB). `query.py` prints the three passages the page would show first.

## How it works

The passages come from public-domain books, the Stanford and Internet Encyclopedias of Philosophy, and Wikipedia's philosophy articles. Each is embedded with [jina-embeddings-v5-text-nano-retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval), and the page encodes the question with a 4-bit version of the same model. A first pass scores every passage by the signs of its vector alone. The best 100 are then ranked by their full vectors, and the page shows the best three from different sections.

## Rebuild the whole index

```bash
uv run fetch.py
uv run extract.py && uv run passages.py && uv run model.py && uv run index.py
```

Fetching the 6,586 sources takes about three hours, since the Stanford Encyclopedia asks for one request every five seconds. Embedding the 145,574 passages takes about an hour and a half on a GPU. `passages.py` prints a hash of the passages: the live index gives `25d7812590c1`.

## License

The code is under the [MIT license](LICENSE). The encoder is under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The encyclopedias' text is not in this repository: `fetch.py` downloads it.
