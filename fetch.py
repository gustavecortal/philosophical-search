# /// script
# requires-python = ">=3.12"
# ///
"""Download each source in sources.json once into <data>/raw/ and record where it came from.

Every source is pinned, so a later run fetches the same text:
- Standard Ebooks: the book's repository at a commit. Its text files (those listed, or the reading order without the
  pages added around every book) and its endnotes are saved together in one JSON file.
- Project Gutenberg: the generated HTML edition from an official mirror, since the Gutenberg website forbids automated
  access (https://www.gutenberg.org/policy/robot_access.html). The mirror rebuilds these files monthly, so the
  recorded hash shows when one has changed.
- Wikipedia: the MediaWiki API at a revision.
- Wikisource: each page of the work at a revision, saved together in one JSON file.
- SEP: an entry in a fixed quarterly archive edition, one request every five seconds (its robots.txt crawl-delay).
- IEP: an entry from its WordPress API. IEP keeps no archive, so the recorded date of the entry's last change and its
  hash show when it has changed.
Each site gets one request at a time, while the sites are fetched in parallel (GROUPS). Existing downloads are kept:
delete a file to fetch it again.

    uv run fetch.py                                   # every source, about three hours
    uv run fetch.py --only se-hume-enquiry 'se-plato-*'
"""

import argparse
import datetime
import fnmatch
import hashlib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import HERE, standard_ebooks_url

USER_AGENT = 'philosophical-search/1.0 (https://gustavecortal.com/philosophical-search/)'
SEP_DELAY = 5  # seconds between SEP requests, its robots.txt crawl-delay
SE_RAW = 'https://raw.githubusercontent.com/standardebooks'
# Pages Standard Ebooks adds around every book.
SE_SKIP = {'titlepage.xhtml', 'imprint.xhtml', 'colophon.xhtml', 'uncopyright.xhtml',
           'halftitlepage.xhtml', 'endnotes.xhtml', 'loi.xhtml'}


def get(url):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def spine(opf):
    """The text files of a Standard Ebooks content.opf, in reading order."""
    by_id = {i: href for href, i in re.findall(r'<item href="text/([^"]+)" id="([^"]+)"', opf)}
    return [by_id[i] for i in re.findall(r'<itemref idref="([^"]+)"', opf) if i in by_id]


def standard_ebooks(doc, sources):
    base = f'{SE_RAW}/{doc["repo"]}/{doc["commit"]}/src/epub'
    opf = get(f'{base}/content.opf').decode('utf-8')
    files = doc.get('files') or [f for f in spine(opf) if f not in SE_SKIP]
    texts = {name: get(f'{base}/text/{name}').decode('utf-8') for name in files}
    if 'text/endnotes.xhtml' in opf:
        texts['endnotes.xhtml'] = get(f'{base}/text/endnotes.xhtml').decode('utf-8')
    data = json.dumps({'content.opf': opf, 'files': texts}, ensure_ascii=False).encode('utf-8')
    url = f'https://github.com/standardebooks/{doc["repo"]}/tree/{doc["commit"]}'
    source_url = standard_ebooks_url(doc['repo'])
    return data, url, source_url, {'commit': doc['commit']}


def gutenberg(doc, sources):
    # The generated HTML edition, named the same way for every ebook.
    url = f'{sources["gutenberg_mirror"]}/cache/epub/{doc["ebook"]}/pg{doc["ebook"]}-images.html'
    return get(url), url, f'https://www.gutenberg.org/ebooks/{doc["ebook"]}', {'ebook': doc['ebook']}


def wikipedia(doc, sources):
    query = {'action': 'parse', 'oldid': doc['revision'], 'prop': 'text|revid|displaytitle',
             'format': 'json', 'formatversion': 2}
    url = f'{sources["wikipedia_api"]}?{urllib.parse.urlencode(query)}'
    data = get(url)
    revision = json.loads(data)['parse']['revid']
    return data, url, f'https://en.wikipedia.org/w/index.php?oldid={revision}', {'revision': revision}


def wikisource(doc, sources):
    pages = {}
    for page in doc['pages']:
        query = {'action': 'parse', 'oldid': page['revision'], 'prop': 'text', 'format': 'json', 'formatversion': 2}
        url = f'{sources["wikisource_api"]}?{urllib.parse.urlencode(query)}'
        pages[page['title']] = json.loads(get(url))['parse']['text']
    data = json.dumps({'pages': pages}, ensure_ascii=False).encode('utf-8')
    title = doc['pages'][0]['title'].split('/')[0]
    source_url = f'https://en.wikisource.org/wiki/{urllib.parse.quote(title.replace(" ", "_"))}'
    return data, sources['wikisource_api'], source_url, {}


def sep(doc, sources, last=[-SEP_DELAY]):
    # Each request starts at least SEP_DELAY seconds after the previous one started. Only one thread fetches SEP.
    time.sleep(max(0, last[0] + SEP_DELAY - time.monotonic()))
    last[0] = time.monotonic()
    url = f'{sources["sep_archive"]}/{doc["edition"]}/entries/{doc["entry"]}/'
    return get(url), url, url, {'edition': doc['edition']}


def iep(doc, sources):
    url = f'{sources["iep_api"]}/posts?slug={doc["entry"]}&_fields=id,slug,link,title,modified,content'
    post = json.loads(get(url))[0]
    data = json.dumps(post, ensure_ascii=False, sort_keys=True).encode('utf-8')
    return data, url, post['link'], {'modified': post['modified']}


FETCHERS = {'standard_ebooks': standard_ebooks, 'gutenberg': gutenberg,
            'wikipedia': wikipedia, 'wikisource': wikisource, 'sep': sep, 'iep': iep}
# Sources fetched one request at a time each, in parallel with the other groups: the two Wikimedia wikis share a
# group, and so do the GitHub repositories of Standard Ebooks and the Gutenberg mirror, which serve few files.
GROUPS = {'sep': 0, 'wikipedia': 1, 'wikisource': 1, 'iep': 2, 'standard_ebooks': 3, 'gutenberg': 3}
SAVE_EVERY = 100  # documents fetched between two writes of fetched.json


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    parser.add_argument('--only', nargs='+', metavar='ID', help='document IDs to fetch, or patterns such as "se-*"')
    args = parser.parse_args()
    sources = json.loads((HERE / 'sources.json').read_text(encoding='utf-8'))
    raw = args.data / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    record_path = raw / 'fetched.json'
    records = json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    lock, failed = threading.Lock(), threading.Event()

    def save():
        record_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    def path_of(doc):
        return raw / f'{doc["document_id"]}.{"html" if doc["source"] in ("gutenberg", "sep") else "json"}'

    def fetch(docs):
        for doc in docs:
            if failed.is_set():  # another group failed: stop, so that the error shows now
                return
            try:
                data, url, source_url, extra = FETCHERS[doc['source']](doc, sources)
            except BaseException:
                failed.set()
                raise
            path_of(doc).write_bytes(data)
            with lock:
                records[doc['document_id']] = {'url': url, 'source_url': source_url, 'file': path_of(doc).name,
                                               'raw_sha256': hashlib.sha256(data).hexdigest(),
                                               'retrieved': datetime.date.today().isoformat(), **extra}
                if len(records) % SAVE_EVERY == 0:
                    save()
                print(doc['document_id'], len(data), 'bytes', flush=True)

    groups = [[] for _ in range(max(GROUPS.values()) + 1)]
    for doc in sources['documents']:
        if args.only and not any(fnmatch.fnmatchcase(doc['document_id'], p) for p in args.only):
            continue
        if not (path_of(doc).exists() and doc['document_id'] in records):
            groups[GROUPS[doc['source']]].append(doc)
    try:
        with ThreadPoolExecutor(len(groups)) as pool:
            try:
                for future in [pool.submit(fetch, docs) for docs in groups]:
                    future.result()
            except BaseException:  # an error or Ctrl-C: the other groups stop after their current request
                failed.set()
                raise
    finally:
        with lock:
            save()


if __name__ == '__main__':
    main()
