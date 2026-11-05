# /// script
# requires-python = ">=3.12"
# dependencies = ["beautifulsoup4==4.15.0"]
# ///
"""Split each downloaded source into ordered blocks: paragraphs, dialogue turns, quotations, verse, list items and
notes, each with the section path it belongs to.

Reads the sources fetch.py downloaded into <data>/raw/ and writes <data>/documents.jsonl (one record per source:
attribution, snapshot, hashes, what was left out as editorial) and <data>/blocks.jsonl. A block ID is
"<document_id>:<position>". The reviewed passages (corpus/reviewed.jsonl) name their first and last block and the hash of
their text, so passages.py notices if extraction changes.

Text keeps the source's wording. clean() and text_of() make the only changes: Unicode NFC, runs of whitespace
collapsed to one space except line breaks in verse, word joiners, zero-width spaces and soft hyphens removed, page
numbers and note markers removed (note markers are kept as the block's note IDs).

    uv run extract.py
"""

import argparse
import hashlib
import json
import multiprocessing
import re
import unicodedata
import urllib.parse
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from common import HERE, standard_ebooks_url

# A link to a note, by the names editions give their notes: Gutenberg's #linknote-11, #Footnote_1, #note_4, #f3 and
# #ch01fn3; Wikipedia's #endnote_a. Links back from a note to the text stay: extraction reads them.
NOTE_LINK = re.compile(r'#(linknote-|footnote_|note_|f\d|ch\d+fn\d+$|endnote_)', re.I)
LINE_BREAK = ' '  # stands for <br> until whitespace is collapsed

# Sections and files left out of a Standard Ebooks document because someone other than the
# author wrote them: editors' introductions, essays and borrowed epigraphs.
SE_EXCLUDE = {
    'se-plato-meno': {'meno-introduction': "Jowett's introduction",
                      'on-the-ideas-of-plato': "Jowett's essay On the Ideas of Plato",
                      'meno-dramatis-personae': 'list of speakers'},
    'se-mill-on-liberty': {'introduction.xhtml': "W. L. Courtney's introduction",
                           'epigraph.xhtml': 'epigraph quoting Wilhelm von Humboldt'},
    'se-nietzsche-genealogy-of-morals': {'editors-note.xhtml': "the English edition's editor's note"},
}

# Each Gutenberg edition is laid out differently: where the work starts and stops, which
# headings form the section path, and how a speaker opens a dialogue turn.
GUTENBERG = {
    'pg26659': {
        'start': ('h3', 'PREFACE.'),
        'skip': (('h2', 'CONTENTS.'), ('h3', 'THE WILL TO BELIEVE.')),
        'stop': ('h3', 'INDEX.'),
        'headings': ['h3', 'h4'],
        'excluded': ['title page, dedication, contents and index'],
    },
    'pg59709': {
        'start': ('h2', 'CHAPTER I.'),
        'stop': ('h2', 'INDEX'),
        'headings': ['h2'],
        'drop': 'div.blockquot',
        'rejoin_fragments': True,
        'drop_bracketed': True,  # Giles's own remarks, such as "[This chapter is supplementary to ch. v.]"
        'excluded': ["Giles's introduction, Aubrey Moore's note, chapter arguments, "
                     "the translator's notes and footnotes, index, errata"],
    },
    'pg10732': {
        'start': ('h2', 'ON THE SUFFERINGS OF THE WORLD.'),
        'headings': ['h2', 'h3'],
        'excluded': ["the translator's note"],
    },
    'pg38427': {
        'start': ('h1', 'Preface To The First Edition.'),
        'stop': ('h1', 'Footnotes'),
        'headings': ['h1', 'h2'],
        'excluded': ["the translators' preface, contents and footnotes"],
    },
    'pg68611': {
        'start': ('h2', 'PREFACE'),
        'stop': ('h2', 'INDEX'),
        'headings': ['h2'],
        'excluded': ['contents and index'],
    },
    'pg3296': {
        'start': ('h2', 'BOOK I'),
        'headings': ['h2'],
        'excluded': ['title page'],
    },
    # Paul's letters in the World English Bible: one heading holding "chapter:verse" lines.
    'pg8272': {'verses': 'Romans', 'excluded': ["the translators' notes in braces"]},
    'pg8273': {'verses': '1 Corinthians', 'excluded': ["the translators' notes in braces"]},
    'pg8275': {'verses': 'Galatians', 'excluded': ["the translators' notes in braces"]},
}

WIKIPEDIA_STOP = {'See also', 'References', 'Notes', 'Further reading', 'External links',
                  'Bibliography', 'Sources', 'Citations', 'Footnotes', 'Works cited'}


def clean(text, verse=False):
    text = unicodedata.normalize('NFC', text)
    text = re.sub('[⁠​­]', '', text)
    text = re.sub(rf'[^\S{LINE_BREAK}]+', ' ', text)
    text = re.sub(f' ?{LINE_BREAK} ?', '\n' if verse else ' ', text)
    return text.strip()


# Characters LaTeX reads as commands, as they are written inside \text{}.
LATEX_TEXT = str.maketrans({'\\': '\\textbackslash{}', '{': '\\{', '}': '\\}', '$': '\\$', '&': '\\&', '#': '\\#',
                            '_': '\\_', '%': '\\%', '^': '\\textasciicircum{}', '~': '\\textasciitilde{}'})


def latex_text(text, command):
    """Words as LaTeX text set by `command` (\\text or \\textbf). A number stays as it is, set as math."""
    text = text.strip()
    return f'{command}{{{text.translate(LATEX_TEXT)}}}' if text and not re.fullmatch(r'-?\d+', text) else text


def table_formula(table):
    """A table of short cells (SEP's decision tables) as a displayed LaTeX array, so it renders like the formulas
    around it: the column and row headings in bold, formulas in cells kept as math. None for a table of long text,
    such as a chronology, which a formula could not wrap, or for one with merged cells."""
    rows = []
    for tr in table.find_all('tr'):
        cells = []
        for td in tr.find_all(['td', 'th']):
            command = '\\textbf' if td.name == 'th' or td.find(['strong', 'b']) else '\\text'
            text = text_of(td)[0]
            if len(text) > 40 or td.get('colspan', '1') != '1' or td.get('rowspan', '1') != '1':
                return None
            # re.split alternates text and the formulas between \( and \), which stay as they are.
            pieces = re.split(r'\\\((.*?)\\\)', re.sub(r'^[−–](?=\d)', '-', text))
            cells.append(''.join(piece if i % 2 else latex_text(piece, command) for i, piece in enumerate(pieces)))
        rows.append(cells)
    if not rows:
        return None
    columns = 'l' + 'c' * (max(map(len, rows)) - 1)
    body = ' \\\\ '.join(' & '.join(cells) for cells in rows)
    return f'\\[\\begin{{array}}{{{columns}}} {body} \\end{{array}}\\]'


def text_of(el, verse=False):
    """The element's text with notes taken out, and the notes' IDs."""
    notes = []
    marked = [a for a in el.find_all('a', href=True) if NOTE_LINK.match(a['href']) and len(a.get_text(strip=True)) <= 6]
    for a in el.select('a[epub\\:type~=noteref], a[href*="notes.html"]') + marked:
        notes.append(a.get('href', '').split('#')[-1])
        a.decompose()
    for junk in el.select('span.pagenum, span.pageno, span.ws-pagenum, sup.reference, sup.noprint, '
                          'span.mw-editsection, style'):
        junk.decompose()
    for br in el.find_all('br'):
        br.replace_with(NavigableString(LINE_BREAK))
    text = el.get_text()
    if notes:  # SEP and some editions bracket a note's number outside its link: "finite.[<a>6</a>] By"
        text = re.sub(r'\s*\[\s*\]', '', text)
    return clean(text, verse), notes


class Blocks:
    def __init__(self, document_id):
        self.document_id, self.items = document_id, []

    def add(self, kind, section, text, **fields):
        if not text:
            return
        block = {'id': f'{self.document_id}:{len(self.items):05d}', 'document_id': self.document_id,
                 'kind': kind, 'section': list(section), 'text': text}
        block.update({k: v for k, v in fields.items() if v})
        self.items.append(block)


def heading_text(section):
    """Title of a Standard Ebooks section, from its heading, hgroup or header."""
    for child in section.find_all(recursive=False):
        if child.name in ('h2', 'h3', 'h4', 'h5', 'h6'):
            return text_of(child)[0]
        if child.name in ('hgroup', 'header'):
            parts = [text_of(x)[0] for x in child.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p'])
                     if x.name != 'p'
                     or (x.get('epub:type') or '') in ('title', 'subtitle', 'se:bridgehead')]
            return '. '.join(p.rstrip('.') for p in parts if p)
    return None


def snapshot(doc, record):
    """The document's pin in corpus/sources.json, refused when fetch.py recorded another one for the downloaded file."""
    pin = {k: doc[k] for k in ('commit', 'revision', 'edition', 'ebook') if k in doc}
    for k, v in pin.items():
        if k in record and str(record[k]) != str(v):
            raise SystemExit(f'{doc["document_id"]}: {record["file"]} was fetched at {k} {record[k]}, but corpus/sources.json '
                             f'pins {v}. Delete the file and run fetch.py again.')
    return pin


def reading_url(doc, record):
    """The page holding the document's whole text, where a link can scroll to a passage: Standard Ebooks'
    single-page edition and Gutenberg's HTML file rather than their catalogue pages."""
    if doc['source'] == 'standard_ebooks':
        return f'{standard_ebooks_url(doc["repo"])}/text/single-page'
    if doc['source'] == 'gutenberg':
        return f'https://www.gutenberg.org/cache/epub/{doc["ebook"]}/pg{doc["ebook"]}-images.html'
    return record['source_url']


def standard_ebooks(doc, raw, out):
    # Each block records its section's place in the single-page edition, which browsers cannot search by text
    # (an XHTML page ignores text fragments).
    single_page = reading_url(doc, None)
    exclude = SE_EXCLUDE.get(doc['document_id'], {})
    files = json.loads(raw)['files']
    for name, xhtml in files.items():
        if name in exclude:
            continue
        body = BeautifulSoup(xhtml, 'html.parser').body
        if name == 'endnotes.xhtml':  # comes last, and one file can serve several works
            cited = {n for b in out.items for n in b.get('notes', [])}
            for li in body.select('li[id^="note-"]'):
                if li.get('id') not in cited:
                    continue
                for back in li.select('a[epub\\:type~=backlink]'):
                    back.decompose()
                out.add('note', ['Notes'], text_of(li)[0], note_id=li.get('id'))
            continue
        for el in body.find_all(['p', 'tr', 'li', 'blockquote']):
            ancestors = list(el.parents)
            if any(a.get('id') in exclude for a in ancestors):
                continue
            if any(a.name in ('header', 'hgroup', 'footer') for a in ancestors):
                continue
            if el.name != 'tr' and el.find_parent('td'):  # part of a dialogue turn
                continue
            if el.name != 'blockquote' and el.find_parent('blockquote', attrs={'epub:type': re.compile('verse')}):
                continue
            if el.name == 'p' and (el.get('epub:type') or '') in ('title', 'subtitle', 'se:bridgehead', 'z3998:signature'):
                continue
            if el.name == 'li' and el.find_parent('li'):
                continue
            # A known fault, kept so that the blocks match the live index: a <p> inside an <li> is emitted twice, in
            # the item's block and in its own (2,091 such pairs in the corpus).
            section = [t for t in (heading_text(a) for a in reversed(ancestors)
                                   if a.name in ('section', 'article')) if t]
            url = next((f'{single_page}#{a["id"]}' for a in ancestors if a.name in ('section', 'article') and a.get('id')), None)
            if el.name == 'tr':
                cells = el.find_all('td', recursive=False)
                if len(cells) != 2 or 'persona' not in (cells[0].get('epub:type') or ''):
                    continue
                speaker = clean(cells[0].get_text())
                text, notes = text_of(cells[1])
                out.add('turn', section, f'{speaker}: {text}', speaker=speaker, notes=notes, url=url)
            elif el.name == 'blockquote':
                if 'verse' in (el.get('epub:type') or ''):
                    text, notes = text_of(el, verse=True)
                    out.add('verse', section, text, notes=notes, url=url)
            elif el.name == 'li':
                text, notes = text_of(el)
                out.add('li', section, text, notes=notes, url=url)
            else:
                kind = 'quote' if el.find_parent('blockquote') else 'p'
                text, notes = text_of(el)
                out.add(kind, section, text, notes=notes, url=url)
    return {'excluded': sorted(exclude.values()),
            'edition_note': 'Standard Ebooks edition; spelling and hyphenation partly modernized.'}


def matches(el, spec):
    tag, text = spec
    return el.name == tag and clean(el.get_text()).startswith(text)


def gutenberg(doc, raw, out):
    rule = GUTENBERG[doc['document_id']]
    soup = BeautifulSoup(raw, 'html.parser')
    for el in soup.select('#pg-header, #pg-footer' + (', ' + rule['drop'] if 'drop' in rule else '')):
        el.decompose()
    if 'verses' in rule:
        return bible(rule, soup, out)
    levels = rule['headings']
    started, skipping, section = False, False, []
    for el in soup.body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p']):
        if not started:
            started = matches(el, rule['start'])
        if not started:
            continue
        if 'stop' in rule and matches(el, rule['stop']):
            break
        if 'skip' in rule:
            if matches(el, rule['skip'][0]):
                skipping = True
            elif matches(el, rule['skip'][1]):
                skipping = False
        if skipping:
            continue
        if el.name in levels:
            for a in el.select('a'):
                a.decompose()
            depth = levels.index(el.name)
            section = section[:depth] + [re.sub(r'\[\s*\]', '', clean(el.get_text())).strip()]
            continue
        if el.name != 'p':
            continue
        classes = el.get('class') or []
        if 'ph3' in classes:  # Giles's chapter title, set as a paragraph under the heading
            section = section[:1] + [clean(el.get_text())]
            continue
        if 'footnote' in classes:
            text, _ = text_of(el)
            out.add('note', section, text)
            continue
        verse = 'poem' in classes
        text, notes = text_of(el, verse=verse)
        if not text or (rule.get('drop_bracketed') and text.startswith('[') and text.endswith(']')):
            continue
        m = re.match(r'(\d+) \(return\) \[ ?(.*?)\]?$', text)  # older editions' unmarked footnotes
        if m:
            out.add('note', section, m.group(2).strip(), note_id=m.group(1))
            continue
        if (rule.get('rejoin_fragments') and out.items and out.items[-1]['kind'] == 'p'
                and text[0].islower()):
            # A translator's note split this sentence; its two halves become one paragraph.
            out.items[-1]['text'] += ' ' + text
            continue
        out.add('verse' if verse else 'p', section, text, notes=notes)
    return {'excluded': rule['excluded']}


def bible(rule, soup, out):
    """One block per verse, under its chapter. Continuation lines join their verse."""
    verses = []
    for line in soup.body.get_text().splitlines():
        m = re.match(r'(\d{3}):(\d{3}) (.*)', line.strip())
        if m:
            verses.append([int(m.group(1)), int(m.group(2)), m.group(3)])
        elif verses and line.strip():
            verses[-1][2] += ' ' + line.strip()
    for chapter, verse, text in verses:
        text = clean(re.sub(r'\{[^}]*\}', '', text))  # translators' notes
        out.add('p', [f'Chapter {chapter}'], text, ref=f'{rule["verses"]} {chapter}:{verse}')
    return {'excluded': rule['excluded']}


def wikipedia(doc, raw, out):
    html = json.loads(raw)['parse']['text']
    root = BeautifulSoup(html, 'html.parser').find('div', class_='mw-parser-output')
    section = []
    for el in root.find_all(recursive=False):
        classes = el.get('class') or []
        if 'mw-heading' in classes:
            h = el.find(['h2', 'h3', 'h4', 'h5', 'h6'])
            title = text_of(h)[0]
            if h.name == 'h2' and title in WIKIPEDIA_STOP:
                break
            depth = int(h.name[1]) - 2
            section = section[:depth] + [title]
        elif el.name == 'p':
            text, _ = text_of(el)
            out.add('p', section, text)
        elif el.name in ('ul', 'ol') and section:
            for li in el.find_all('li', recursive=False):
                out.add('li', section, text_of(li)[0])
        elif el.name == 'blockquote':
            out.add('quote', section, text_of(el)[0])
    return {'excluded': ['infoboxes, figures, tables, navigation boxes, reference markers, '
                         'and the sections from "See also" onward']}


WIKISOURCE = {
    'ws-berkeley-three-dialogues': {
        'speakers': {'Phil': 'Philonous', 'Philonous': 'Philonous', 'Hyl': 'Hylas', 'Hylas': 'Hylas'},
        'excluded': ["the editor's preface and footnotes, title lines"],
    },
}


def wikisource(doc, raw, out):
    rule = WIKISOURCE[doc['document_id']]
    names = '|'.join(rule['speakers'])
    for title, html in json.loads(raw)['pages'].items():
        section = [title.split('/')[-1]]
        url = f'https://en.wikisource.org/wiki/{urllib.parse.quote(title.replace(" ", "_"))}'  # each part has its page
        root = BeautifulSoup(html, 'html.parser').find('div', class_='mw-parser-output')
        for junk in root.select('div.ws-header, div.ws-noexport, ol.references, div.reflist'):
            junk.decompose()
        pending = None  # a speaker named on a line of their own, before a drop-cap paragraph
        for el in root.find_all('p'):
            text, notes = text_of(el)
            if not text or not re.search('[a-z]', text):  # empty, or a title line in capitals
                continue
            m = re.fullmatch(rf'({names})\.?', text)
            if m:
                pending = m.group(1)
                continue
            m = re.match(rf'({names})\.\s*', text)
            if pending:
                text = f'{pending}. {text}'
                speaker, pending = rule['speakers'][pending], None
            else:
                speaker = rule['speakers'][m.group(1)] if m else None
            out.add('turn' if speaker else 'p', section, text, speaker=speaker, notes=notes, url=url)
    return {'excluded': rule['excluded']}


def sep(doc, raw, out):
    soup = BeautifulSoup(raw, 'html.parser')
    for p in soup.find(id='preamble').find_all('p'):
        out.add('p', [], text_of(p)[0])
    section = []

    def append(formula):
        # A displayed formula (\[ ... \]) or table sits between paragraphs and completes the one before it, so it
        # joins that block on a line of its own. A block of its own would renumber every later block.
        if not formula:
            return
        if not out.items:
            raise SystemExit(f'{doc["document_id"]}: a formula or table before the first block')
        # A known fault, kept so that the blocks match the live index: a formula or table right after a heading joins
        # the last block of the section before (9 formulas in 4 entries).
        out.items[-1]['text'] += '\n' + formula

    for el in soup.find(id='main-text').children:
        if isinstance(el, NavigableString):
            for formula in re.findall(r'\\\[.*?\\\]', clean(str(el))):
                append(formula)
        elif el.name == 'table':
            append(table_formula(el))
        elif el.name in ('h2', 'h3', 'h4'):
            depth = int(el.name[1]) - 2
            section = section[:depth] + [text_of(el)[0]]
        elif el.name == 'p':
            text, notes = text_of(el)
            out.add('p', section, text, notes=notes)
        elif el.name == 'blockquote':
            text, notes = text_of(el)
            out.add('quote', section, text, notes=notes)
        elif el.name in ('ul', 'ol'):
            for li in el.find_all('li', recursive=False):
                tables = [table.extract() for table in li.find_all('table')]
                text, notes = text_of(li)
                out.add('li', section, text, notes=notes)
                for table in tables:
                    append(table_formula(table))
    # One author per line after "Copyright © year by", each perhaps followed by an <email>.
    copyright = soup.find(id='article-copyright')
    for br in copyright.find_all('br'):
        br.replace_with('\n')
    text = re.sub(r'<[^>]*>', '', copyright.get_text())
    names = [clean(n) for n in re.sub(r'^\s*Copyright © \d{4} by', '', text).split('\n') if clean(n)]
    assert names, f'no author in the byline: {text!r}'
    author = names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]
    published = clean(soup.find(id='pubinfo').get_text())
    return {'author': author, 'published': published,
            'excluded': ['notes (on a separate page), bibliography, tools, related entries, tables of long text']}


IEP_AUTHOR = re.compile(r'Authors? (and Article )?Inform\w*', re.I)


def iep(doc, raw, out):
    html = json.loads(raw)['content']['rendered']
    soup = BeautifulSoup(html, 'html.parser')
    # The author's name is the first line after the last "Author Information" (spelled several ways, as a heading
    # or a paragraph), then an email, an institution and a country.
    starts = [m.start() for m in IEP_AUTHOR.finditer(html)]
    byline = html[starts[-1]:] if starts else ''
    lines = [clean(BeautifulSoup(line, 'html.parser').get_text()) for line in re.split(r'<br\s*/?>|</p>|</div>', byline)]
    lines = [line for line in lines if line]
    first = next(filter(None, [IEP_AUTHOR.sub('', lines[0]).strip(' :'), *lines[1:2]]), None) if lines else None
    if not first:
        raise SystemExit(f'{doc["document_id"]}: no author name')
    author = 'Anonymous' if 'anonymous' in first.lower() else first
    section, contents = [], None
    # The outermost blocks: unclosed tags in some entries nest the rest of the entry inside an element.
    kinds = ['h1', 'h2', 'h3', 'h4', 'p', 'blockquote', 'ul', 'ol']
    for el in [el for el in soup.find_all(kinds) if not el.find_parent(kinds)]:
        # A known fault, kept so that the blocks match the live index: the text_of(el) calls here take the note links
        # out before the paragraph branch reads them, so a paragraph keeps "[]" where a note was and loses its notes.
        title = text_of(el)[0] if el.name in ('h2', 'h3', 'h4') else None
        if IEP_AUTHOR.match(text_of(el)[0]):
            break
        if title and re.match(r'(\d+\. )?References and Further Reading$', title):
            section = None  # the bibliography, until the author
        elif section is None or el.name == 'h1' or el is contents:
            continue
        elif title == 'Table of Contents':
            contents = el.find_next('ol')
        elif title:
            # Entries number their sections "1." and subsections "a.", but some tag them a level too low.
            depth = 0 if re.match(r'\d+\.', title) else 1 if re.match(r'[a-h]\.', title) else int(el.name[1]) - 2
            section = section[:depth] + [title]
        elif el.name == 'p':
            text, notes = text_of(el)
            out.add('p', section, text, notes=notes)
        elif el.name == 'blockquote':
            out.add('quote', section, text_of(el)[0])
        elif el.name in ('ul', 'ol'):
            for li in el.find_all('li', recursive=False):
                out.add('li', section, text_of(li)[0])
    return {'author': author, 'excluded': ['the table of contents, the bibliography and the author information']}


WALKERS = {'standard_ebooks': standard_ebooks, 'gutenberg': gutenberg,
           'wikipedia': wikipedia, 'wikisource': wikisource, 'sep': sep, 'iep': iep}


def extract(job):
    """A document's record and blocks, from its downloaded file."""
    doc, record, raw_dir = job
    raw = (raw_dir / record['file']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['raw_sha256'], doc['document_id']
    out = Blocks(doc['document_id'])
    info = WALKERS[doc['source']](doc, raw.decode('utf-8'), out)
    document = {
        **{k: doc[k] for k in ('document_id', 'work_id', 'title', 'source')},
        'author': info.get('author') or doc['author'],
        # tex_macros: the LaTeX macros an SEP entry defines for its formulas (its local.js), for rendering them.
        **{k: doc[k] for k in ('translator', 'tex_macros') if k in doc},
        'language': 'en',
        **({'published': info['published']} if 'published' in info else {}),
        'source_url': record['source_url'],
        'reading_url': reading_url(doc, record),
        'snapshot': snapshot(doc, record),
        'retrieved': record['retrieved'],
        'raw_sha256': record['raw_sha256'],
        **({'edition_note': info['edition_note']} if 'edition_note' in info else {}),
        'excluded': info['excluded'],
        'blocks': len(out.items),
    }
    return document, out.items


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--data', type=Path, default=HERE / 'data')
    args = parser.parse_args()
    sources = json.loads((HERE / 'corpus' / 'sources.json').read_text(encoding='utf-8'))
    raw_dir = args.data / 'raw'
    fetched = json.loads((raw_dir / 'fetched.json').read_text(encoding='utf-8'))
    jobs = [(doc, fetched[doc['document_id']], raw_dir) for doc in sources['documents'] if doc['document_id'] in fetched]
    # Documents are parsed in parallel and written in the order of corpus/sources.json. Text mode writes the platform's line
    # endings, as write_jsonl does.
    with (multiprocessing.Pool() as pool, open(args.data / 'documents.jsonl', 'w', encoding='utf-8') as documents,
          open(args.data / 'blocks.jsonl', 'w', encoding='utf-8') as blocks):
        for document, items in pool.imap(extract, jobs, chunksize=4):
            documents.write(json.dumps(document, ensure_ascii=False) + '\n')
            blocks.writelines(json.dumps(b, ensure_ascii=False) + '\n' for b in items)
            print(f'{document["document_id"]:32} {len(items):5d} blocks')


if __name__ == '__main__':
    main()
