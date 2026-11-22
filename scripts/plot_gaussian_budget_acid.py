#!/usr/bin/env python3
"""Regenerate the existing RE10K -> ACID budget figure with consistent offsets.

The original plotting source/raw table is not present in this checkout. Read
the supplied vector PDF instead: preserve all numerical labels, plotted data,
fonts and layout, changing only the Ours-minus-Merging annotation colors.

Performance improvement is green, degradation is red, and exactly zero is gray.
PSNR/SSIM: positive is improvement. LPIPS: negative is improvement.
There is no tolerance/dead zone for small displayed nonzero offsets.

Dependencies: pypdf; pymupdf for PNG/SVG previews (omit with --pdf-only).

Examples:
    python scripts/plot_gaussian_budget_acid.py
    python scripts/plot_gaussian_budget_acid.py --input /path/to/fig_budget_latency_re10k_acid.pdf
    python scripts/plot_gaussian_budget_acid.py --output-dir outputs/paper_figures --dpi 300
"""

import argparse
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject, FloatObject


ROOT = Path(__file__).resolve().parents[1]
GREEN = '#008000'
RED = '#d62728'
NEUTRAL = '#555555'
METRICS = ('PSNR', 'SSIM', 'LPIPS')
OFFSET = re.compile(r'([+\-\N{MINUS SIGN}]\d+(?:\.\d+)?)(?:\s*dB)?\Z')


@dataclass(frozen=True)
class Annotation:
    block: int
    metric: str
    label: str
    delta: Decimal

    @property
    def color(self):
        improvement = -self.delta if self.metric == 'LPIPS' else self.delta
        return GREEN if improvement > 0 else RED if improvement < 0 else NEUTRAL

    def metadata(self):
        return {'text_block': self.block, 'metric': self.metric, 'label': self.label,
                'ours_minus_merging': str(self.delta), 'color': self.color}


def rgb(hex_color):
    return [FloatObject(int(hex_color[i:i+2], 16) / 255) for i in (1, 3, 5)]


def text_blocks(page):
    """Map decoded Unicode text to PDF BT/ET objects with public callbacks.

    Count BT *after* extraction processes it: BT can flush the preceding object.
    This avoids interpreting embedded font encodings or rebuilding glyphs.
    """
    blocks = []
    active = [-1]
    def after(operator, operands, current_matrix, text_matrix):
        if operator == b'BT':
            active[0] += 1
            blocks.append('')
    def visit(text, current_matrix, text_matrix, font, size):
        if text.strip() and active[0] >= 0:
            blocks[active[0]] += text
    page.extract_text(visitor_operand_after=after, visitor_text=visit)
    stream = ContentStream(page['/Contents'], page.pdf)
    count = sum(operator == b'BT' for operands, operator in stream.operations)
    if count != len(blocks):
        raise ValueError('Text object mapping failed; refusing to alter the figure')
    return [text.strip() for text in blocks]


def annotations(page):
    """Recognize the existing three-panel Matplotlib figure's signed deltas."""
    metric = None
    headers = []
    result = []
    for block, text in enumerate(text_blocks(page)):
        heading = next((name for name in METRICS if text.startswith(name)), None)
        if heading:
            metric = heading
            headers.append(heading)
            continue
        match = OFFSET.fullmatch(text)
        if match:
            if metric is None:
                raise ValueError('Signed text outside a metric panel; ambiguous annotation')
            result.append(Annotation(block, metric, text,
                                     Decimal(match[1].replace('\N{MINUS SIGN}', '-'))))
    if headers != list(METRICS):
        raise ValueError(f'Expected PSNR, SSIM, LPIPS panels in that order; found {headers}')
    counts = {name: sum(item.metric == name for item in result) for name in METRICS}
    if not all(counts.values()) or len(set(counts.values())) != 1:
        raise ValueError(f'Incomplete or ambiguous performance offsets: {counts}')
    return result


def recolor_page(page, changes):
    """Insert local color wrappers without reserializing original coordinates.

    Matplotlib puts BT/ET on separate lines. Confirm that these line tokens
    exactly match the parsed text-object sequence before changing any bytes.
    Keeping the original bytes avoids PDF float-serialization rounding.
    """
    original = page.get_contents().get_data()
    tokens = list(re.finditer(rb'(?m)^[ \t]*(BT|ET)[ \t]*\r?$', original))
    count = sum(op == b'BT' for operands, op in ContentStream(page['/Contents'], page.pdf).operations)
    if [match[1] for match in tokens] != [b'BT', b'ET'] * count:
        raise ValueError('Expected standalone Matplotlib text objects; refusing ambiguous PDF editing')
    selected = {item.block: item for item in changes}
    pieces = []
    block = -1
    cursor = 0
    for match in tokens:
        pieces.append(original[cursor:match.start()])
        if match[1] == b'BT':
            block += 1
            if block in selected:
                color = ' '.join(format(float(value), '.12g') for value in rgb(selected[block].color))
                pieces.append(f'q\n{color} rg\n'.encode('ascii'))
        pieces.append(match[0])
        if match[1] == b'ET' and block in selected:
            pieces.append(b'\nQ')
        cursor = match.end()
    pieces.append(original[cursor:])
    stream = DecodedStreamObject()
    stream.set_data(b''.join(pieces))
    page.replace_contents(stream)


def rebuild(source, output_dir, dpi=300, pdf_only=False):
    source, output_dir = source.expanduser().resolve(), output_dir.expanduser().resolve()
    destination = output_dir / 'fig_budget_latency_re10k_acid.pdf'
    if destination == source:
        raise ValueError('Use a separate output directory to preserve the original PDF')
    if dpi <= 0:
        raise ValueError('DPI must be positive')
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError('Expected the single-page RE10K -> ACID performance figure')
    original_text = reader.pages[0].extract_text()
    changes = annotations(reader.pages[0])
    original_operations = list(ContentStream(reader.pages[0]['/Contents'], reader).operations)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[0])
    recolor_page(page, changes)
    # Strip inserted local color wrappers and verify every original operation,
    # including numerical data/coordinates and font commands, is unchanged.
    rewritten = ContentStream(page['/Contents'], writer).operations
    stripped, block, index = [], -1, 0
    chosen = {item.block for item in changes}
    while index < len(rewritten):
        operands, operator = rewritten[index]
        if (operator == b'q' and index + 2 < len(rewritten)
                and rewritten[index+1][1] == b'rg' and rewritten[index+2][1] == b'BT'
                and block + 1 in chosen):
            index += 2
            operands, operator = rewritten[index]
        if operator == b'BT':
            block += 1
        stripped.append((operands, operator))
        index += 1
        if operator == b'ET' and block in chosen:
            if rewritten[index][1] != b'Q':
                raise AssertionError('Missing local graphics-state restoration')
            index += 1
    if stripped != original_operations:
        raise AssertionError('Unexpected changes outside the annotation color wrappers')
    writer.add_metadata({key: value for key, value in (reader.metadata or {}).items()
                         if isinstance(value, str)})
    writer.add_metadata({'/Subject': 'RE10K to ACID; performance offset colors: improvement green, degradation red; numerical data unchanged'})
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.pdf')
    with temporary.open('wb') as handle:
        writer.write(handle)
    check = PdfReader(temporary)
    if check.pages[0].extract_text() != original_text or check.pages[0].mediabox != reader.pages[0].mediabox:
        raise AssertionError('PDF text or page dimensions changed')
    temporary.replace(destination)
    outputs = [destination]
    if not pdf_only:
        import pymupdf
        with pymupdf.open(destination) as document:
            if document.is_repaired:
                raise AssertionError('Output PDF required structural repair')
            png = destination.with_suffix('.png')
            document[0].get_pixmap(dpi=dpi).save(png)
            svg = destination.with_suffix('.svg')
            svg.write_text(document[0].get_svg_image(text_as_path=True), encoding='utf-8')
        outputs.extend([png, svg])
    metadata = output_dir / 'figure_metadata.json'
    metadata.write_text(json.dumps({
        'source_pdf': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'evaluation': 'RE10K -> ACID', 'comparison': 'Ours minus Merging',
        'color_rule': 'performance improvement green; degradation red; exactly zero gray; no nonzero dead zone',
        'higher_is_better': {'PSNR': True, 'SSIM': True, 'LPIPS': False},
        'green': GREEN, 'red': RED, 'neutral': NEUTRAL,
        'numerical_text_unchanged': True, 'original_drawing_operations_preserved': True,
        'annotations': [item.metadata() for item in changes],
        'outputs': list(map(str, outputs)),
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    return outputs, changes


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument('--input', type=Path, default=Path.home() / 'Downloads/fig_budget_latency_re10k_acid.pdf')
    cli.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/gaussian_control/figures/re10k_to_acid')
    cli.add_argument('--dpi', type=int, default=300)
    cli.add_argument('--pdf-only', action='store_true')
    args = cli.parse_args()
    outputs, changes = rebuild(args.input, args.output_dir, args.dpi, args.pdf_only)
    for item in changes:
        print(f'{item.metric:5s} {item.label:10s} -> {item.color}')
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
