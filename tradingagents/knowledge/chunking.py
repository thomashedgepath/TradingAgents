"""Split vendor payloads into retrievable chunks.

Vendor payloads are rendered markdown, not prose: news calls come back as a run
of per-article blocks, fundamentals as sectioned tables. Splitting on those
natural boundaries first means a news chunk usually *is* one article and a
fundamentals chunk *is* one section, which is the granularity a reader wants
back. Only oversized blocks get sliced further, and a fixed-size fallback keeps
pathological input (one enormous unbroken block) bounded.
"""

from __future__ import annotations

import re

# Target and ceiling in characters. ~1200 chars is roughly 300 tokens, small
# enough that several hits fit an agent prompt and large enough to keep an
# article's lede and body together.
TARGET_CHARS = 1200
MAX_CHARS = 2000
MIN_CHARS = 80

# A block boundary is a markdown heading, a horizontal rule, or a blank line
# followed by a bolded/### lead-in — the shapes the vendor formatters emit
# between articles and between report sections.
_BOUNDARY_RE = re.compile(
    r"\n(?=#{1,6}\s)"           # markdown heading
    r"|\n(?=-{3,}\s*\n)"        # horizontal rule
    r"|\n(?=\*{3,}\s*\n)"
    r"|\n\s*\n(?=\*\*)"         # blank line then a bolded lead-in
    r"|\n\s*\n(?=###)",
)


def _split_blocks(text: str) -> list[str]:
    """Split on structural boundaries, falling back to paragraphs."""
    blocks = [b.strip() for b in _BOUNDARY_RE.split(text) if b and b.strip()]
    if len(blocks) > 1:
        return blocks
    # No structural markers: fall back to blank-line paragraphs.
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _hard_split(block: str, limit: int) -> list[str]:
    """Slice an oversized block on sentence/line boundaries, then bluntly.

    The loop always advances by at least one character-window, so a block with
    no usable boundary still terminates instead of re-emitting the same prefix.
    """
    out: list[str] = []
    remaining = block
    while len(remaining) > limit:
        window = remaining[:limit]
        cut = max(window.rfind(". "), window.rfind("\n"))
        if cut < limit // 2:
            cut = limit  # no sensible boundary in the back half — cut flat
        else:
            cut += 1
        out.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()
    if remaining.strip():
        out.append(remaining.strip())
    return out


def chunk_text(
    text: str,
    *,
    target: int = TARGET_CHARS,
    maximum: int = MAX_CHARS,
    minimum: int = MIN_CHARS,
) -> list[str]:
    """Split ``text`` into chunks, merging runt blocks and slicing huge ones.

    Returns ``[]`` for empty or whitespace-only input. A single short document
    (a one-line macro reading, say) comes back as one chunk even if it is below
    ``minimum`` — the floor governs *merging*, not admission, so no content is
    ever dropped.
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    buffer = ""

    for block in _split_blocks(text):
        pieces = _hard_split(block, maximum) if len(block) > maximum else [block]
        for piece in pieces:
            if not buffer:
                buffer = piece
            elif len(buffer) < minimum or len(buffer) + len(piece) + 2 <= target:
                # Merge while the accumulated text is still runt-sized or the
                # combination stays under target.
                buffer = f"{buffer}\n\n{piece}"
            else:
                chunks.append(buffer)
                buffer = piece
            if len(buffer) >= target:
                chunks.append(buffer)
                buffer = ""

    if buffer.strip():
        chunks.append(buffer)

    return [c.strip() for c in chunks if c.strip()]
