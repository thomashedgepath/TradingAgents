"""Chunker must split vendor payloads on their natural boundaries and terminate.

The corpus stores rendered markdown, so retrieval quality depends on chunks
landing on article/section edges rather than mid-sentence, and on the splitter
never looping on pathological input.
"""

import pytest

from tradingagents.knowledge.chunking import MAX_CHARS, chunk_text


@pytest.mark.unit
def test_empty_input_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


@pytest.mark.unit
def test_short_document_survives_as_one_chunk():
    # Below the merge floor, but admission is not gated on it — dropping a
    # short macro reading would silently lose data.
    text = "CPI rose 0.2% in June."
    assert chunk_text(text) == [text]


@pytest.mark.unit
def test_splits_on_markdown_headings():
    text = "\n".join(
        f"## Article {i}\n\n{'body sentence. ' * 40}" for i in range(4)
    )
    chunks = chunk_text(text)
    assert len(chunks) >= 4
    # Each article's heading should start a chunk rather than be buried.
    assert sum(c.startswith("## Article") for c in chunks) >= 4


@pytest.mark.unit
def test_no_chunk_exceeds_the_ceiling_on_unbroken_text():
    chunks = chunk_text("x" * (MAX_CHARS * 5))
    assert chunks
    assert all(len(c) <= MAX_CHARS for c in chunks)


@pytest.mark.unit
def test_hard_split_terminates_without_losing_content():
    # No sentence or newline boundaries anywhere: the blunt-cut path.
    body = "abcdefghij" * (MAX_CHARS // 2)
    chunks = chunk_text(body)
    assert "".join(chunks) == body


@pytest.mark.unit
def test_runt_blocks_merge_instead_of_becoming_their_own_chunks():
    text = "\n\n".join(["short one", "short two", "short three", "short four"])
    chunks = chunk_text(text)
    assert len(chunks) == 1


@pytest.mark.unit
def test_paragraph_fallback_when_no_structural_markers():
    text = "\n\n".join(["para body. " * 60, "other body. " * 60])
    chunks = chunk_text(text)
    assert len(chunks) >= 2
