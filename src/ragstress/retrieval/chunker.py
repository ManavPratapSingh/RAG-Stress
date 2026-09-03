"""Sentence-boundary chunker (SPEC §5.1, FR-RET-1)."""

import re

from pydantic import BaseModel


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    sentence_span: tuple[int, int]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences cleanly on punctuation boundaries."""
    # Split on terminal punctuation followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def chunk_passage(
    doc_id: str,
    title: str,
    text: str,
    max_words: int = 200,
) -> list[Chunk]:
    """Split a passage into sentence-aligned chunks without cutting mid-sentence.

    Default max_words=200 preserves most 60-180 word passages as single chunks.
    """
    sentences = split_sentences(text)
    if not sentences:
        # ponytail: fallback if passage text has no sentence terminals
        return [Chunk(chunk_id=f"{doc_id}#c0", doc_id=doc_id, text=text, sentence_span=(0, 0))]

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_words = 0
    start_idx = 0

    for i, sent in enumerate(sentences):
        sent_words = len(sent.split())
        if current_sentences and (current_words + sent_words > max_words):
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#c{len(chunks)}",
                    doc_id=doc_id,
                    text=chunk_text,
                    sentence_span=(start_idx, i - 1),
                )
            )
            current_sentences = [sent]
            current_words = sent_words
            start_idx = i
        else:
            current_sentences.append(sent)
            current_words += sent_words

    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#c{len(chunks)}",
                doc_id=doc_id,
                text=chunk_text,
                sentence_span=(start_idx, len(sentences) - 1),
            )
        )

    return chunks
