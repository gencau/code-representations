from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tree_sitter_languages import get_parser
from transformers import AutoTokenizer


# -----------------------------
# Tokenizer interface + helpers
# -----------------------------

class TokenizerLike:
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: Sequence[int]) -> str:
        raise NotImplementedError


def chunk_by_tokens(
    text: str,
    tokenizer: TokenizerLike,
    chunk_tokens: int = 512,
    overlap_tokens: int = 20,
) -> List[str]:
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be < chunk_tokens")

    token_ids = tokenizer.encode(text)
    if not token_ids:
        return []

    chunks: List[str] = []
    step = chunk_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        end = min(start + chunk_tokens, len(token_ids))
        chunks.append(tokenizer.decode(token_ids[start:end]))
        if end == len(token_ids):
            break
    return chunks


# -----------------------------
# Tree-sitter method extraction
# -----------------------------

@dataclass(frozen=True)
class MethodUnit:
    language: str
    file_path: str
    node_type: str
    text: str
    start_byte: int
    end_byte: int


@dataclass(frozen=True)
class PackedChunk:
    language: str
    file_path: str
    chunk_index: int
    text: str
    # for traceability
    unit_spans: List[Tuple[str, int, int]]  # (node_type, start_byte, end_byte)


METHOD_NODE_TYPES: Dict[str, Tuple[str, ...]] = {
    "python": ("function_definition",),
    "java": ("method_declaration", "constructor_declaration"),
    "kotlin": ("function_declaration",),
}


def _iter_nodes_dfs(root: Any) -> Iterable[Any]:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        for child in reversed(getattr(node, "children", []) or []):
            stack.append(child)


def extract_method_units(code: str, language: str, file_path: str) -> List[MethodUnit]:
    if language not in METHOD_NODE_TYPES:
        raise ValueError(f"Unsupported language: {language}")

    parser = get_parser(language)
    code_bytes = code.encode("utf-8", errors="replace")
    tree = parser.parse(code_bytes)
    root = tree.root_node

    units: List[MethodUnit] = []
    wanted = set(METHOD_NODE_TYPES[language])

    for node in _iter_nodes_dfs(root):
        if node.type in wanted and node.end_byte > node.start_byte:
            txt = code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace").strip()
            if txt:
                units.append(
                    MethodUnit(
                        language=language,
                        file_path=file_path,
                        node_type=node.type,
                        text=txt,
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                    )
                )

    # Sort by source order
    units.sort(key=lambda u: (u.start_byte, u.end_byte))
    return units


# -----------------------------
# Packing to fill token budget
# -----------------------------

def pack_units_to_token_budget(
    units: List[MethodUnit],
    tokenizer: TokenizerLike,
    chunk_tokens: int = 512,
    overlap_tokens: int = 20,
    unit_separator: str = "\n",
    add_unit_headers: bool = False,
) -> List[PackedChunk]:

    if not units:
        return []

    # Reserve budget so we can prepend overlap without truncation
    pack_budget = chunk_tokens - overlap_tokens
    if pack_budget <= 0:
        raise ValueError("chunk_tokens must be > overlap_tokens")

    language = units[0].language
    file_path = units[0].file_path

    rendered_units: List[Tuple[MethodUnit, str, int]] = []
    sep_ids = tokenizer.encode(unit_separator)
    sep_tokens = len(sep_ids)

    for u in units:
        header = ""
        if add_unit_headers:
            header = f"# {u.node_type} {u.file_path}:{u.start_byte}-{u.end_byte}\n"
        block = header + u.text
        n_tokens = len(tokenizer.encode(block))
        rendered_units.append((u, block, n_tokens))

    packed_texts: List[Tuple[str, List[Tuple[str, int, int]]]] = []
    current_parts: List[str] = []
    current_spans: List[Tuple[str, int, int]] = []
    current_tokens = 0

    def flush_current():
        nonlocal current_parts, current_spans, current_tokens
        if current_parts:
            packed_texts.append(("".join(current_parts), list(current_spans)))
        current_parts = []
        current_spans = []
        current_tokens = 0

    for u, block, n_tokens in rendered_units:
        # Oversized unit: chunk internally (full 512/20 windowing)
        if n_tokens > chunk_tokens:
            flush_current()
            windows = chunk_by_tokens(block, tokenizer, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
            for w in windows:
                packed_texts.append((w, [(u.node_type, u.start_byte, u.end_byte)]))
            continue

        # Check budget using pack_budget (not chunk_tokens)
        extra = (sep_tokens if current_parts else 0) + n_tokens
        if current_tokens + extra > pack_budget:
            flush_current()

        if current_parts:
            current_parts.append(unit_separator)
            current_tokens += sep_tokens

        current_parts.append(block)
        current_tokens += n_tokens
        current_spans.append((u.node_type, u.start_byte, u.end_byte))

    flush_current()

    # Now apply overlap safely without truncating meaningful tail content
    packed_chunks: List[PackedChunk] = []
    prev_tail: List[int] = []

    for idx, (txt, spans) in enumerate(packed_texts):
        ids = tokenizer.encode(txt)

        if prev_tail:
            # Prepend overlap and keep full chunk (should fit by construction)
            ids = prev_tail + ids
            # Safety clip (should rarely trigger)
            ids = ids[:chunk_tokens]
            txt = tokenizer.decode(ids)

        full_ids = tokenizer.encode(txt)
        prev_tail = full_ids[-overlap_tokens:] if overlap_tokens > 0 else []

        packed_chunks.append(
            PackedChunk(
                language=language,
                file_path=file_path,
                chunk_index=idx,
                text=txt,
                unit_spans=spans,
            )
        )

    return packed_chunks

# -----------------------------
# One-stop helper
# -----------------------------

def make_packed_method_chunks(
    *,
    code: str,
    language: str,
    file_path: str,
    tokenizer: TokenizerLike,
    chunk_tokens: int = 512,
    overlap_tokens: int = 20,
) -> List[PackedChunk]:
    units = extract_method_units(code, language, file_path)
    if not units:
        # Fallback: chunk whole file
        windows = chunk_by_tokens(code, tokenizer, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens)
        return [
            PackedChunk(language=language, file_path=file_path, chunk_index=i, text=w, unit_spans=[("file", 0, len(code.encode("utf-8", "replace")))])
            for i, w in enumerate(windows)
        ]

    return pack_units_to_token_budget(
        units,
        tokenizer,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        add_unit_headers=False,
    )



# -----------------------------
# Example tokenizer wiring (HuggingFace)
# -----------------------------

def make_hf_tokenizer(model_name_or_path: str) -> TokenizerLike:
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    class HFWrapper(TokenizerLike):
        def encode(self, text: str) -> List[int]:
            return tok.encode(text, add_special_tokens=False)

        def decode(self, ids: Sequence[int]) -> str:
            return tok.decode(ids, skip_special_tokens=True)

    return HFWrapper()


# -----------------------------
# Usage example
# -----------------------------

if __name__ == "__main__":
    # Replace with the actual tokenizer
    # If you're using a local HF checkpoint, pass its path.
    tokenizer = make_hf_tokenizer("openai/gpt-oss-20b")

    sample_py = """
import os

class A:
    def f(self, x):
        return x + 1

def g(y):
    return y * 2
"""

    chunks = make_packed_method_chunks(
        code=sample_py,
        language="python",
        file_path="example.py",
        tokenizer=tokenizer,
        chunk_tokens=512,
        overlap_tokens=20,
    )

    print(f"Produced {len(chunks)} chunks")
    for c in chunks:
        print("---")
        print(c.chunk_index, c.file_path, c.unit_spans)
        print(c.text[:200])
