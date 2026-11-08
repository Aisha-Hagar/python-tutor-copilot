import logging
import re

logger = logging.getLogger(__name__)


def extract_available_pages(context: str) -> list[int]:
    """Return sorted unique page numbers found in the retrieved context."""
    return sorted({int(m) for m in re.findall(r"\d+", context)})


def extract_cited_pages(draft: str) -> list[int]:
    """Return page numbers cited in the draft, in order of appearance."""
    return [int(m) for m in re.findall(r"\[page (\d+)\]", draft)]


def find_invalid_citations(draft: str, available_pages: list[int]) -> list[int]:
    """Return cited pages that are not in the available set"""
    return [p for p in extract_cited_pages(draft) if p not in available_pages]


def strip_invalid_citations(draft: str, available_pages: list[int]) -> str:
    """Remove [page N] where N is not in available pages."""
    def repl(match):
        page = int(match.group(1))
        return match.group(0) if page in available_pages else ""
    return re.sub(r"\[page (\d+)\]", repl, draft).replace("  ", " ").strip()


def format_context_with_ids(hits) -> tuple[str, dict[str, int]]:
    """
    Convert Qdrant hits into a context string with short IDs like [S1].
    :return: (formatted_context, id_to_page_map)
    """
    blocks = []
    id_map: dict[str, int] = {}
    for i, h in enumerate(hits, start=1):
        sid = f"S{i}"
        payload = h.payload or {}
        page = payload.get("page", 0)
        id_map[sid] = page
        blocks.append(f"[{sid}, page {page}]: {payload.get('text', '')}")
    return "\n\n".join(blocks), id_map


def replace_ids_with_pages(text: str, id_map: dict[str, int]) -> str:
    """Convert [SID] back to [page N] for display."""
    for sid, page in id_map.items():
        text = text.replace(f"[{sid}]", f"[page {page}]")
    return text


def resolve_source_ids(draft: str, source_map: dict[str, int]) -> str:
    """Replace [S1] with [page 139] using the source map."""
    def repl(match):
        sid = match.group(1).upper()
        if sid in source_map:
            return f"[page {source_map[sid]}]"
        return match.group(0)   # leave unknown IDs untouched
    return re.sub(r"\[(S\d+)\]", repl, draft)


def extract_id_page_map(context: str) -> dict[str, int]:
    """Parse '[S1, page 139]' markers into {'S1': 139, ...}."""
    return {sid: int(page) for sid, page in
            re.findall(r"\[(S\d+),\s*page\s*(\d+)\]", context)}