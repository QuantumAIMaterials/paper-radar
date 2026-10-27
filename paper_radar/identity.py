from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paper_radar.models import Paper


def normalize_title(title: str) -> str:
    # Keep scientific letters such as alpha and beta distinct.
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", title).casefold())


def normalize_doi(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().lower().removeprefix("doi:").strip()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/",
                   "http://dx.doi.org/"):
        value = value.removeprefix(prefix)
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else ""


def identity_keys(
    paper_id: str, title: str, doi: object, aliases: object = (),
) -> set[str]:
    keys = {f"id:{paper_id}"} if paper_id else set()
    if title_key := normalize_title(title):
        keys.add(f"title:{title_key}")
    if doi_key := normalize_doi(doi):
        keys.add(f"doi:{doi_key}")
    # Legacy Crossref state records did not store a separate DOI field.
    if paper_id.startswith("crossref:") and (
        doi_key := normalize_doi(paper_id.removeprefix("crossref:"))
    ):
        keys.add(f"doi:{doi_key}")
    if isinstance(aliases, (list, tuple)):
        keys.update(key for key in aliases if isinstance(key, str) and key)
    return keys


def paper_keys(paper: Paper) -> set[str]:
    return identity_keys(paper.paper_id, paper.title, paper.doi, paper.aliases)
