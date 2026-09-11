#!/usr/bin/env python3
"""Validate the tracked Markdown source for the loqculate GitHub Wiki."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

_REQUIRED_PAGES = ("Home", "_Sidebar", "_Footer")
_PORTABLE_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')
_WIKI_LINK = re.compile(r"\[\[([^]\n]+)\]\]")
_MARKDOWN_LINK = re.compile(r"(?<!!)\[([^]\n]+)\]\(([^)\n]+)\)")
_LINK_TITLE = re.compile(r"""\s+(?:"[^"]*"|'[^']*'|\([^)]*\))\s*$""")
_INLINE_CODE = re.compile(r"`+[^`\n]*`+")
_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})")
# Substrings that must not appear on wiki pages.
_LOCAL_ONLY = (".wiki.git", "plan/")


def _blank(text: str) -> str:
    """Replace non-newline characters with spaces so match positions survive."""

    return "".join("\n" if char == "\n" else " " for char in text)


def _without_code(text: str) -> str:
    """Mask fenced and inline code, where link-looking text is not a link."""

    lines = text.splitlines(keepends=True)
    masked: list[str] = []
    fence_char: str | None = None
    fence_length = 0

    for line in lines:
        match = _FENCE.match(line)
        if fence_char is None:
            if match is not None:
                marker = match.group(2)
                fence_char = marker[0]
                fence_length = len(marker)
                masked.append(_blank(line))
            else:
                masked.append(line)
            continue

        masked.append(_blank(line))
        stripped = line.lstrip(" ")
        if re.match(rf"^{re.escape(fence_char)}{{{fence_length},}}[ \t]*\r?$", stripped):
            fence_char = None
            fence_length = 0

    return _INLINE_CODE.sub(lambda match: _blank(match.group(0)), "".join(masked))


def _page_slug(value: str) -> str:
    """Return the GitHub Wiki-equivalent key for a page name."""

    decoded = unquote(value).strip().casefold()
    return re.sub(r"[\s_-]+", "-", decoded)


def _line_number(text: str, position: int) -> int:
    return text.count("\n", 0, position) + 1


def _page_target(raw_target: str) -> tuple[str, bool]:
    """Return a page target and whether it uses a repository-style .md suffix."""

    target = unquote(raw_target.strip())
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = _LINK_TITLE.sub("", target).strip()

    target = target.split("#", 1)[0].strip()
    target = target.removeprefix("./")
    has_md_suffix = target.casefold().endswith(".md")
    if has_md_suffix:
        target = target[:-3]
    return target, has_md_suffix


def _is_external(target: str) -> bool:
    return bool(
        re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target, flags=re.IGNORECASE)
    ) or target.startswith("#")


def validate(wiki_dir: Path) -> list[str]:
    """Return validation errors for ``wiki_dir``; an empty list means valid."""

    errors: list[str] = []
    if not wiki_dir.is_dir():
        return [f"wiki directory not found: {wiki_dir}"]

    entries = sorted(wiki_dir.iterdir(), key=lambda path: path.name.casefold())
    pages: list[Path] = []
    aliases: dict[str, Path] = {}

    for entry in entries:
        relative = f"{wiki_dir}/{entry.name}"
        if entry.is_symlink() or not entry.is_file():
            errors.append(f"wiki entries must be regular top-level files: {relative}")
            continue
        if not entry.name.endswith(".md"):
            errors.append(f"wiki source currently permits Markdown pages only: {relative}")
            continue
        if _PORTABLE_FILENAME_CHARS.search(entry.name):
            errors.append(f"wiki filename contains a non-portable character: {relative}")
        if entry.stat().st_size == 0:
            errors.append(f"wiki page is empty: {relative}")

        pages.append(entry)
        page_name = entry.name[:-3]
        for alias in (page_name.casefold(), _page_slug(page_name)):
            previous = aliases.get(alias)
            if previous is not None and previous != entry:
                errors.append(
                    f"case-insensitive or GitHub Wiki slug collision: {previous} and {relative}"
                )
            else:
                aliases[alias] = entry

    for required in _REQUIRED_PAGES:
        if required.casefold() not in aliases:
            errors.append(f"required wiki page is missing: {wiki_dir}/{required}.md")

    for page in pages:
        try:
            text = page.read_text(encoding="utf-8")
        except OSError as error:
            errors.append(f"cannot read {page}: {error}")
            continue

        lowered = text.casefold()
        for needle in _LOCAL_ONLY:
            start = 0
            token = needle.casefold()
            while True:
                idx = lowered.find(token, start)
                if idx < 0:
                    break
                line = _line_number(text, idx)
                errors.append(
                    f"{page}:{line} names local-only development text ({needle})"
                )
                start = idx + len(token)

        visible_text = _without_code(text)
        for match in _WIKI_LINK.finditer(visible_text):
            specification = match.group(1).strip()
            page_target = specification.split("|", 1)[0].strip()
            if not page_target or page_target.startswith("#"):
                continue

            target, has_md_suffix = _page_target(page_target)
            line = _line_number(text, match.start())
            if has_md_suffix:
                errors.append(f"{page}:{line} uses a repository-style .md wiki link: {page_target}")
            if target and not _is_external(target) and _page_slug(target) not in aliases:
                errors.append(f"{page}:{line} links to a missing wiki page: {page_target}")

        for match in _MARKDOWN_LINK.finditer(visible_text):
            raw_target = match.group(2).strip()
            if _is_external(raw_target):
                continue

            target, has_md_suffix = _page_target(raw_target)
            line = _line_number(text, match.start())
            if has_md_suffix:
                errors.append(f"{page}:{line} uses a repository-style .md wiki link: {raw_target}")
            if target and _page_slug(target) not in aliases:
                errors.append(f"{page}:{line} links to a missing wiki page: {raw_target}")

    return errors


def main() -> int:
    """Validate the requested wiki directory and return its process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wiki_dir", nargs="?", type=Path, default=Path("wiki"))
    args = parser.parse_args()

    errors = validate(args.wiki_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    page_count = sum(
        1 for path in args.wiki_dir.iterdir() if path.is_file() and path.name.endswith(".md")
    )
    print(f"validated {page_count} wiki pages in {args.wiki_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
