"""Read shared Notion pages and cache them as normalized note records."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_API_VERSION = "2026-03-11"
MIN_REQUEST_INTERVAL_SECONDS = 0.35


class NotionClient:
    """Small REST client using only Python's standard library."""

    def __init__(self, token: str):
        self.token = token
        self.last_request_at = 0.0

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{NOTION_API_BASE}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"

        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Notion-Version": NOTION_API_VERSION,
                "Content-Type": "application/json",
            },
        )

        for attempt in range(4):
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
            self.last_request_at = time.monotonic()

            try:
                with urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < 3:
                    try:
                        retry_after = float(exc.headers.get("Retry-After", "1"))
                    except (TypeError, ValueError):
                        retry_after = 1.0
                    time.sleep(max(retry_after, 0.1))
                    continue
                try:
                    message = json.loads(error_body).get("message", error_body)
                except json.JSONDecodeError:
                    message = error_body
                raise RuntimeError(f"Notion API returned HTTP {exc.code}: {message}") from exc
            except URLError as exc:
                raise RuntimeError(f"Could not reach the Notion API: {exc.reason}") from exc

        raise RuntimeError("Notion API kept rate-limiting the sync. Try again shortly.")

    def list_shared_pages(self) -> list[dict[str, Any]]:
        """Search all pages shared with this integration, following every cursor."""

        pages = []
        payload: dict[str, Any] = {
            "page_size": 100,
            "filter": {"property": "object", "value": "page"},
        }
        while True:
            response = self.request("POST", "/search", payload=payload)
            pages.extend(response.get("results", []))
            if not response.get("has_more") or not response.get("next_cursor"):
                return pages
            payload["start_cursor"] = response["next_cursor"]

    def list_block_children(self, block_id: str) -> list[dict[str, Any]]:
        """Read every child block, following the endpoint's pagination cursor."""

        blocks = []
        params: dict[str, Any] = {"page_size": 100}
        while True:
            response = self.request(
                "GET",
                f"/blocks/{block_id}/children",
                params=params,
            )
            blocks.extend(response.get("results", []))
            if not response.get("has_more") or not response.get("next_cursor"):
                return blocks
            params["start_cursor"] = response["next_cursor"]

    def page_text(self, page_id: str, depth: int = 0) -> str:
        """Flatten text blocks into readable text, including nested block groups."""

        if depth > 12:
            return ""

        lines = []
        for block in self.list_block_children(page_id):
            block_type = block.get("type", "")
            content = block.get(block_type, {})
            rich_text = content.get("rich_text", [])
            text = "".join(item.get("plain_text", "") for item in rich_text).strip()

            if block_type.startswith("heading_") and text:
                text = f"{'#' * (int(block_type[-1]) + 1)} {text}"
            elif block_type == "bulleted_list_item" and text:
                text = f"- {text}"
            elif block_type == "numbered_list_item" and text:
                text = f"1. {text}"
            elif block_type == "to_do" and text:
                checkbox = "x" if content.get("checked") else " "
                text = f"[{checkbox}] {text}"
            elif block_type == "child_page":
                text = content.get("title", text)
            elif block_type == "child_database":
                text = content.get("title", text)
            elif block_type == "table_row":
                cells = content.get("cells", [])
                text = " | ".join(
                    "".join(item.get("plain_text", "") for item in cell)
                    for cell in cells
                )
            elif block_type == "equation":
                text = content.get("expression", text)
            elif block_type in {"bookmark", "embed", "link_preview"}:
                caption = "".join(
                    item.get("plain_text", "") for item in content.get("caption", [])
                )
                text = " ".join(part for part in (caption, content.get("url", "")) if part)
            elif not text:
                caption = content.get("caption", [])
                text = "".join(item.get("plain_text", "") for item in caption).strip()

            if text:
                lines.append(text)

            # Child pages are also returned by the search endpoint; don't duplicate their body here.
            if block.get("has_children") and block_type != "child_page":
                child_text = self.page_text(block["id"], depth + 1)
                if child_text:
                    lines.append(child_text)

        return "\n".join(lines)


def page_title(page: dict[str, Any]) -> str:
    """Find the title property without assuming the page's title column name."""

    for prop in page.get("properties", {}).values():
        if prop.get("type") == "title":
            return "".join(item.get("plain_text", "") for item in prop.get("title", [])).strip()
    return "Untitled Notion page"


def sync_notion_notes(cache_path: Path) -> int:
    """Fetch pages shared with the integration and atomically replace the local cache."""

    token = os.environ.get("NOTION_API_KEY")
    if not token:
        raise RuntimeError("NOTION_API_KEY is not set. Export your Notion integration token in the terminal.")

    client = NotionClient(token)
    pages = client.list_shared_pages()
    notes = []
    seen_page_ids = set()
    for page in pages:
        page_id = page.get("id")
        if not page_id or page_id in seen_page_ids:
            continue
        seen_page_ids.add(page_id)
        title = page_title(page)
        body = client.page_text(page_id)
        text = "\n\n".join(part for part in (title, body) if part).strip()
        if not text:
            continue

        notes.append(
            {
                "source": f"notion:{page_id}",
                "text": text,
                "metadata": {
                    "kind": "notion",
                    "notion_id": page_id,
                    "page_title": title,
                    "updated_at": page.get("last_edited_time", ""),
                    "url": page.get("url", ""),
                },
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix="notion-cache-",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            temporary_path = Path(cache_file.name)
            json.dump(notes, cache_file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()

    return len(notes)
