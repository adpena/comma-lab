from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError

import tools.audit_public_publish_links as audit_mod
from tools.audit_public_publish_links import (
    audit_public_publish_links,
    extract_public_links,
    live_link_violations,
    unauthenticated_status,
)


def test_extract_public_links_skips_placeholder_lines(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "\n".join(
            [
                "repo: https://github.com/adpena/tac",
                "future: ${CLOUDFLARE_PAGES_URL}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    links = extract_public_links([tmp_path], base_root=tmp_path)

    assert [link.url for link in links] == ["https://github.com/adpena/tac"]
    assert links[0].path == "README.md"


def test_static_audit_flags_private_comma_lab_links(tmp_path: Path) -> None:
    private_url = "https://github.com/adpena/" + "comma-lab/tree/main/docs"
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "release.md").write_text(
        f"private repo: {private_url}\n",
        encoding="utf-8",
    )

    payload = audit_public_publish_links([tmp_path], base_root=tmp_path)

    assert payload["link_count"] == 1
    assert payload["violation_count"] == 1
    assert payload["violations"][0]["kind"] == "private_link"


def test_live_audit_uses_unauthenticated_checker_and_deduplicates(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "a: https://example.invalid/private\n"
        "b: https://example.invalid/private\n"
        "c: https://example.invalid/public\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    def checker(url: str) -> tuple[int | None, str]:
        calls.append(url)
        if url.endswith("/private"):
            return 403, "HEAD"
        return 200, "HEAD"

    payload = audit_public_publish_links([tmp_path], base_root=tmp_path, live=True, checker=checker)

    assert calls == ["https://example.invalid/private", "https://example.invalid/public"]
    assert payload["link_count"] == 3
    assert payload["violation_count"] == 2
    assert {violation["kind"] for violation in payload["violations"]} == {
        "unauthenticated_link_failure"
    }


def test_live_link_violations_flags_network_errors(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x: https://example.invalid/x\n", encoding="utf-8")
    links = extract_public_links([tmp_path], base_root=tmp_path)

    violations = live_link_violations(links, checker=lambda _: (None, "TimeoutError"))

    assert len(violations) == 1
    assert violations[0].detail == "TimeoutError returned no status"


def test_unauthenticated_status_falls_back_to_get_after_head_404(monkeypatch) -> None:
    calls: list[str] = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    def fake_urlopen(request, timeout: float):
        calls.append(request.get_method())
        if request.get_method() == "HEAD":
            raise HTTPError(request.full_url, 404, "not found", hdrs=None, fp=None)
        return Response()

    monkeypatch.setattr(audit_mod.urllib.request, "urlopen", fake_urlopen)

    assert unauthenticated_status("https://example.invalid/dataset") == (200, "GET")
    assert calls == ["HEAD", "GET"]
