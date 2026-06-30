#!/usr/bin/env python3
"""Idempotent Cloudflare NAMED-tunnel + DNS setup via the API-token path (no cert.pem).

The operator wants a STABLE hostname (e.g. comma-lab.adpena.com) instead of an
ephemeral trycloudflare quick-tunnel URL that churns on every restart. There is
NO ``~/.cloudflared/cert.pem`` (cloudflared is not browser-logged-in), so the
``cloudflared tunnel create/route`` CLI path is unavailable; this uses the
REST API (Bearer ``CLOUDFLARE_API_TOKEN``) + the token-run connector path
(``cloudflared tunnel run`` with ``TUNNEL_TOKEN`` in the env — NO secret on the
cmdline, so the durable-daemon registry never records the token).

``ensure_named_tunnel(hostname, port)`` is fully idempotent: it reuses an existing
tunnel / DNS record when present and only creates what is missing. It returns the
tunnel id + connector token (for the caller to put in the env) + an Access-gating
status. The token is NEVER printed; the ``__main__`` path prints a redacted
summary only.

SECURITY (CLAUDE.md "Public Disclosure Hygiene", NON-NEGOTIABLE): the dashboard's
TRIALITY tab describes our method, so the public hostname must be GATED. This
helper ATTEMPTS Cloudflare Access (email one-time-PIN for the allowed address);
if the API token lacks Access scope (common — Access:Edit is a separate
permission) it returns ``access_enabled=False`` with the reason so the caller can
fall back to the dashboard's own app-layer access-key gate and FLAG the operator
rather than silently exposing the method.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

BASE = "https://api.cloudflare.com/client/v4"


def _api(method: str, path: str, token: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            return r.getcode(), json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"success": False, "errors": [{"message": f"HTTP {e.code}"}]}
    except Exception as e:  # network / DNS error
        return 0, {"success": False, "errors": [{"message": str(e)}]}


def _apex(hostname: str) -> str:
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def ensure_named_tunnel(hostname: str, port: int, token: str | None = None,
                        name: str = "comma-lab",
                        allow_email: str | None = None) -> dict:
    """Ensure a named tunnel + ingress (→ http://127.0.0.1:<port>) + proxied CNAME
    + (best-effort) Access policy. Returns a dict including ``tunnel_token`` (the
    connector token, secret) or an ``error``."""
    token = token or os.environ.get("CLOUDFLARE_API_TOKEN")
    if not token:
        return {"ok": False, "error": "CLOUDFLARE_API_TOKEN not set"}
    out: dict = {"ok": False, "hostname": hostname, "port": port, "name": name}

    # 1. account
    c, j = _api("GET", "/accounts", token)
    if not j.get("success") or not j.get("result"):
        return {**out, "error": f"accounts {c}: {j.get('errors')}"}
    acct = j["result"][0]["id"]
    out["account_id"] = acct

    # 2. zone
    apex = _apex(hostname)
    c, j = _api("GET", f"/zones?name={apex}", token)
    if not j.get("success") or not j.get("result"):
        return {**out, "error": f"zone {apex} {c}: {j.get('errors')}"}
    zone = j["result"][0]["id"]
    out["zone_id"] = zone

    # 3. tunnel (idempotent — reuse if present)
    c, j = _api("GET", f"/accounts/{acct}/cfd_tunnel?name={name}&is_deleted=false", token)
    existing = j.get("result") or [] if j.get("success") else []
    if existing:
        tid = existing[0]["id"]
        out["created"] = False
    else:
        c, j = _api("POST", f"/accounts/{acct}/cfd_tunnel", token,
                    {"name": name, "config_src": "cloudflare"})
        if not j.get("success"):
            return {**out, "error": f"create tunnel {c}: {j.get('errors')}"}
        tid = j["result"]["id"]
        out["created"] = True
    out["tunnel_id"] = tid

    # 3b. connector token (always fetch via the token endpoint — works new+existing)
    c, j = _api("GET", f"/accounts/{acct}/cfd_tunnel/{tid}/token", token)
    if not j.get("success"):
        return {**out, "error": f"tunnel token {c}: {j.get('errors')}"}
    out["tunnel_token"] = j["result"]  # SECRET — caller keeps it out of logs/cmdline

    # 4. ingress (remote-managed config)
    ingress = {"config": {"ingress": [
        {"hostname": hostname, "service": f"http://127.0.0.1:{port}"},
        {"service": "http_status:404"},
    ]}}
    c, j = _api("PUT", f"/accounts/{acct}/cfd_tunnel/{tid}/configurations", token, ingress)
    out["ingress_ok"] = bool(j.get("success"))
    if not j.get("success"):
        out["ingress_error"] = j.get("errors")

    # 5. DNS CNAME -> <tid>.cfargotunnel.com (proxied), idempotent
    content = f"{tid}.cfargotunnel.com"
    rec = {"type": "CNAME", "name": hostname, "content": content, "proxied": True, "ttl": 1}
    c, j = _api("GET", f"/zones/{zone}/dns_records?name={hostname}", token)
    recs = j.get("result") or [] if j.get("success") else []
    if recs:
        rid = recs[0]["id"]
        c, j = _api("PUT", f"/zones/{zone}/dns_records/{rid}", token, rec)
    else:
        c, j = _api("POST", f"/zones/{zone}/dns_records", token, rec)
    out["dns_ok"] = bool(j.get("success"))
    if not j.get("success"):
        out["dns_error"] = j.get("errors")

    # 6. Access (best-effort; token may lack Access:Edit -> FLAG, don't fail)
    out["access_enabled"] = False
    if allow_email:
        acc = _ensure_access(acct, hostname, allow_email, token)
        out["access_enabled"] = acc.get("enabled", False)
        out["access_detail"] = acc.get("detail")

    out["ok"] = bool(out.get("ingress_ok") and out.get("dns_ok") and out.get("tunnel_token"))
    return out


def _ensure_access(acct: str, hostname: str, email: str, token: str) -> dict:
    """Best-effort Cloudflare Access self-hosted app + one-time-PIN policy for a
    single allowed email. Returns {enabled, detail}. A 403 (no Access scope) is
    reported, not raised."""
    c, j = _api("GET", f"/accounts/{acct}/access/apps", token)
    if not j.get("success"):
        return {"enabled": False, "detail": f"access/apps {c}: {j.get('errors')}"}
    app_id = None
    for a in j.get("result", []):
        if a.get("domain") == hostname:
            app_id = a.get("id")
            break
    if app_id is None:
        c, j = _api("POST", f"/accounts/{acct}/access/apps", token, {
            "name": f"dashboard {hostname}", "domain": hostname, "type": "self_hosted",
            "session_duration": "24h",
        })
        if not j.get("success"):
            return {"enabled": False, "detail": f"create app {c}: {j.get('errors')}"}
        app_id = j["result"]["id"]
    pol = {"name": "operator-only", "decision": "allow", "precedence": 1,
           "include": [{"email": {"email": email}}]}
    c, j = _api("POST", f"/accounts/{acct}/access/apps/{app_id}/policies", token, pol)
    if not j.get("success"):
        # policy may already exist; treat 409/duplicate as enabled
        msg = str(j.get("errors"))
        if "duplicate" in msg.lower() or "exists" in msg.lower():
            return {"enabled": True, "detail": "policy already present"}
        return {"enabled": False, "detail": f"policy {c}: {j.get('errors')}"}
    return {"enabled": True, "detail": "access app + one-time-PIN policy created"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hostname", default="comma-lab.adpena.com")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--name", default="comma-lab")
    ap.add_argument("--allow-email", default=None,
                    help="email to allow via Cloudflare Access (best-effort)")
    a = ap.parse_args()
    res = ensure_named_tunnel(a.hostname, a.port, name=a.name, allow_email=a.allow_email)
    # REDACT the secret token in any printed output.
    redacted = dict(res)
    if "tunnel_token" in redacted:
        redacted["tunnel_token"] = f"<redacted len={len(redacted['tunnel_token'])}>"
    print(json.dumps(redacted, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
