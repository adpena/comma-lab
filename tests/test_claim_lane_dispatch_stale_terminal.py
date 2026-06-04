import datetime as dt

from tools import claim_lane_dispatch as claims


def test_generic_stale_status_closes_dispatch_claim() -> None:
    now = dt.datetime(2026, 6, 4, 5, 30, tzinfo=dt.UTC)
    rows = [
        claims.Claim(
            timestamp_utc="2026-06-04T04:00:00Z",
            agent="codex:test",
            lane_id="snerv_lane",
            platform="local_mlx",
            instance_job_id="local-pid-123",
            predicted_eta_utc="",
            status="active_running",
            notes="old live row",
        ),
        claims.Claim(
            timestamp_utc="2026-06-04T05:00:00Z",
            agent="codex:test",
            lane_id="snerv_lane",
            platform="local_mlx",
            instance_job_id="local-pid-123",
            predicted_eta_utc="",
            status="stale_pid_absent_checkpoint_exports_harvested",
            notes="terminal stale harvest row",
        ),
    ]

    summary = claims._summarize_claims(rows, now_utc=now, ttl_hours=24)

    assert summary["active_count"] == 0
    assert summary["stale_nonterminal_count"] == 0
    assert summary["terminal_latest_count"] == 1
    assert summary["terminal_latest"][0]["status"] == (
        "stale_pid_absent_checkpoint_exports_harvested"
    )

