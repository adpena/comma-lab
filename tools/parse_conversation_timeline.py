#!/usr/bin/env python3
"""Extract a research narrative timeline from Claude Code conversation transcripts.

Parses JSONL transcripts (Claude Code sessions and OMX agent logs) to produce
a structured JSON timeline of scores, breakthroughs, failures, ideas, and decisions.

Usage:
    python tools/parse_conversation_timeline.py \
        --transcript PATH_TO_TRANSCRIPT.jsonl \
        --omx-dir .omx/logs/ \
        --out reports/graphs/site/conversation_timeline.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Matches score-like numbers in context: "score: 1.727", "scorer 3.547", standalone decimals
# that look like metric scores (0.x - 9.x range, 2-4 decimal places).
SCORE_RE = re.compile(
    r"""
    (?:score[:\s]*|scorer[:\s]*|result[:\s]*|total[:\s]*|current_workflow[:\s"*]*|rule_faithful[:\s"*]*)
    (\d+\.\d{2,5})
    |
    \b(\d\.\d{2,5})\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

BREAKTHROUGH_RE = re.compile(
    r"\b(BREAKTHROUGH|PROMOTED|PROMOTE|new.best|new.record|beats.leaderboard)\b", re.IGNORECASE
)

FAILURE_RE = re.compile(
    r"\b(FAILED|FAILURE|NEGATIVE|didn['']t work|worse|OOM|out.of.memory|regression|degradation|dead.end)\b",
    re.IGNORECASE,
)

IDEA_RE = re.compile(
    r"\b(what about|what if|should we|maybe we|could we|how about|idea:|experiment:)\b", re.IGNORECASE
)

DECISION_RE = re.compile(
    r"\b(decision:|promote|demote|kill|proceed|approved|rejected|abort|go with)\b", re.IGNORECASE
)


def extract_text(content) -> str:
    """Pull plain text from a message content field (str or content-block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    # Skip thinking blocks – they are internal reasoning
                    pass
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def extract_scores(text: str) -> list[float]:
    """Return all plausible metric scores found in *text*."""
    scores = []
    for m in SCORE_RE.finditer(text):
        val_str = m.group(1) or m.group(2)
        try:
            val = float(val_str)
            # Filter to plausible contest score range (0.0 – 30.0)
            if 0.0 < val < 30.0:
                scores.append(val)
        except ValueError:
            pass
    return scores


def classify_message(text: str) -> list[str]:
    """Return a list of event types detected in *text*."""
    types = []
    if BREAKTHROUGH_RE.search(text):
        types.append("breakthrough")
    if FAILURE_RE.search(text):
        types.append("failure")
    if IDEA_RE.search(text):
        types.append("idea")
    if DECISION_RE.search(text):
        types.append("decision")
    if extract_scores(text):
        types.append("score")
    return types


# ---------------------------------------------------------------------------
# Transcript parser (Claude Code JSONL)
# ---------------------------------------------------------------------------

def parse_transcript(path: str) -> dict:
    """Stream-parse a Claude Code JSONL transcript and return structured data."""
    timeline = []
    score_trajectory = []
    total_messages = 0
    total_user = 0
    total_assistant = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            msg_type = obj.get("type")
            if msg_type not in ("user", "assistant"):
                continue

            total_messages += 1
            timestamp = obj.get("timestamp", "")
            message = obj.get("message", {})
            if not isinstance(message, dict):
                continue

            role = message.get("role", msg_type)
            content = message.get("content", "")
            text = extract_text(content)

            if not text.strip():
                continue

            if role == "user":
                total_user += 1
                speaker = "user"
            else:
                total_assistant += 1
                speaker = "assistant"

            # Classify
            types = classify_message(text)
            scores = extract_scores(text)

            # Always record score trajectory
            for s in scores:
                score_trajectory.append({"timestamp": timestamp, "score": s})

            if types:
                excerpt = text[:200].replace("\n", " ").strip()
                for t in types:
                    entry = {
                        "timestamp": timestamp,
                        "type": t,
                        "speaker": speaker,
                        "excerpt": excerpt,
                    }
                    if t == "score" and scores:
                        entry["scores"] = scores
                    timeline.append(entry)

    return {
        "total_messages": total_messages,
        "total_user": total_user,
        "total_assistant": total_assistant,
        "timeline": timeline,
        "score_trajectory": score_trajectory,
    }


# ---------------------------------------------------------------------------
# OMX log parser
# ---------------------------------------------------------------------------

def parse_omx_logs(omx_dir: str) -> dict:
    """Parse all turns-*.jsonl files in the OMX logs directory."""
    timeline = []
    score_trajectory = []
    total_turns = 0

    log_dir = Path(omx_dir)
    if not log_dir.is_dir():
        print(f"  [warn] OMX dir not found: {omx_dir}", file=sys.stderr)
        return {"total_turns": 0, "timeline": [], "score_trajectory": []}

    for log_file in sorted(log_dir.glob("turns-*.jsonl")):
        with open(log_file, "r", encoding="utf-8") as f:
            for raw_line in f:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                total_turns += 1
                timestamp = obj.get("timestamp", "")
                # OMX logs have input_preview and output_preview
                input_text = obj.get("input_preview", "")
                output_text = obj.get("output_preview", "")

                for text, speaker in [(input_text, "omx-input"), (output_text, "omx-output")]:
                    if not text:
                        continue
                    types = classify_message(text)
                    scores = extract_scores(text)

                    for s in scores:
                        score_trajectory.append({"timestamp": timestamp, "score": s})

                    if types:
                        excerpt = text[:200].replace("\n", " ").strip()
                        for t in types:
                            entry = {
                                "timestamp": timestamp,
                                "type": t,
                                "speaker": speaker,
                                "excerpt": excerpt,
                            }
                            if t == "score" and scores:
                                entry["scores"] = scores
                            timeline.append(entry)

    return {
        "total_turns": total_turns,
        "timeline": timeline,
        "score_trajectory": score_trajectory,
    }


# ---------------------------------------------------------------------------
# Key moments selector
# ---------------------------------------------------------------------------

PRIORITY = {"breakthrough": 0, "failure": 1, "idea": 2, "decision": 3, "score": 4}


def select_key_moments(timeline: list, limit: int = 50) -> list:
    """Pick the top *limit* most significant moments from the combined timeline."""
    # Prioritize breakthroughs > failures > ideas > decisions > scores
    sorted_events = sorted(timeline, key=lambda e: (PRIORITY.get(e["type"], 9), e.get("timestamp", "")))
    return sorted_events[:limit]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse Claude Code conversation transcripts into a research timeline.")
    parser.add_argument("--transcript", required=True, help="Path to the main JSONL transcript file")
    parser.add_argument("--omx-dir", default=None, help="Path to OMX logs directory containing turns-*.jsonl")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    parser.add_argument("--key-moments-limit", type=int, default=50, help="Max key moments to include (default 50)")
    args = parser.parse_args()

    print(f"Parsing transcript: {args.transcript}")
    transcript_data = parse_transcript(args.transcript)
    print(f"  Messages: {transcript_data['total_messages']} (user={transcript_data['total_user']}, assistant={transcript_data['total_assistant']})")
    print(f"  Timeline events: {len(transcript_data['timeline'])}")
    print(f"  Score points: {len(transcript_data['score_trajectory'])}")

    omx_data = {"total_turns": 0, "timeline": [], "score_trajectory": []}
    if args.omx_dir:
        print(f"Parsing OMX logs: {args.omx_dir}")
        omx_data = parse_omx_logs(args.omx_dir)
        print(f"  OMX turns: {omx_data['total_turns']}")
        print(f"  OMX timeline events: {len(omx_data['timeline'])}")
        print(f"  OMX score points: {len(omx_data['score_trajectory'])}")

    # Merge timelines
    combined_timeline = transcript_data["timeline"] + omx_data["timeline"]
    combined_timeline.sort(key=lambda e: e.get("timestamp", ""))

    combined_scores = transcript_data["score_trajectory"] + omx_data["score_trajectory"]
    combined_scores.sort(key=lambda e: e.get("timestamp", ""))

    key_moments = select_key_moments(combined_timeline, limit=args.key_moments_limit)

    result = {
        "metadata": {
            "transcript_file": os.path.basename(args.transcript),
            "omx_dir": args.omx_dir,
            "total_messages": transcript_data["total_messages"],
            "total_user": transcript_data["total_user"],
            "total_assistant": transcript_data["total_assistant"],
            "omx_total_turns": omx_data["total_turns"],
            "timeline_event_count": len(combined_timeline),
            "score_point_count": len(combined_scores),
        },
        "timeline": combined_timeline,
        "score_trajectory": combined_scores,
        "key_moments": key_moments,
    }

    # Ensure output directory exists
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Combined timeline events: {len(combined_timeline)}")
    print(f"  Combined score points: {len(combined_scores)}")
    print(f"  Key moments: {len(key_moments)}")


if __name__ == "__main__":
    main()
