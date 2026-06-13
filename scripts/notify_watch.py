"""Push match notifications to ntfy.sh — kickoffs, goals, full-time results.

Designed to run on a schedule (GitHub Actions, every ~5 min) with a small JSON state
file carried between runs, so each event notifies exactly once. Subscribers just
install the ntfy app (iOS/Android) or open ntfy.sh and subscribe to the topic.

Env:
    NTFY_TOPIC   required — the topic to publish to (treat it like a password)
    NTFY_SERVER  default https://ntfy.sh
    NTFY_EVENTS  default "kickoff,goal,fulltime" (csv; subset to taste)
    NTFY_TEAMS   optional csv filter, e.g. "Mexico,Spain" — only their matches
    NOTIFY_STATE default .notify_state.json — state file path

Local test:  NTFY_TOPIC=my-topic python scripts/notify_watch.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from src.config import CONFIG  # noqa: E402
from src.data.espn_live import fetch_scoreboard  # noqa: E402

SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
TOPIC = os.environ.get("NTFY_TOPIC", "")
EVENTS = {e.strip() for e in
          os.environ.get("NTFY_EVENTS", "kickoff,goal,fulltime").split(",") if e.strip()}
TEAMS = {CONFIG.normalize(t.strip()) for t in
         os.environ.get("NTFY_TEAMS", "").split(",") if t.strip()}
STATE_PATH = Path(os.environ.get("NOTIFY_STATE", ".notify_state.json"))


def publish(title: str, body: str, tags: str = "soccer", priority: str = "default") -> None:
    requests.post(f"{SERVER}/{TOPIC}", data=body.encode("utf-8"),
                  headers={"Title": title.encode("utf-8"), "Tags": tags,
                           "Priority": priority}, timeout=15)


def main() -> int:
    if not TOPIC:
        print("NTFY_TOPIC not set — nothing to do")
        return 0
    state: dict = {}
    first_run = not STATE_PATH.exists()  # seed silently: no catch-up blast
    if not first_run:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))

    board = fetch_scoreboard()
    sent = 0
    for m in board:
        home, away = m["home"], m["away"]
        if first_run or (TEAMS and not ({home, away} & TEAMS)):
            state[f"{home}|{away}|{m['kickoff']:%Y%m%d}"] = {
                "state": m["state"], "hs": m["home_score"], "as": m["away_score"]}
            continue
        key = f"{home}|{away}|{m['kickoff']:%Y%m%d}"
        prev = state.get(key, {"state": "pre", "hs": 0, "as": 0})
        score = f"{home} {m['home_score']}-{m['away_score']} {away}"

        if (m["state"] == "in" and prev["state"] == "pre" and "kickoff" in EVENTS):
            publish("Kick-off", f"{home} vs {away} is under way",
                    tags="soccer,stadium")
            sent += 1
        if (m["state"] in ("in", "post") and "goal" in EVENTS
                and (m["home_score"], m["away_score"]) != (prev["hs"], prev["as"])
                and prev["state"] != "pre"):  # skip the catch-up on first sighting
            publish(f"Goal — {m['detail']}", score, tags="soccer", priority="high")
            sent += 1
        if m["state"] == "post" and prev["state"] != "post" and "fulltime" in EVENTS:
            publish("Full time", score, tags="checkered_flag", priority="high")
            sent += 1

        state[key] = {"state": m["state"], "hs": m["home_score"],
                      "as": m["away_score"]}

    STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    in_play = sum(1 for m in board if m["state"] == "in")
    print(f"board={len(board)} in_play={in_play} notifications_sent={sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
