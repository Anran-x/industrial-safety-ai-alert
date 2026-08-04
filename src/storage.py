"""
SQLite 告警事件存储:幂等插入、按时间查询、CSV 导出。
复用 SQL 能力,形成"告警-存档-追溯"闭环。
"""
import csv
import json
import sqlite3
from pathlib import Path
from typing import List, Optional

from src.config import ALERT_DB
from src.multimodal.report_gen import AlertEvent


class AlertStore:
    def __init__(self, db_path: str = str(ALERT_DB)):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                confidence REAL,
                zone TEXT DEFAULT '',
                track_id INTEGER,
                screenshot TEXT DEFAULT '',
                detail TEXT DEFAULT ''
            )
            """
        )
        self._conn.commit()

    def insert(self, ev: AlertEvent) -> int:
        cur = self._conn.execute(
            "INSERT INTO alerts(ts,alert_type,confidence,zone,track_id,screenshot,detail) VALUES(?,?,?,?,?,?,?)",
            (ev.timestamp, ev.alert_type, ev.confidence, ev.zone, ev.track_id, ev.screenshot, ev.detail),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def query(self, alert_type: Optional[str] = None, limit: int = 200) -> List[dict]:
        if alert_type:
            rows = self._conn.execute(
                "SELECT * FROM alerts WHERE alert_type=? ORDER BY id DESC LIMIT ?", (alert_type, limit)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        cols = [d[0] for d in self._conn.execute("SELECT * FROM alerts LIMIT 1").description]
        return [dict(zip(cols, r)) for r in rows]

    def export_csv(self, path: str) -> str:
        rows = self.query(limit=100000)
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        return path

    def stats(self) -> dict:
        d = {}
        for row in self._conn.execute("SELECT alert_type, COUNT(*) FROM alerts GROUP BY alert_type"):
            d[row[0]] = row[1]
        return d