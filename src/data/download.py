"""
Атомарная загрузка записей MIT-BIH Arrhythmia Database с PhysioNet.
"""

import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import yaml

DS1_RECORDS: List[str] = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2_RECORDS: List[str] = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]
ALL_RECORDS: List[str] = DS1_RECORDS + DS2_RECORDS
BASE_URL = "https://physionet.org/files/mitdb/1.0.0"


def _download_file_atomic(url: str, target_path: Path, retries: int = 5, backoff: float = 2.0) -> None:
    """Скачивает файл во временный буфер с повторами при сбоях сети и атомарным переносом."""
    target_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=f".{target_path.name}_", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                expected_len = response.headers.get("Content-Length")
                data = response.read()
                if len(data) == 0:
                    raise IOError(f"Пустой ответ при загрузке {url}")
                if expected_len is not None and len(data) != int(expected_len):
                    raise IOError(f"Неполная загрузка {url}: получено {len(data)} из {expected_len} байт")
                tmp_path.write_bytes(data)
            os.replace(tmp_path, target_path)
            return
        except (urllib.error.URLError, TimeoutError, IOError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise
            wait = backoff * attempt
            print(f"[WARN] Сбой сети ({exc}), повтор {attempt}/{retries} через {wait:.0f} с...")
            time.sleep(wait)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise


def download_mitbih(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rec in ALL_RECORDS:
        for ext in (".hea", ".dat", ".atr"):
            dest = raw_dir / f"{rec}{ext}"
            if dest.exists() and dest.stat().st_size > 16:
                continue
            url = f"{BASE_URL}/{rec}{ext}"
            print(f"[DOWNLOAD] {url} -> {dest}")
            _download_file_atomic(url, dest)
            time.sleep(0.15)
    print(f"[OK] Все 44 записи MIT-BIH (DS1 + DS2) проверены в '{raw_dir}'.")


if __name__ == "__main__":
    cfg_path = ROOT_DIR / "configs/config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw_dir = ROOT_DIR / cfg.get("data", {}).get("raw_dir", "data/raw/mitbih")
    download_mitbih(raw_dir)