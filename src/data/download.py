"""
Модуль загрузки MIT-BIH Arrhythmia Database и валидации разметки AAMI — [DATA-01]
================================================================================
Ответственный: Data & Signal Processing Lead
Ветка: feature/data-pipeline

Задачи:
1. Скачать записи MIT-BIH ('mitdb') из списков DS1 и DS2 в `data/raw/` через AWS S3 зеркало PhysioNet.
2. Проверить отсутствие пересечения пациентов между обучающей и тестовой выборками.
3. Распарсить аннотации кардиологов, выполнить маппинг в 4 класса AAMI ('N', 'S', 'V', 'F')
   и вывести статистику дисбаланса классов для DS1 и DS2.
"""

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml
import wfdb
from tqdm import tqdm

from src.contracts import AAMI_MAPPING, IDX_TO_LABEL, LABEL_TO_IDX


# Зеркала PhysioNet: AWS S3 обычно работает без блокировок и таймаутов
MIRROR_URLS = [
    "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0",
    "https://physionet.org/files/mitdb/1.0.0",
]


def verify_patient_split(ds1_records: List[str], ds2_records: List[str]) -> None:
    """Проверяет, что пациенты в DS1 и DS2 строго разделены."""
    overlap = set(ds1_records).intersection(set(ds2_records))
    assert len(overlap) == 0, (
        f"КРИТИЧЕСКАЯ ОШИБКА УТЕЧКИ ДАННЫХ: пациенты {overlap} присутствуют и в DS1, и в DS2!"
    )


def _create_http_session() -> requests.Session:
    """Создаёт HTTP-сессию с автоматическими повторами при обрывах связи."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_single_file(session: requests.Session, filename: str, target_path: Path) -> None:
    """Скачивает один файл, перебирая доступные зеркала."""
    last_error = None
    for base_url in MIRROR_URLS:
        url = f"{base_url}/{filename}"
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            target_path.write_bytes(response.content)
            return
        except Exception as err:
            last_error = err

    raise ConnectionError(
        f"Не удалось скачать '{filename}' ни с одного зеркала. Последняя ошибка: {last_error}"
    )


def download_mitdb_records(raw_dir: Path, records: List[str]) -> None:
    """Скачивает недостающие файлы (.dat, .hea, .atr) для списка записей MIT-BIH."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    extensions = [".dat", ".hea", ".atr"]

    files_to_download: List[Tuple[str, Path]] = []
    for rec in records:
        for ext in extensions:
            filename = f"{rec}{ext}"
            target_file = raw_dir / filename
            if not target_file.exists() or target_file.stat().st_size == 0:
                files_to_download.append((filename, target_file))

    if not files_to_download:
        print(f"[INFO] Все файлы для {len(records)} записей уже загружены в '{raw_dir}'.")
        return

    print(f"[INFO] Скачивание {len(files_to_download)} файлов в '{raw_dir}'...")
    session = _create_http_session()

    for filename, target_file in tqdm(files_to_download, desc="Загрузка MIT-BIH", unit="file"):
        _download_single_file(session, filename, target_file)

    print("[INFO] Загрузка успешно завершена.")


def parse_record_annotations(raw_dir: Path, record_id: str) -> Tuple[List[int], List[int], int]:
    """
    Читает аннотации кардиологов для одной записи и маппит их в классы AAMI.

    Возвращает:
        peaks:   список координат R-пиков (в отсчётах)
        labels:  список целочисленных меток [0..3]
        dropped: количество отброшенных аннотаций (класс 'Q' и служебные метки)
    """
    record_path = str(raw_dir / record_id)
    ann = wfdb.rdann(record_path, "atr")

    peaks: List[int] = []
    labels: List[int] = []
    dropped = 0

    for sample_idx, symbol in zip(ann.sample, ann.symbol):
        if symbol in AAMI_MAPPING:
            aami_class = AAMI_MAPPING[symbol]
            peaks.append(int(sample_idx))
            labels.append(LABEL_TO_IDX[aami_class])
        else:
            dropped += 1

    return peaks, labels, dropped


def summarize_split(raw_dir: Path, records: List[str], split_name: str) -> Dict[str, int]:
    """Собирает и печатает статистику классов AAMI по списку записей."""
    total_counts: Counter = Counter()
    total_dropped = 0

    for rec in tqdm(records, desc=f"Анализ {split_name}", unit="rec"):
        _, labels, dropped = parse_record_annotations(raw_dir, rec)
        total_counts.update(labels)
        total_dropped += dropped

    total_beats = sum(total_counts.values())
    print(f"\n=== Статистика классов для {split_name} ({len(records)} пациентов) ===")
    print(f"Всего валидных ударов (AAMI): {total_beats:,} | Отброшено (класс Q / служебные): {total_dropped:,}")

    for class_idx in range(len(IDX_TO_LABEL)):
        class_name = IDX_TO_LABEL[class_idx]
        count = total_counts[class_idx]
        pct = (count / total_beats * 100.0) if total_beats > 0 else 0.0
        print(f"  Класс {class_name}: {count:>7,} ({pct:5.2f}%)")

    return {IDX_TO_LABEL[idx]: total_counts[idx] for idx in range(len(IDX_TO_LABEL))}


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    ds1_records: List[str] = data_cfg["ds1_train_records"]
    ds2_records: List[str] = data_cfg["ds2_test_records"]

    # 1. Проверяем отсутствие утечки пациентов между выборками
    verify_patient_split(ds1_records, ds2_records)
    print("[OK] Проверка сплита: пересечений между DS1 и DS2 нет.")

    # 2. Скачиваем все 44 записи через AWS S3 зеркало
    all_records = ds1_records + ds2_records
    download_mitdb_records(raw_dir, all_records)

    # 3. Выводим распределение классов отдельно для train (DS1) и test (DS2)
    summarize_split(raw_dir, ds1_records, "DS1 (Train / Val)")
    summarize_split(raw_dir, ds2_records, "DS2 (Hold-out Test)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and parse MIT-BIH Arrhythmia Database")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Путь к файлу конфигурации")
    args = parser.parse_args()
    main(args.config)