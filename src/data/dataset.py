"""
Нарезка окон, расчёт RR-признаков и формирование датасетов DS1/DS2 — [DATA-03]
===============================================================================
Ветка: feature/data-pipeline (интеграция в main)
Файл: src/data/dataset.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import wfdb
import yaml

from src.contracts import AAMI_MAPPING, LABEL_TO_IDX, NUM_CLASSES
from src.data.filtering import pan_tompkins_detector, preprocess_ecg


def extract_beats_from_record(
    record_id: str,
    raw_dir: Path,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    filter_order: int = 4,
    win_before: int = 96,
    win_after: int = 160,
    rr_avg_window: int = 10,
    tolerance_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Обрабатывает одну запись MIT-BIH:
    1. Фильтрация и Z-нормализация канала MLII.
    2. Автономная детекция R-пиков алгоритмом Пана — Томпкинса.
    3. Сопоставление с эталонной разметкой (±50 мс) и маппинг в классы AAMI ('N', 'S', 'V', 'F').
    4. Выделение окон [1, 256] и 4 признаков ритма [pre_RR, post_RR, local_RR, ratio_RR].
    """
    rec_path = str(raw_dir / record_id)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")

    ch_idx = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    raw_signal = record.p_signal[:, ch_idx]
    rec_fs = float(record.fs) if record.fs else fs

    clean_signal = preprocess_ecg(
        raw_signal,
        fs=rec_fs,
        lowcut=lowcut,
        highcut=highcut,
        order=filter_order,
    )
    det_peaks = pan_tompkins_detector(clean_signal, fs=rec_fs, is_preprocessed=True)

    if len(det_peaks) < 3:
        return (
            np.empty((0, 1, win_before + win_after), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    # Эталонные пики, входящие в 4 целевых класса AAMI (N, S, V, F)
    ref_samples = []
    ref_classes = []
    for sample_idx, sym in zip(ann.sample, ann.symbol):
        aami_cls = AAMI_MAPPING.get(sym)
        if aami_cls in LABEL_TO_IDX:
            ref_samples.append(int(sample_idx))
            ref_classes.append(LABEL_TO_IDX[aami_cls])

    ref_samples_arr = np.asarray(ref_samples, dtype=np.int64)
    ref_classes_arr = np.asarray(ref_classes, dtype=np.int64)
    tol_samples = int(round((tolerance_ms / 1000.0) * rec_fs))

    # Предварительный расчёт всех RR-интервалов в секундах по детектированным пикам
    rr_intervals_sec = np.diff(det_peaks) / rec_fs

    signals_list: List[np.ndarray] = []
    rr_list: List[List[float]] = []
    labels_list: List[int] = []

    sig_len = len(clean_signal)
    patient_num = int(record_id)

    for i in range(1, len(det_peaks) - 1):
        r_idx = int(det_peaks[i])

        # Проверка границ окна [r_idx - 96 : r_idx + 160]
        if r_idx - win_before < 0 or r_idx + win_after > sig_len:
            continue

        # Поиск ближайшей эталонной аннотации в пределах допуска ±50 мс
        if len(ref_samples_arr) == 0:
            continue
        diffs = np.abs(ref_samples_arr - r_idx)
        nearest_idx = int(np.argmin(diffs))
        if diffs[nearest_idx] > tol_samples:
            continue

        label_idx = int(ref_classes_arr[nearest_idx])

        # Вырезаем окно морфологии длиной 256 отсчётов
        window = clean_signal[r_idx - win_before : r_idx + win_after].astype(np.float32)

        # Вычисляем 4 табличных признака ритма
        pre_rr = float(rr_intervals_sec[i - 1])
        post_rr = float(rr_intervals_sec[i])
        hist_start = max(0, i - rr_avg_window)
        local_rr = float(np.mean(rr_intervals_sec[hist_start:i])) if i > hist_start else pre_rr
        ratio_rr = pre_rr / (local_rr + 1e-8)

        signals_list.append(window[np.newaxis, :])  # [1, 256]
        rr_list.append([pre_rr, post_rr, local_rr, ratio_rr])
        labels_list.append(label_idx)

    if not signals_list:
        return (
            np.empty((0, 1, win_before + win_after), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )

    signals_arr = np.stack(signals_list, axis=0).astype(np.float32)
    rr_arr = np.asarray(rr_list, dtype=np.float32)
    labels_arr = np.asarray(labels_list, dtype=np.int64)
    patients_arr = np.full(len(labels_arr), patient_num, dtype=np.int64)

    return signals_arr, rr_arr, labels_arr, patients_arr


def build_split_dataset(
    records: List[str],
    raw_dir: Path,
    sp_cfg: Dict,
    fs: float,
    split_name: str,
) -> Dict[str, np.ndarray]:
    """Собирает единый датасет (DS1 или DS2) по списку записей пациентов."""
    all_signals = []
    all_rr = []
    all_labels = []
    all_patients = []

    print(f"\n[DATA-03] Обработка выборки {split_name} ({len(records)} записей)...")
    for rec_id in records:
        sigs, rrs, lbls, pids = extract_beats_from_record(
            record_id=rec_id,
            raw_dir=raw_dir,
            fs=fs,
            lowcut=float(sp_cfg.get("bandpass_low_hz", 0.5)),
            highcut=float(sp_cfg.get("bandpass_high_hz", 40.0)),
            filter_order=int(sp_cfg.get("filter_order", 4)),
            win_before=int(sp_cfg.get("window_before_r", 96)),
            win_after=int(sp_cfg.get("window_after_r", 160)),
            rr_avg_window=int(sp_cfg.get("rr_moving_avg_window", 10)),
            tolerance_ms=float(sp_cfg.get("detector_tolerance_ms", 50.0)),
        )
        all_signals.append(sigs)
        all_rr.append(rrs)
        all_labels.append(lbls)
        all_patients.append(pids)
        print(f"  Запись {rec_id}: извлечено {len(lbls):,} ударов")

    dataset = {
        "signals": np.concatenate(all_signals, axis=0),
        "rr": np.concatenate(all_rr, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "patient_ids": np.concatenate(all_patients, axis=0),
    }

    # Строгая проверка по контракту данных (формы и типы массивов)
    n = len(dataset["labels"])
    window_len = int(sp_cfg.get("window_before_r", 96)) + int(sp_cfg.get("window_after_r", 160))
    assert n > 0, f"Выборка {split_name} пуста!"
    assert dataset["signals"].shape == (n, 1, window_len) and dataset["signals"].dtype == np.float32
    assert dataset["rr"].shape == (n, 4) and dataset["rr"].dtype == np.float32
    assert dataset["labels"].dtype == np.int64 and np.all((dataset["labels"] >= 0) & (dataset["labels"] < NUM_CLASSES))
    assert dataset["patient_ids"].shape == (n,) and dataset["patient_ids"].dtype == np.int64

    class_names = ["N", "S", "V", "F"]
    print(f"[OK] {split_name} собран: всего {n:,} ударов")
    for idx, cname in enumerate(class_names):
        cnt = int(np.sum(dataset["labels"] == idx))
        print(f"     Класс {cname}: {cnt:>6,} ({cnt / n * 100:.2f}%)")

    return dataset


def main(config_path: str = "configs/config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    sp_cfg = cfg["signal_processing"]

    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    fs = float(data_cfg.get("sampling_rate", 360.0))

    train_out = Path(data_cfg.get("train_file", "data/processed/ds1_train.npz"))
    test_out = Path(data_cfg.get("test_file", "data/processed/ds2_test.npz"))
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    ds1 = build_split_dataset(data_cfg["ds1_train_records"], raw_dir, sp_cfg, fs, "DS1 (Train)")
    np.savez_compressed(train_out, **ds1)
    print(f"[OK] Файл сохранён: '{train_out}'")

    ds2 = build_split_dataset(data_cfg["ds2_test_records"], raw_dir, sp_cfg, fs, "DS2 (Hold-out Test)")
    np.savez_compressed(test_out, **ds2)
    print(f"[OK] Файл сохранён: '{test_out}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Формирование датасетов DS1 и DS2 из сырых записей MIT-BIH")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Путь к конфигурации")
    args = parser.parse_args()
    main(args.config)