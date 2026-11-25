"""
Формирование пациентонезависимых наборов DS1 (train) и DS2 (test) из MIT-BIH.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import wfdb
import yaml

from src.contracts import LABEL_TO_IDX, validate_npz_arrays
from src.data.download import DS1_RECORDS, DS2_RECORDS
from src.data.filtering import evaluate_r_peak_detector, pan_tompkins_detector, preprocess_ecg

# Маппинг символов MIT-BIH в суперклассы AAMI EC57 (исключая Q — неопределённые/кардиостимулятор)
AAMI_SYMBOL_MAP: Dict[str, str] = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}


def compute_rr_features(rr_intervals_sec: np.ndarray, idx: int) -> np.ndarray:
    """
    Формирует 6 физиологических признаков ритма для удара idx (между интервалами idx-1 и idx):
      1. pre_rr: предыдущий RR-интервал (клиппированный в [0.25, 2.2] с)
      2. post_rr: следующий RR-интервал (клиппированный в [0.25, 2.2] с)
      3. local_rr: скользящее среднее 8 предшествующих RR-интервалов (БЕЗ текущего pre_rr!)
      4. pre_to_local: отношение pre_rr / local_rr (маркер преждевременных экстрасистол S/V)
      5. post_to_local: отношение post_rr / local_rr (маркер компенсаторной паузы)
      6. rr_asymmetry: (post_rr - pre_rr) / local_rr
    """
    clipped = np.clip(rr_intervals_sec, 0.25, 2.2)
    pre_rr = float(clipped[idx - 1])
    post_rr = float(clipped[idx])

    # Исключаем текущий pre_rr (idx - 1) из локальной базы, чтобы экстрасистола не искажала норму
    hist_end = idx - 1
    if hist_end > 0:
        hist_start = max(0, hist_end - 8)
        local_rr = float(np.median(clipped[hist_start:hist_end]))
    else:
        local_rr = pre_rr

    local_rr = max(local_rr, 0.25)
    pre_to_local = pre_rr / local_rr
    post_to_local = post_rr / local_rr
    rr_asymmetry = (post_rr - pre_rr) / local_rr

    return np.array(
        [pre_rr, post_rr, local_rr, pre_to_local, post_to_local, rr_asymmetry],
        dtype=np.float32,
    )


def build_split_dataset(
    records: List[str],
    raw_dir: Path,
    cfg: Dict,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    data_cfg = cfg.get("data", {})
    sig_cfg = cfg.get("signal_processing", {})

    fs = int(data_cfg.get("sampling_rate", 360))
    win_before = int(sig_cfg.get("window_before_r", 96))
    win_after = int(sig_cfg.get("window_after_r", 160))
    tol_ms = float(sig_cfg.get("r_match_tolerance_ms", 50.0))
    tol_samples = int(round(tol_ms * fs / 1000.0))

    all_signals: List[np.ndarray] = []
    all_rr: List[np.ndarray] = []
    all_labels: List[int] = []
    all_pids: List[int] = []

    total_tp, total_fp, total_fn = 0, 0, 0
    total_detected, total_ref_aami = 0, 0

    for rec_name in records:
        rec_path = str(raw_dir / rec_name)
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        # Выбираем канал MLII (обычно индекс 0, кроме записи 114 где MLII на индексе 1)
        sig_names = [s.upper() for s in record.sig_name]
        ch_idx = sig_names.index("MLII") if "MLII" in sig_names else 0
        raw_sig = record.p_signal[:, ch_idx]

        clean_sig = preprocess_ecg(
            raw_sig,
            fs=fs,
            lowcut=float(sig_cfg.get("bandpass_low", 0.5)),
            highcut=float(sig_cfg.get("bandpass_high", 40.0)),
            order=int(sig_cfg.get("filter_order", 4)),
            norm_window_sec=float(sig_cfg.get("norm_window_sec", 4.0)),
        )

        # Автономный поиск R-пиков без использования разметки
        det_peaks = pan_tompkins_detector(clean_sig, fs=fs)

        # Эталонные аннотации AAMI
        ref_samples = []
        ref_classes = []
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in AAMI_SYMBOL_MAP:
                ref_samples.append(int(s))
                ref_classes.append(LABEL_TO_IDX[AAMI_SYMBOL_MAP[sym]])
        ref_samples_arr = np.asarray(ref_samples, dtype=np.int64)
        ref_classes_arr = np.asarray(ref_classes, dtype=np.int64)

        det_stats = evaluate_r_peak_detector(det_peaks, ref_samples_arr, fs=fs, tolerance_ms=tol_ms)
        total_tp += det_stats["TP"]
        total_fp += det_stats["FP"]
        total_fn += det_stats["FN"]
        total_detected += len(det_peaks)
        total_ref_aami += len(ref_samples_arr)

        if len(det_peaks) < 3:
            continue

        rr_intervals_sec = np.diff(det_peaks) / float(fs)
        matched_ref = np.zeros(len(ref_samples_arr), dtype=bool)
        pid_int = int(rec_name)

        for i in range(1, len(det_peaks) - 1):
            r_pos = int(det_peaks[i])
            if r_pos - win_before < 0 or r_pos + win_after > len(clean_sig):
                continue

            free_idx = np.where(~matched_ref)[0]
            if len(free_idx) == 0:
                break
            diffs = np.abs(ref_samples_arr[free_idx] - r_pos)
            best_local = int(np.argmin(diffs))
            if diffs[best_local] > tol_samples:
                continue

            ref_i = free_idx[best_local]
            matched_ref[ref_i] = True

            beat_win = clean_sig[r_pos - win_before : r_pos + win_after]
            rr_feat = compute_rr_features(rr_intervals_sec, i)

            all_signals.append(beat_win)
            all_rr.append(rr_feat)
            all_labels.append(int(ref_classes_arr[ref_i]))
            all_pids.append(pid_int)

    signals_arr = np.asarray(all_signals, dtype=np.float32)[:, np.newaxis, :]
    rr_arr = np.asarray(all_rr, dtype=np.float32)
    labels_arr = np.asarray(all_labels, dtype=np.int64)
    pids_arr = np.asarray(all_pids, dtype=np.int64)

    validate_npz_arrays(
        signals_arr,
        rr_arr,
        labels_arr,
        pids_arr,
        expected_signal_len=win_before + win_after,
        expected_rr_dim=6,
    )

    se = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    summary_det = {
        "split": split_name,
        "total_reference_aami_beats": total_ref_aami,
        "total_detected_peaks": total_detected,
        "matched_tp": total_tp,
        "unmatched_detector_fp": total_fp,
        "missed_reference_fn": total_fn,
        "detector_sensitivity": round(se, 4),
        "detector_ppv": round(ppv, 4),
    }
    return signals_arr, rr_arr, labels_arr, pids_arr, summary_det


def main() -> None:
    cfg_path = ROOT_DIR / "configs/config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw_dir = ROOT_DIR / cfg.get("data", {}).get("raw_dir", "data/raw/mitbih")
    out_dir = ROOT_DIR / cfg.get("data", {}).get("processed_dir", "data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Генерация DS1 (22 пациента для обучения)...")
    ds1_sig, ds1_rr, ds1_y, ds1_pid, ds1_det = build_split_dataset(DS1_RECORDS, raw_dir, cfg, "ds1_train")
    np.savez_compressed(out_dir / "ds1_train.npz", signals=ds1_sig, rr=ds1_rr, labels=ds1_y, patient_ids=ds1_pid)
    print(f"[OK] DS1 сохранён: {len(ds1_y):,} ударов | Детектор R-пиков Se={ds1_det['detector_sensitivity']}, PPV={ds1_det['detector_ppv']}")

    print("[2/2] Генерация DS2 (22 новых пациента для тестирования)...")
    ds2_sig, ds2_rr, ds2_y, ds2_pid, ds2_det = build_split_dataset(DS2_RECORDS, raw_dir, cfg, "ds2_test")
    np.savez_compressed(out_dir / "ds2_test.npz", signals=ds2_sig, rr=ds2_rr, labels=ds2_y, patient_ids=ds2_pid)
    print(f"[OK] DS2 сохранён: {len(ds2_y):,} ударов | Детектор R-пиков Se={ds2_det['detector_sensitivity']}, PPV={ds2_det['detector_ppv']}")

    det_report_path = ROOT_DIR / "reports/detector_ds2_stats.json"
    det_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(det_report_path, "w", encoding="utf-8") as f:
        json.dump({"ds1": ds1_det, "ds2": ds2_det}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()