"""
Модуль предобработки ЭКГ-сигнала и детекции R-пиков — [DATA-02]
================================================================
Ветка: feature/data-pipeline
Файл: src/data/filtering.py

Реализует:
1. Полосовой фильтр Баттерворта (0.5–40 Гц, 4-й порядок) + Z-нормализацию.
2. Автономный детектор R-пиков по алгоритму Пана — Томпкинса (без опоры на wfdb.rdann).
3. Валидацию детектора относительно эталонной разметки с допуском ±50 мс.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import yaml

from src.contracts import AAMI_MAPPING

# Все символы MIT-BIH, обозначающие реальные сокращения сердца (включая класс Q: '/', 'f', 'Q'),
# чтобы при оценке детектора R-пиков не штрафовать его за обнаружение кардиостимулированных/сливных ударов.
ALL_BEAT_SYMBOLS = set(AAMI_MAPPING.keys()) | {"/", "f", "Q"}


def bandpass_filter(
    signal: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """
    Полосовой фильтр Баттерворта (нулевая фаза через filtfilt) для подавления
    дрейфа изолинии (<0.5 Гц) и сетевых помех (50/60 Гц).
    """
    sig = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, sig)


def z_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-нормализация сигнала (mean=0, std=1)."""
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + eps)


def preprocess_ecg(
    raw_signal: np.ndarray,
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Полная предобработка канала ЭКГ: полосовая фильтрация 0.5–40 Гц + Z-нормализация."""
    filtered = bandpass_filter(raw_signal, fs=fs, lowcut=lowcut, highcut=highcut, order=order)
    return z_normalize(filtered)


def pan_tompkins_detector(
    signal: np.ndarray,
    fs: float = 360.0,
    is_preprocessed: bool = False,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    filter_order: int = 4,
) -> np.ndarray:
    """
    Автономный детектор R-пиков по алгоритму Пана — Томпкинса.

    Этапы:
    1. Полосовой фильтр 5–15 Гц для выделения энергии QRS-комплекса.
    2. 5-точечное дифференцирование для выделения крутизны фронтов QRS.
    3. Возведение в квадрат (нелинейное усиление).
    4. Скользящее оконное интегрирование (окно 150 мс).
    5. Адаптивная двухпороговая логика (SPKI / NPKI) с рефрактерным периодом 200 мс
       и механизмом search-back при пропуске удара (> 1.66 * RR_avg).
    6. Уточнение координаты вершины R-зубца на очищенном сигнале (0.5–40 Гц).

    Args:
        signal: Одномерный массив отсчётов ЭКГ.
        fs: Частота дискретизации (Гц).
        is_preprocessed: Передан ли уже отфильтрованный и нормализованный сигнал.

    Returns:
        Отсортированный 1D-массив индексов R-пиков (dtype=np.int64).
    """
    sig = np.asarray(signal, dtype=np.float64)
    base_signal = (
        sig
        if is_preprocessed
        else preprocess_ecg(sig, fs=fs, lowcut=lowcut, highcut=highcut, order=filter_order)
    )

    # 1. Фильтрация полосы QRS (5–15 Гц)
    nyq = 0.5 * fs
    b_qrs, a_qrs = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    qrs_filtered = filtfilt(b_qrs, a_qrs, base_signal)

    # 2. Производная Пана — Томпкинса
    diff_kernel = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (fs / 8.0)
    differentiated = np.convolve(qrs_filtered, diff_kernel, mode="same")

    # 3. Возведение в квадрат
    squared = differentiated ** 2

    # 4. Скользящее оконное интегрирование (150 мс)
    win_size = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win_size) / win_size, mode="same")

    # 5. Поиск кандидатов с рефрактерным периодом 200 мс
    refractory_samples = int(round(0.200 * fs))
    candidate_peaks, _ = find_peaks(integrated, distance=refractory_samples)

    if len(candidate_peaks) == 0:
        return np.array([], dtype=np.int64)

    # Инициализация порогов по первым 2 секундам
    init_len = min(len(integrated), int(2.0 * fs))
    spki = 0.25 * float(np.max(integrated[:init_len]))
    npki = 0.50 * float(np.mean(integrated[:init_len]))
    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1

    detected_peaks: List[int] = []
    rr_history: List[int] = []
    refine_radius = int(round(0.075 * fs))  # ±75 мс для точной привязки к вершине R-пика

    def _refine_peak(approx_idx: int) -> int:
        left = max(0, approx_idx - refine_radius)
        right = min(len(base_signal), approx_idx + refine_radius + 1)
        # Используем abs(), чтобы корректно центрировать и инвертированные V-экстрасистолы
        return left + int(np.argmax(np.abs(base_signal[left:right])))

    for pk in candidate_peaks:
        pk_val = float(integrated[pk])

        # Search-back: проверка на пропущенный удар при длинной паузе
        if len(detected_peaks) >= 2 and len(rr_history) >= 2:
            rr_avg = float(np.mean(rr_history[-8:]))
            last_pk = detected_peaks[-1]
            if (pk - last_pk) > 1.66 * rr_avg:
                sb_start = last_pk + refractory_samples
                sb_end = pk - refractory_samples
                if sb_end > sb_start:
                    sb_segment = integrated[sb_start:sb_end]
                    sb_rel = int(np.argmax(sb_segment))
                    sb_val = float(sb_segment[sb_rel])
                    if sb_val > threshold_i2:
                        refined_sb = _refine_peak(sb_start + sb_rel)
                        if (refined_sb - detected_peaks[-1]) >= refractory_samples:
                            rr_history.append(refined_sb - detected_peaks[-1])
                            detected_peaks.append(refined_sb)
                            spki = 0.25 * sb_val + 0.75 * spki
                            threshold_i1 = npki + 0.25 * (spki - npki)
                            threshold_i2 = 0.5 * threshold_i1

        # Классификация кандидата как R-пик или шум
        if pk_val >= threshold_i1:
            refined_pk = _refine_peak(int(pk))
            if not detected_peaks or (refined_pk - detected_peaks[-1]) >= refractory_samples:
                if detected_peaks:
                    rr_history.append(refined_pk - detected_peaks[-1])
                detected_peaks.append(refined_pk)
                spki = 0.125 * pk_val + 0.875 * spki
            else:
                npki = 0.125 * pk_val + 0.875 * npki
        else:
            npki = 0.125 * pk_val + 0.875 * npki

        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1

    return np.array(detected_peaks, dtype=np.int64)


def evaluate_r_peak_detector(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    fs: float = 360.0,
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """
    Валидирует детектор R-пиков относительно эталонной разметки wfdb.rdann
    с допуском ±tolerance_ms (по config.yaml: 50 мс = 18 отсчётов при 360 Гц).

    Сопоставление выполняется строго 1-к-1 по ближайшему эталонному пику.
    """
    det = np.sort(np.asarray(detected_peaks, dtype=np.int64))
    ref = np.sort(np.asarray(reference_peaks, dtype=np.int64))
    tol_samples = int(round((tolerance_ms / 1000.0) * fs))

    if len(ref) == 0 and len(det) == 0:
        return {"TP": 0, "FP": 0, "FN": 0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0, "MAE_ms": 0.0}
    if len(ref) == 0:
        return {"TP": 0, "FP": len(det), "FN": 0, "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "MAE_ms": 0.0}
    if len(det) == 0:
        return {"TP": 0, "FP": 0, "FN": len(ref), "Precision": 0.0, "Recall": 0.0, "F1": 0.0, "MAE_ms": 0.0}

    matched_ref = np.zeros(len(ref), dtype=bool)
    tp = 0
    abs_errors: List[int] = []

    for d in det:
        diffs = np.abs(ref - d)
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= tol_samples and not matched_ref[min_idx]:
            matched_ref[min_idx] = True
            tp += 1
            abs_errors.append(int(diffs[min_idx]))

    fp = len(det) - tp
    fn = len(ref) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mae_ms = (float(np.mean(abs_errors)) / fs) * 1000.0 if abs_errors else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 5),
        "Recall": round(recall, 5),
        "F1": round(f1, 5),
        "MAE_ms": round(mae_ms, 2),
    }


def validate_records(
    records: List[str],
    raw_dir: str = "data/raw",
    fs: float = 360.0,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    filter_order: int = 4,
    tolerance_ms: float = 50.0,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Запускает фильтрацию, детекцию и расчёт метрик для списка записей MIT-BIH.
    """
    import wfdb

    per_record: Dict[str, Dict[str, float]] = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for rec_id in records:
        rec_path = str(Path(raw_dir) / rec_id)
        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        # Выбираем стандартное отведение MLII, если доступно, иначе канал 0
        ch_idx = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
        raw_signal = record.p_signal[:, ch_idx]
        rec_fs = float(record.fs) if record.fs else fs

        ref_peaks = np.array(
            [s for s, sym in zip(ann.sample, ann.symbol) if sym in ALL_BEAT_SYMBOLS],
            dtype=np.int64,
        )

        clean_signal = preprocess_ecg(
            raw_signal, fs=rec_fs, lowcut=lowcut, highcut=highcut, order=filter_order
        )
        det_peaks = pan_tompkins_detector(clean_signal, fs=rec_fs, is_preprocessed=True)
        metrics = evaluate_r_peak_detector(det_peaks, ref_peaks, fs=rec_fs, tolerance_ms=tolerance_ms)

        per_record[rec_id] = metrics
        total_tp += metrics["TP"]
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (2 * overall_p * overall_r) / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

    summary = {
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "Precision": round(overall_p, 5),
        "Recall": round(overall_r, 5),
        "F1": round(overall_f1, 5),
    }
    return per_record, summary


def plot_detection_check(
    record_id: str,
    raw_dir: str = "data/raw",
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    save_path: Optional[str] = None,
) -> None:
    """
    Визуализация для проверки Definition of Done:
    отображает исходный и отфильтрованный сигнал с маркерами найденных R-пиков и эталона.
    """
    import matplotlib.pyplot as plt
    import wfdb

    rec_path = str(Path(raw_dir) / record_id)
    record = wfdb.rdrecord(rec_path)
    ann = wfdb.rdann(rec_path, "atr")

    ch_idx = record.sig_name.index("MLII") if "MLII" in record.sig_name else 0
    raw_sig = record.p_signal[:, ch_idx]
    fs = float(record.fs)

    clean_sig = preprocess_ecg(raw_sig, fs=fs)
    det_peaks = pan_tompkins_detector(clean_sig, fs=fs, is_preprocessed=True)
    ref_peaks = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in ALL_BEAT_SYMBOLS])

    s_idx = int(start_sec * fs)
    e_idx = int((start_sec + duration_sec) * fs)
    t = np.arange(s_idx, e_idx) / fs

    det_seg = det_peaks[(det_peaks >= s_idx) & (det_peaks < e_idx)]
    ref_seg = ref_peaks[(ref_peaks >= s_idx) & (ref_peaks < e_idx)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(t, raw_sig[s_idx:e_idx], color="gray", linewidth=1.0, label="Raw ECG (MLII)")
    axes[0].set_title(f"Record {record_id}: Исходный сигнал vs Фильтрация 0.5–40 Гц и детекция R-пиков")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, clean_sig[s_idx:e_idx], color="#1f77b4", linewidth=1.1, label="Filtered + Z-norm")
    axes[1].scatter(
        ref_seg / fs, clean_sig[ref_seg], marker="o", s=70, facecolors="none", edgecolors="green", linewidths=1.5, label="Reference (wfdb)"
    )
    axes[1].scatter(
        det_seg / fs, clean_sig[det_seg], marker="x", s=60, color="red", linewidths=1.5, label="Pan-Tompkins R-peaks"
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Z-score")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    sp_cfg = cfg["signal_processing"]

    # Проверяем на первых 3 записях из ds2_test_records (при условии, что данные уже скачаны в data/raw)
    sample_records = data_cfg["ds2_test_records"][:3]
    print(f"[DATA-02] Валидация детектора Пана — Томпкинса на записях: {sample_records}")

    per_rec, summary = validate_records(
        records=sample_records,
        raw_dir=data_cfg["raw_dir"],
        fs=data_cfg["sampling_rate"],
        lowcut=sp_cfg["bandpass_low_hz"],
        highcut=sp_cfg["bandpass_high_hz"],
        filter_order=sp_cfg["filter_order"],
        tolerance_ms=sp_cfg["detector_tolerance_ms"],
    )

    for rec, m in per_rec.items():
        print(f"  Record {rec}: Precision={m['Precision']:.4f}, Recall={m['Recall']:.4f}, MAE={m['MAE_ms']} ms")
    print(f"[DATA-02] Summary: {summary}")