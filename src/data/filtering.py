"""
Фильтрация сигнала ЭКГ, детектор R-пиков Пана — Томпкинса и оценка детекции.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def preprocess_ecg(
    raw_signal: np.ndarray,
    fs: int = 360,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    order: int = 4,
    norm_window_sec: float = 4.0,
) -> np.ndarray:
    """
    Полосовая фильтрация Баттерворта и локальная робастная Z-нормализация
    (скользящее окно 4 с), устойчивая к единичным крупным артефактам.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    filtered = filtfilt(b, a, raw_signal.astype(np.float64))

    win = max(int(norm_window_sec * fs), fs)
    half_w = win // 2
    n = len(filtered)
    normed = np.empty_like(filtered, dtype=np.float32)

    # Блочная оценка медианного масштаба (MAD -> std) для защиты от выбросов
    step = max(win // 2, 1)
    centers = list(range(0, n, step))
    local_means = np.empty(len(centers), dtype=np.float64)
    local_scales = np.empty(len(centers), dtype=np.float64)

    for idx, c in enumerate(centers):
        seg = filtered[max(0, c - half_w) : min(n, c + half_w)]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        robust_std = 1.4826 * mad if mad > 1e-6 else float(np.std(seg))
        local_means[idx] = med
        local_scales[idx] = max(robust_std, 1e-4)

    sample_idx = np.arange(n, dtype=np.float64)
    center_arr = np.asarray(centers, dtype=np.float64)
    interp_mean = np.interp(sample_idx, center_arr, local_means)
    interp_scale = np.interp(sample_idx, center_arr, local_scales)

    normed[:] = np.clip((filtered - interp_mean) / interp_scale, -15.0, 15.0).astype(np.float32)
    return normed


def _refine_peak(
    filtered_signal: np.ndarray,
    approx_idx: int,
    fs: int = 360,
    search_ms: float = 40.0,
    dominant_sign: float = 1.0,
) -> int:
    """
    Уточняет позицию R-пика в узком окне ±40 мс с учётом доминирующей полярности R-зубца,
    чтобы центр окна не съезжал на глубокий Q или S зубец.
    """
    radius = max(int(round(search_ms * fs / 1000.0)), 1)
    start = max(0, approx_idx - radius)
    end = min(len(filtered_signal), approx_idx + radius + 1)
    segment = filtered_signal[start:end]
    if len(segment) == 0:
        return approx_idx

    oriented = segment * dominant_sign
    if np.max(oriented) > 0.35 * np.max(np.abs(segment)):
        return start + int(np.argmax(oriented))
    return start + int(np.argmax(np.abs(segment)))


def pan_tompkins_detector(ecg_signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Адаптивный детектор R-пиков Пана — Томпкинса:
    - корректная ориентированная производная
    - робастная инициализация порогов spki/npki
    - проверка крутизны фронта против высоких T-зубцов
    - корректный search-back с проверкой рефрактерного периода 200 мс.
    """
    n = len(ecg_signal)
    if n < fs:
        return np.array([], dtype=np.int64)

    # Полосовая фильтрация QRS (5–15 Гц) для детекции
    nyq = 0.5 * fs
    b_qrs, a_qrs = butter(2, [5.0 / nyq, 15.0 / nyq], btype="band")
    qrs_filt = filtfilt(b_qrs, a_qrs, ecg_signal.astype(np.float64))

    # Производная: y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
    # В np.convolve ядро переворачивается, поэтому задаём его в обратном порядке
    deriv_kernel = np.array([1.0, 2.0, 0.0, -2.0, -1.0], dtype=np.float64) / 8.0
    diff_sig = np.convolve(qrs_filt, deriv_kernel, mode="same")
    squared_sig = diff_sig ** 2

    mwi_win = max(int(round(0.150 * fs)), 1)
    mwi_sig = np.convolve(squared_sig, np.ones(mwi_win, dtype=np.float64) / mwi_win, mode="same")

    # Определяем доминирующую полярность R-зубца по 95-му и 5-му перцентилям
    p95 = float(np.percentile(ecg_signal, 95))
    p05 = float(np.percentile(ecg_signal, 5))
    dominant_sign = 1.0 if abs(p95) >= abs(p05) else -1.0

    # Робастная инициализация порогов по первым 4 секундам (разбитым на 1-сек окна)
    init_sec = min(4, n // fs)
    sec_maxima = [float(np.max(mwi_sig[s * fs : (s + 1) * fs])) for s in range(init_sec)]
    robust_init_peak = float(np.median(sec_maxima)) if sec_maxima else float(np.percentile(mwi_sig, 95))
    spki = 0.5 * robust_init_peak
    npki = 0.1 * robust_init_peak
    threshold_i1 = npki + 0.25 * (spki - npki)
    threshold_i2 = 0.5 * threshold_i1

    refractory = int(round(0.200 * fs))
    twave_limit = int(round(0.360 * fs))
    min_dist = int(round(0.150 * fs))

    candidate_peaks, _ = find_peaks(mwi_sig, distance=min_dist)
    detected_peaks: List[int] = []
    rr_history: List[int] = []
    last_qrs_slope: float = float(np.percentile(np.abs(diff_sig), 95))

    def _slope_at(idx: int) -> float:
        s = max(0, idx - mwi_win // 2)
        e = min(n, idx + mwi_win // 2 + 1)
        return float(np.max(np.abs(diff_sig[s:e]))) if e > s else 0.0

    for i, pk in enumerate(candidate_peaks):
        peaki = float(mwi_sig[pk])
        cur_slope = _slope_at(pk)

        # 1. Сначала проверяем текущего кандидата pk
        is_qrs = False
        if peaki >= threshold_i1:
            if not detected_peaks:
                is_qrs = True
            else:
                dist_from_last = pk - detected_peaks[-1]
                if dist_from_last < refractory:
                    is_qrs = False
                elif dist_from_last <= twave_limit and cur_slope < 0.5 * last_qrs_slope:
                    # Отсеиваем высокий зубец T по пологому фронту
                    is_qrs = False
                else:
                    is_qrs = True

        # 2. Search-back при длинной паузе (> 1.66 * среднего RR)
        rr_avg = float(np.mean(rr_history[-8:])) if rr_history else float(0.8 * fs)
        search_limit = int(round(1.66 * rr_avg))
        anchor_prev = detected_peaks[-1] if detected_peaks else 0

        if (pk - anchor_prev) > search_limit:
            sb_start = anchor_prev + refractory
            sb_end = pk - refractory if is_qrs else pk
            if sb_end > sb_start:
                # Перебираем все кандидаты внутри пропущенного окна
                in_window = [
                    c for c in candidate_peaks
                    if sb_start <= c < sb_end and float(mwi_sig[c]) >= threshold_i2
                ]
                if in_window:
                    best_sb = max(in_window, key=lambda c: float(mwi_sig[c]))
                    refined_sb = _refine_peak(ecg_signal, int(best_sb), fs=fs, dominant_sign=dominant_sign)
                    # Проверяем рефрактерный период после уточнения с обеих сторон
                    ok_left = (not detected_peaks) or (refined_sb - detected_peaks[-1] >= refractory)
                    ok_right = (not is_qrs) or (pk - refined_sb >= refractory)
                    if ok_left and ok_right:
                        if detected_peaks:
                            rr_history.append(refined_sb - detected_peaks[-1])
                        detected_peaks.append(refined_sb)
                        sb_val = float(mwi_sig[best_sb])
                        spki = 0.25 * sb_val + 0.75 * spki
                        last_qrs_slope = 0.25 * _slope_at(best_sb) + 0.75 * last_qrs_slope
                        threshold_i1 = npki + 0.25 * (spki - npki)
                        threshold_i2 = 0.5 * threshold_i1

        # 3. Фиксируем текущего кандидата pk (как QRS или как шум)
        if is_qrs:
            refined_pk = _refine_peak(ecg_signal, int(pk), fs=fs, dominant_sign=dominant_sign)
            if not detected_peaks or (refined_pk - detected_peaks[-1] >= refractory):
                if detected_peaks:
                    rr_history.append(refined_pk - detected_peaks[-1])
                detected_peaks.append(refined_pk)
                spki = 0.125 * peaki + 0.875 * spki
                last_qrs_slope = 0.125 * cur_slope + 0.875 * last_qrs_slope
            else:
                npki = 0.125 * peaki + 0.875 * npki
        else:
            npki = 0.125 * peaki + 0.875 * npki
            # Мягкая релаксация spki при длительном отсутствии детекций
            if detected_peaks and (pk - detected_peaks[-1]) > int(2.5 * fs):
                spki = max(0.85 * spki, 2.0 * npki)

        threshold_i1 = npki + 0.25 * (spki - npki)
        threshold_i2 = 0.5 * threshold_i1

    return np.asarray(detected_peaks, dtype=np.int64)


def evaluate_r_peak_detector(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    fs: int = 360,
    tolerance_ms: float = 50.0,
) -> Dict[str, float]:
    """
    Сопоставление 1-к-1 между найденными пиками и эталонной разметкой:
    для каждого найденного пика ищется ближайший ещё не занятый эталон в пределах ±tolerance_ms.
    """
    tol_samples = int(round(tolerance_ms * fs / 1000.0))
    matched_ref = np.zeros(len(reference_peaks), dtype=bool)
    tp = 0

    for det in detected_peaks:
        free_indices = np.where(~matched_ref)[0]
        if len(free_indices) == 0:
            break
        diffs = np.abs(reference_peaks[free_indices] - det)
        best_local = int(np.argmin(diffs))
        if diffs[best_local] <= tol_samples:
            matched_ref[free_indices[best_local]] = True
            tp += 1

    fp = int(len(detected_peaks) - tp)
    fn = int(len(reference_peaks) - tp)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * se * ppv / (se + ppv) if (se + ppv) > 0 else 0.0

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Sensitivity": round(se, 4),
        "PPV": round(ppv, 4),
        "F1": round(f1, 4),
    }