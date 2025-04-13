import wfdb
import numpy as np
import pandas as pd
import os
import neurokit2 as nk
from scipy.signal import find_peaks


def get_heart_rate(record):
    
    signal = record.p_signal[:, 0]
    fs = record.fs
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    hr = nk.ecg_rate(rpeaks, sampling_rate=fs)
    return np.mean(hr)

def get_rr_stdM(record, threshold_factor=0.2):
    signal = record.p_signal[:, 0]  # Usamos la primera derivación
    fs = record.fs
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

    # Calcular los intervalos RR (en segundos)
    rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / fs
    
    # Calcular la desviación estándar de los intervalos RR
    rr_std = np.std(rr_intervals)
    
    # Calcular la media de los intervalos RR
    rr_mean = np.mean(rr_intervals)
    
    # Determinar si los intervalos RR son irregulares (basado en la desviación estándar)
    is_irregular = rr_std > threshold_factor * rr_mean  # Considera irregular si la desviación estándar es > 20% de la media
    
    return rr_std, is_irregular

def get_rr_std(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / fs
    return rr_intervals

def get_qrs_durationM(record):
    
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    q_peaks = np.array(delineate[1]["ECG_Q_Peaks"])
    s_peaks = np.array(delineate[1]["ECG_S_Peaks"])
    
    durations = (s_peaks - q_peaks) / fs
    durations = durations[~np.isnan(durations)]
    
    return np.mean(durations) if len(durations) > 0 else np.nan

def get_qrs_features(record, qrs_duration_threshold=120, qrs_amplitude_variation_threshold=0.2):
    """
    Función para extraer las características del complejo QRS, determinar si el QRS es ancho (>120 ms),
    y verificar la alternancia eléctrica (variaciones de amplitud en el QRS).

    Parámetros:
        - file_base: Nombre del archivo base del ECG.
        - qrs_duration_threshold: Umbral de duración del QRS para considerarlo ancho (en ms).
        - qrs_amplitude_variation_threshold: Umbral de variación en la amplitud del QRS para detectar alternancia eléctrica.
        
    Retorna:
        - results: Una lista con las características del complejo QRS, si es ancho, y si hay alternancia eléctrica.
    """

    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []
    
    qrs_amplitude_previous = None  # Para comparar la amplitud del QRS en los latidos consecutivos
    alternancia_electrica_detectada = False  # Flag para la alternancia eléctrica

    for i, r_idx in enumerate(rpeaks["ECG_R_Peaks"]):
        try:
            q_idx = delineate[1]["ECG_Q_Peaks"][i]
            s_idx = delineate[1]["ECG_S_Peaks"][i]
            
            # Duración del QRS (en ms)
            qrs_duration = (s_idx - q_idx) * 1000 / fs  # Duración en milisegundos
            
            # Verificar si el QRS es ancho (>120 ms)
            is_qrs_wide = qrs_duration > qrs_duration_threshold

            # Amplitud relativa (como la amplitud del complejo QRS comparada con la onda T)
            q_value = abs(cleaned[q_idx])
            r_value = abs(cleaned[r_idx])
            s_value = abs(cleaned[s_idx])
            qrs_amplitude = q_value + r_value + s_value  # Amplitud total del QRS (suma de los picos Q, R, y S)

            # Verificar si hay alternancia eléctrica (variaciones significativas en la amplitud entre latidos consecutivos)
            if qrs_amplitude_previous is not None:
                amplitude_variation = abs(qrs_amplitude - qrs_amplitude_previous)
                if amplitude_variation > qrs_amplitude_variation_threshold:
                    alternancia_electrica_detectada = True

            # Almacenar la amplitud para el siguiente ciclo
            qrs_amplitude_previous = qrs_amplitude

            results.append({
                "index_QRS": [i + 1, q_idx, r_idx, s_idx],
                "qrs_duration_ms": qrs_duration,
                "qrs_amplitude": qrs_amplitude,
                "is_qrs_wide": is_qrs_wide,
                "electric_alternance": alternancia_electrica_detectada
            })
        except (IndexError, TypeError):
            continue
    
    return results

def get_qrs_summary(record):
    qrs_features = get_qrs_features(record)

    amplitudes = [t["qrs_amplitude"] for t in qrs_features if "qrs_amplitude" in t]
    durations = [t["qrs_duration_ms"] for t in qrs_features if "qrs_duration_ms" in t]

    mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else None
    mean_duration = sum(durations) / len(durations) if durations else None

    return {
        "mean_QRS_amplitude": mean_amplitude,
        "mean_QRS_duration_ms": mean_duration
    }
    
def get_qt_intervalM(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    q_peaks = np.array(delineate[1]["ECG_Q_Peaks"])
    t_offsets = np.array(delineate[1]["ECG_T_Offsets"])
    
    qt = (t_offsets - q_peaks) / fs
    qt = qt[~np.isnan(qt)]
    
    return np.mean(qt) if len(qt) > 0 else np.nan

def get_qt_interval(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    q_peaks = np.array(delineate[1]["ECG_Q_Peaks"])
    t_offsets = np.array(delineate[1]["ECG_T_Offsets"])
    
    qt = (t_offsets - q_peaks) / fs
    
    
    return qt if len(qt) > 0 else np.nan

def get_qtc_bazett(record, threshold_ms=500):
    """
    Calcula el QTc (corregido por Bazett) para cada complejo y cuenta los que superan el umbral.
    
    Parámetros:
        - file_base: nombre base del archivo ECG.
        - threshold_ms: umbral en milisegundos para considerar QTc prolongado (por defecto 480 ms).
    
    Retorna:
        - qtc_values: lista de QTc por latido (en ms).
        - count_prolonged: número de QTc > threshold.
    """
    
    signal = record.p_signal[:, 0]
    fs = record.fs

    # QT en segundos por latido
    qt_intervals = get_qt_interval(record)  # debe retornar una lista de QT en segundos
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    
    # RR por latido (mismo largo que QT menos 1)
    r_locs = rpeaks["ECG_R_Peaks"]
    rr_intervals = np.diff(r_locs) / fs  # en segundos
    
    # Asegurar que las longitudes coincidan para el pareo
    min_len = min(len(qt_intervals), len(rr_intervals))
    qt_intervals = qt_intervals[:min_len]
    rr_intervals = rr_intervals[:min_len]
    
    # Calcular QTc latido a latido
    qtc_values = [(qt / np.sqrt(rr)) * 1000 for qt, rr in zip(qt_intervals, rr_intervals)]  # en ms
    count_prolonged = sum(1 for qtc in qtc_values if qtc > threshold_ms)
    
    return {
        "QTc_values_ms": qtc_values,
        "QTc_prolonged_count": count_prolonged
    }
    
def get_p_durationM(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    p_onsets = np.array(delineate[1]["ECG_P_Onsets"])
    p_offsets = np.array(delineate[1]["ECG_P_Offsets"])
    
    durations = (p_offsets - p_onsets) / fs
    durations = durations[~np.isnan(durations)]
    
    return np.mean(durations) if len(durations) > 0 else np.nan

def get_p_duration(record):
    
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    p_onsets = np.array(delineate[1]["ECG_P_Onsets"])
    p_offsets = np.array(delineate[1]["ECG_P_Offsets"])
    
    durations = (p_offsets - p_onsets) / fs
    durations = durations[~np.isnan(durations)]
    
    return durations if len(durations) > 0 else np.nan

def get_pr_intervalM(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    p_onsets = np.array(delineate[1]["ECG_P_Onsets"])
    q_peaks = np.array(delineate[1]["ECG_Q_Peaks"])
    
    pr = (q_peaks - p_onsets) / fs
    pr = pr[~np.isnan(pr)]
    
    return np.mean(pr) if len(pr) > 0 else np.nan

def get_pr_interval(record):
    signal = record.p_signal[:, 0]  # Usamos todas las derivaciones
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    pr_intervals = []
    pr_depression_inferior = False

    p_onsets = delineate[1]["ECG_P_Onsets"]
    q_peaks = delineate[1]["ECG_Q_Peaks"]

    # Identificar índices de derivaciones inferiores
    lead_names = record.sig_name
    inferior_leads = ["II", "III", "aVF"]
    inferior_indices = [i for i, name in enumerate(lead_names) if name in inferior_leads]

    for i in range(min(len(p_onsets), len(q_peaks))):
        try:
            onset = p_onsets[i]
            q_peak = q_peaks[i]

            if onset is None or q_peak is None or q_peak <= onset:
                continue

            # Calcular el intervalo PR (en segundos)
            pr = (q_peak - onset) / fs
            pr_intervals.append(pr)

            # Revisar si hay depresión del PR en derivaciones inferiores
            for idx in inferior_indices:
                pr_segment = cleaned[onset:q_peak, idx]
                baseline = cleaned[onset - 40:onset, idx] if onset >= 40 else cleaned[:onset, idx]
                baseline_mean = np.mean(baseline)
                pr_mean = np.mean(pr_segment)

                if (baseline_mean - pr_mean) > 0.1:  # 0.1 mV de depresión
                    pr_depression_inferior = True
                    break  # Con que ocurra una vez, basta

        except (IndexError, TypeError):
            continue

    return {
        "pr_intervals_s": pr_intervals if len(pr_intervals) > 0 else np.nan,
        "pr_depression_in_inferior_leads": pr_depression_inferior
    }
    
def get_q_wave_features(record):

    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []
    for i, r_idx in enumerate(rpeaks["ECG_R_Peaks"]):
        try:
            q_idx = delineate[1]["ECG_Q_Peaks"][i]

            q_value = abs(cleaned[q_idx])
            r_value = abs(cleaned[r_idx])

            duration = (r_idx - q_idx) * 1000 / fs  # en milisegundos
            relative_amplitude = q_value / r_value if r_value != 0 else 0

            # Identificación de onda Q patológica:
            q_pathological = False
            # Amplitud Q patológica: > 0.3 mV (300 uV)
            if q_value > 0.3:
                q_pathological = True
            # Duración Q patológica: > 40 ms (0.04 segundos)
            elif duration > 40:
                q_pathological = True

            # Añadir la información de la onda Q patológica a los resultados
            results.append({
                "index_Q": [i + 1, q_idx],
                "duration_Q_ms": duration,
                "Q_amplitude": q_value,
                "relative_amplitude_Q/R": relative_amplitude,
                "Q_pathological": q_pathological
            })
        except (IndexError, TypeError):
            continue

    return results

def get_q_wave_summary(record):
    q_features = get_q_wave_features(record)

    amplitudes = [q["Q_amplitude"] for q in q_features if "Q_amplitude" in q]
    durations = [q["duration_Q_ms"] for q in q_features if "duration_Q_ms" in q]

    mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else None
    mean_duration = sum(durations) / len(durations) if durations else None

    return {
        "mean_Q_amplitude": mean_amplitude,
        "mean_Q_duration_ms": mean_duration
    }

def get_p_wave_features(record, p_amplitude_threshold=0.1, p_duration_threshold=150):
    """
    Función para extraer las características de la onda P y determinar si está ausente o alterada.

    Parámetros:
        - file_base: Nombre del archivo base del ECG.
        - p_amplitude_threshold: Umbral de la amplitud de la onda P para considerar que es anómala.
        - p_duration_threshold: Umbral de la duración de la onda P en ms para considerar que es anómala.
        
    Retorna:
        - results: Una lista con las características de las ondas P y si son alteradas o ausentes.
    """
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []
    p_missing = False
    for i in range(len(delineate[1]["ECG_P_Peaks"])):
        try:
            p_idx = delineate[1]["ECG_P_Peaks"][i]
            onset_p = delineate[1]["ECG_P_Onsets"][i]
            offset_p = delineate[1]["ECG_P_Offsets"][i]
            p_value = abs(cleaned[p_idx])

            # Duración de la onda P en ms
            duration_p = (offset_p - onset_p) * 1000 / fs

            # Amplitud relativa (como la amplitud de la onda P comparada con la onda R)
            r_idx = rpeaks["ECG_R_Peaks"][i]
            r_value = abs(cleaned[r_idx]) if cleaned[r_idx] != 0 else 1e-6
            relative_amplitude_p = p_value / r_value

            # Verificar si la onda P es alterada (amplitud o duración fuera de umbrales)
            p_is_altered = (p_value < p_amplitude_threshold) or (duration_p > p_duration_threshold)

            results.append({
                "index_P": [i + 1, delineate[1]["ECG_P_Peaks"][i]],
                "duration_P_ms": duration_p,
                "P_amplitude": p_value,
                "relative_amplitude_P/R": relative_amplitude_p,
                "p_is_altered": p_is_altered
            })
        except (IndexError, TypeError):
            p_missing = True
            continue

    # Si no se encontró ninguna onda P, marcamos como ausente
    if p_missing:
        results.append({"p_missing": True})

    return results

def get_p_wave_summary(record):
    p_features = get_p_wave_features(record)

    amplitudes = []
    durations = []
    num_alteradas = 0
    p_missing = False

    for p in p_features:
        if "p_missing" in p and p["p_missing"]:
            p_missing = True
            continue
        if "P_amplitude" in p:
            amplitudes.append(p["P_amplitude"])
        if "duration_P_ms" in p:
            durations.append(p["duration_P_ms"])
        if p.get("p_is_altered"):
            num_alteradas += 1

    mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else None
    mean_duration = sum(durations) / len(durations) if durations else None

    return {
        "mean_P_amplitude": mean_amplitude,
        "mean_P_duration_ms": mean_duration,
        "num_P_altered": num_alteradas,
        "p_missing": p_missing
    }

def get_r_wave_features(record, threshold_r_amplitude=0.5, r_v1_threshold=7):
    """
    Función para extraer las características de la onda R y detectar ondas R altas en las derivaciones precordiales (V1-V6).
    Además, se verifica si la amplitud de R en V1 es mayor a 7 mm.
    
    Parámetros:
        - file_base: Nombre del archivo base del ECG.
        - threshold_r_amplitude: Umbral de amplitud para considerar una onda R como "alta" (en milivoltios).
        - r_v1_threshold: Umbral de amplitud para considerar la onda R en V1 como "alta" (en milímetros).
    
    Retorna:
        - results: Un diccionario con las características de la onda R y la indicación de si hay ondas R altas en las derivaciones precordiales.
    """
    # Cargar los datos del ECG
    
    signal = record.p_signal  # La señal es de varias derivaciones
    fs = record.fs  # Frecuencia de muestreo
    
    # Limpiar la señal
    cleaned = nk.ecg_clean(signal[:, 0], sampling_rate=fs)
    
    # Detectar los picos R
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    
    # Delimitar las derivaciones precordiales (V1 a V6)
    v1_signal = signal[:, 0]  # Derivación V1
    v2_signal = signal[:, 1]  # Derivación V2
    v3_signal = signal[:, 2]  # Derivación V3
    v4_signal = signal[:, 3]  # Derivación V4
    v5_signal = signal[:, 4]  # Derivación V5
    v6_signal = signal[:, 5]  # Derivación V6
    
    # Delimitar las ondas R
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    results = []
    for i in range(len(delineate[1]["ECG_R_Onsets"])):
        try:
            r_idx = rpeaks["ECG_R_Peaks"][i]
            onset_r = delineate[1]["ECG_R_Onsets"][i]
            offset_r = delineate[1]["ECG_R_Offsets"][i]
            r_value = abs(cleaned[r_idx])

            # Duración de la onda R en milisegundos
            duration_r = (offset_r - onset_r) * 1000 / fs

            # Detectar si la onda R es alta en V1-V6
            r_amplitude_v1 = abs(v1_signal[r_idx])
            r_amplitude_v2 = abs(v2_signal[r_idx])
            r_amplitude_v3 = abs(v3_signal[r_idx])
            r_amplitude_v4 = abs(v4_signal[r_idx])
            r_amplitude_v5 = abs(v5_signal[r_idx])
            r_amplitude_v6 = abs(v6_signal[r_idx])
            
            # Comprobar si alguna onda R es alta (supera el umbral en alguna derivación precordial)
            r_high = any(amplitude > threshold_r_amplitude for amplitude in [r_amplitude_v1, r_amplitude_v2, r_amplitude_v3, r_amplitude_v4, r_amplitude_v5, r_amplitude_v6])
            
            # Verificar si la amplitud de R en V1 es mayor a 7 mm
            r_v1_high = r_amplitude_v1 > r_v1_threshold  # En milímetros, ya que la señal está en milivoltios

            # Añadir los resultados
            results.append({
                "index_R": [i + 1, r_idx],
                "duration_R_ms": duration_r,
                "R_amplitude": r_value,
                "R_high_in_V1_V6": r_high,
                "RV1high": r_v1_high
            })
        except (IndexError, TypeError):
            continue
    return results

def get_r_wave_summary(record):
    r_features = get_r_wave_features(record)

    amplitudes = [r["R_amplitude"] for r in r_features if "R_amplitude" in r]
    durations = [r["duration_R_ms"] for r in r_features if "duration_R_ms" in r]

    mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else None
    mean_duration = sum(durations) / len(durations) if durations else None

    return {
        "mean_R_amplitude": mean_amplitude,
        "mean_R_duration_ms": mean_duration
    }

def get_s_wave_features(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []
    for i, r_idx in enumerate(rpeaks["ECG_R_Peaks"]):
        try:
            s_idx = delineate[1]["ECG_S_Peaks"][i]
            s_value = abs(cleaned[s_idx])
            r_value = abs(cleaned[r_idx])

            # Duración de la onda S en ms
            duration_s = (s_idx - r_idx) * 1000 / fs
            # Amplitud relativa (como la amplitud de la onda S comparada con la onda R)
            relative_amplitude_s = s_value / r_value if r_value != 0 else 0

            results.append({
                "index_S": [i + 1, s_idx],
                "duration_S_ms": duration_s,
                "S_amplitude": s_value,
                "relative_amplitude_S/R": relative_amplitude_s
            })
        except (IndexError, TypeError):
            continue
    return results

def get_s_wave_summary(record):
    s_features = get_s_wave_features(record)

    amplitudes = [s["S_amplitude"] for s in s_features if "S_amplitude" in s]
    durations = [s["duration_S_ms"] for s in s_features if "duration_S_ms" in s]

    mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else None
    mean_duration = sum(durations) / len(durations) if durations else None

    return {
        "mean_S_amplitude": mean_amplitude,
        "mean_S_duration_ms": mean_duration
    }

def get_t_wave_features(record):
    
    signal = record.p_signal[:, 0]  # Usamos la primera derivación para simplificar
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []

    for i, r_idx in enumerate(delineate[1]["ECG_S_Peaks"]):
        try:
            onset_t = delineate[1]["ECG_T_Onsets"][i]
            offset_t = delineate[1]["ECG_T_Offsets"][i]
            t_idx = delineate[1]["ECG_T_Peaks"][i]
            t_value = abs(cleaned[t_idx])
            r_value = abs(cleaned[r_idx])
            duration_t = (offset_t - onset_t) * 1000 / fs
            relative_amplitude_t = t_value / r_value if r_value != 0 else 0
            t_inverted = cleaned[t_idx] < 0

            # Negatividad en V1 o V2
            t_negative_right_precordial = False
            if record.sig_name[0] in ["V1", "V2"] and cleaned[t_idx] < 0:
                t_negative_right_precordial = True

            # Detección de T bífida: buscar 2 picos dentro del segmento T
            t_segment = cleaned[onset_t:offset_t]
            peaks, _ = find_peaks(t_segment, distance=int(0.05 * fs))  # al menos 50 ms entre picos
            t_bifid = len(peaks) >= 2

            results.append({
                "index_T": [i + 1, t_idx],
                "duration_T_ms": duration_t,
                "T_amplitude": t_value,
                "relative_amplitude_T/R": relative_amplitude_t,
                "T_inverted": t_inverted,
                "T_negative_right_precordial": t_negative_right_precordial,
                "T_bifid": t_bifid
            })
        except (IndexError, TypeError):
            continue

    return results

def get_t_wave_summary(record):
    t_features = get_t_wave_features(record)

    amplitudes = [t["T_amplitude"] for t in t_features if "T_amplitude" in t]
    durations = [t["duration_T_ms"] for t in t_features if "duration_t_ms" in t]

    mean_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else None
    mean_duration = sum(durations) / len(durations) if durations else None

    return {
        "mean_T_amplitude": mean_amplitude,
        "mean_T_duration_ms": mean_duration
    }

def get_st_segment_features(record):
    
    signals = record.p_signal
    fs = record.fs
    leads = record.sig_name

    # Usamos una derivación general para detectar los picos
    signal = signals[:, 0]
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []

    for i, r_idx in enumerate(rpeaks["ECG_R_Peaks"]):
        try:
            # Estimar inicio del ST (preferimos R_Offset, si no usamos S_Peak)
            st_start = delineate[1]["ECG_R_Offsets"][i]
            if st_start is None:
                st_start = delineate[1]["ECG_S_Peaks"][i]
            st_end = delineate[1]["ECG_T_Onsets"][i]

            if st_start is None or st_end is None or st_end <= st_start:
                continue

            elevated_segment_count = 0
            contiguous_elevated = []
            st_elevated_v1_v3 = False
            brugada_type1 = False

            for ch_idx, lead in enumerate(leads):
                signal = signals[:, ch_idx]

                if st_end >= len(signal):
                    continue

                st_segment = signal[st_start:st_end]
                st_mean = np.mean(st_segment)

                # Contamos cada segmento ST elevado
                if st_mean > 0.1:
                    elevated_segment_count += 1
                    contiguous_elevated.append(lead)

                if lead in ["V1", "V2", "V3"] and st_mean > 0.2:
                    st_elevated_v1_v3 = True

                if lead in ["V1", "V2"]:
                    window = signal[st_start-20:st_end+20]
                    if len(window) > 1 and np.all(np.diff(window) < 0):
                        brugada_type1 = True

            diffuse_st_elevation = elevated_segment_count >= 6

            results.append({
                "index_R": [i + 1, r_idx],
                "ST_elevation_contiguous": len(contiguous_elevated) >= 2,
                "elevated_segment_count": elevated_segment_count,
                "Elevated_V1_V3": st_elevated_v1_v3,
                "Type1": brugada_type1,
                "DiffuseSTElevation": diffuse_st_elevation
            })

        except (IndexError, TypeError, KeyError):
            continue

    return results

def get_brd_bri_patterns(record, threshold_s_depth=0.2, threshold_r_prime_amplitude=0.1):
    """
    Función para detectar los patrones de RSR' en V1-V2 (bloqueo de rama derecha) y la onda S profunda en I, V5-V6 (bloqueo de rama izquierda).
    
    Parámetros:
        - file_base: Nombre del archivo base del ECG.
        - threshold_s_depth: Umbral para considerar que una onda S es profunda.
        - threshold_r_prime_amplitude: Umbral para detectar una onda R' significativa.
    
    Retorna:
        - results: Un diccionario con los patrones encontrados (RSR' en V1-V2 y onda S profunda en I, V5-V6).
    """
    signal = record.p_signal
    fs = record.fs
    cleaned = nk.ecg_clean(signal[:, 0], sampling_rate=fs)
    
    # Encontrar los picos R para derivaciones
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    
    # Delimitar las derivaciones relevantes por nombre
    v1_signal = signal[:, record.sig_name.index('v1')]  # Usamos el nombre de la derivación 'v1'
    v5_signal = signal[:, record.sig_name.index('v5')]  # Usamos el nombre de la derivación 'v5'
    v6_signal = signal[:, record.sig_name.index('v6')]  # Usamos el nombre de la derivación 'v6'
    
    # Detectar el patrón RSR' en V1 (Bloqueo de Rama Derecha)
    brd_pattern = detect_rsr_pattern(v1_signal, fs, threshold_r_prime_amplitude)
    
    # Detectar la onda S profunda en V5-V6 (Bloqueo de Rama Izquierda)
    bri_pattern = detect_s_wave_depth(v5_signal, v6_signal, fs, threshold_s_depth)
    
    results = {
        "brd_pattern": brd_pattern,
        "bri_pattern": bri_pattern
    }
    
    return results

def detect_rsr_pattern(signal, fs, threshold_r_prime_amplitude):
    """
    Detecta el patrón RSR' en V1-V2, característico de un Bloqueo de Rama Derecha.
    
    Parámetros:
        - signal: La señal ECG en V1 o V2.
        - fs: La frecuencia de muestreo.
        - threshold_r_prime_amplitude: Umbral de amplitud para considerar que una R' es significativa.
    
    Retorna:
        - brd_detected: True si el patrón RSR' se detecta, False de lo contrario.
    """
    
    # Asegurarse de que la señal no esté vacía antes de limpiar
    if signal.size == 0:
        raise ValueError("La señal proporcionada está vacía.")
    
    # Asegurarse de que fs (frecuencia de muestreo) sea un valor válido
    if fs <= 0:
        raise ValueError(f"La frecuencia de muestreo 'fs' no es válida: {fs}")
    
    # Limpiar la señal ECG
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    except Exception as e:
        raise RuntimeError(f"Error al limpiar la señal: {e}")
    
    # Encontrar los picos R
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    
    # Buscar los picos R, S y R' en la señal de V1
    try:
        delineate = nk.ecg_delineate(signal, sampling_rate=fs, method="dwt", show=False)
    except Exception as e:
        raise RuntimeError(f"Error al delinear la señal ECG: {e}")
    
    r_peaks = rpeaks["ECG_R_Peaks"]  # Esto es un array de índices
    s_peaks = delineate[1]["ECG_S_Peaks"]  # Esto es un array de índices
    
    brd_detected = False
    min_len = min(len(r_peaks) - 1, len(s_peaks))  # -1 para evitar IndexError en r_peaks[i + 1]
    for i in range(1, min_len):  # Verificar en cada complejo
        r_idx = r_peaks[i]
        s_idx = s_peaks[i]
        
        # Detectar una onda R seguida de una onda S y luego una R'
        if r_idx > s_idx and r_idx - s_idx > 30:  # Comprobar que haya suficiente espacio entre R y S
            try:
                r_prime_idx = r_peaks[i + 1]  # Detectar la R' (el siguiente pico R)
                if abs(signal[r_prime_idx]) > threshold_r_prime_amplitude:  # Comprobar que R' sea significativo
                    brd_detected = True
                    break
            except IndexError:
                continue
    
    return brd_detected

def detect_s_wave_depth(v5_signal, v6_signal, fs, threshold_s_depth):
    """
    Detecta una onda S profunda en las derivaciones I, V5, V6, característico de un Bloqueo de Rama Izquierda.
    
    Parámetros:
        - v5_signal: La señal ECG en la derivación V5.
        - v6_signal: La señal ECG en la derivación V6.
        - fs: La frecuencia de muestreo.
        - threshold_s_depth: Umbral para detectar una onda S profunda.
    
    Retorna:
        - bri_detected: True si se detecta una onda S profunda, False de lo contrario.
    """
    # Buscar los picos S en V5 y V6
    delineate_v5 = nk.ecg_delineate(v5_signal, sampling_rate=fs, method="dwt", show=False)
    delineate_v6 = nk.ecg_delineate(v6_signal, sampling_rate=fs, method="dwt", show=False)
    
    s_peaks_v5 = delineate_v5[1]["ECG_S_Peaks"]
    s_peaks_v6 = delineate_v6[1]["ECG_S_Peaks"]
    
    bri_detected = False
    
    # Asegurarse de que los picos S en ambas derivaciones estén alineados
    min_len = min(len(s_peaks_v5), len(s_peaks_v6))
    
    for i in range(min_len):
        s_v5 = s_peaks_v5[i]
        s_v6 = s_peaks_v6[i]
        
        # Asegurarse de que los índices son válidos y dentro del rango
        if s_v5 < len(v5_signal) and s_v6 < len(v6_signal):
            if abs(v5_signal[s_v5]) < -threshold_s_depth and abs(v6_signal[s_v6]) < -threshold_s_depth:
                bri_detected = True
                break
    
    return bri_detected

def get_sokolow_lyon_index(record, threshold=35):
    """
    Calcula el índice de Sokolow-Lyon: S(V1) + R(V5 o V6).
    Se considera positivo si es mayor al umbral (por defecto 35 mm = 3.5 mV).

    Parámetros:
        - record: Objeto WFDB leído con wfdb.rdrecord(), que contiene el ECG multicanal.
        - threshold: Umbral en mm para considerar el índice positivo (default: 35 mm).

    Retorna:
        - results: Diccionario con amplitudes y si el índice es positivo.
    """
    signal = record.p_signal  # Señales multicanal
    fs = record.fs  # Frecuencia de muestreo

    # Obtener índices de derivaciones V1, V5 y V6
    try:
        idx_v1 = record.sig_name.index('v1')
        idx_v5 = record.sig_name.index('v5')
        idx_v6 = record.sig_name.index('v6')
    except ValueError as e:
        raise ValueError(f"No se encontraron derivaciones necesarias en el registro: {e}")

    # Extraer señales individuales
    v1_signal = signal[:, idx_v1]
    v5_signal = signal[:, idx_v5]
    v6_signal = signal[:, idx_v6]

    # Limpiar una señal para detección de R peaks (puede ser V5 o cualquier canal estable)
    cleaned = nk.ecg_clean(v5_signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)

    # Delineado en V1 para obtener picos S
    try:
        delineate_v1 = nk.ecg_delineate(v1_signal, rpeaks["ECG_R_Peaks"], sampling_rate=fs, method="dwt", show=False)
    except IndexError:
        delineate_v1 = nk.ecg_delineate(v1_signal, rpeaks["ECG_R_Peaks"], sampling_rate=fs, method="peak", show=False)

    # Extraer picos
    s_peaks_v1 = delineate_v1[1].get("ECG_S_Peaks", [])
    r_peaks = rpeaks["ECG_R_Peaks"]

    # Asegurar que los índices estén dentro del tamaño de cada señal
    s_amplitudes_v1 = [abs(v1_signal[i]) for i in s_peaks_v1 if i < len(v1_signal)]
    r_amplitudes_v5 = [abs(v5_signal[i]) for i in r_peaks if i < len(v5_signal)]
    r_amplitudes_v6 = [abs(v6_signal[i]) for i in r_peaks if i < len(v6_signal)]

    # Calcular máximas amplitudes
    s_amplitude_v1 = max(s_amplitudes_v1) if s_amplitudes_v1 else 0
    r_amplitude_v5 = max(r_amplitudes_v5) if r_amplitudes_v5 else 0
    r_amplitude_v6 = max(r_amplitudes_v6) if r_amplitudes_v6 else 0

    # Índice de Sokolow-Lyon = S en V1 + mayor R en V5/V6
    r_amplitude_max = max(r_amplitude_v5, r_amplitude_v6)
    sokolow_lyon_index = s_amplitude_v1 + r_amplitude_max

    # Evaluar si es positivo según el umbral
    is_positive = sokolow_lyon_index > threshold

    results = {
        "S_in_V1_amplitude_mm": s_amplitude_v1,
        "R_in_V5_amplitude_mm": r_amplitude_v5,
        "R_in_V6_amplitude_mm": r_amplitude_v6,
        "Sokolow_Lyon_index_mm": sokolow_lyon_index,
        "Index_positive": is_positive
    }

    return results

def extract_features(patient_id, file):
    features={}
    basePath = 'ECG_Database'
    file_path = os.path.join(basePath, patient_id, file)
    file_base = os.path.splitext(file_path)[0]
    record = wfdb.rdrecord(file_base)
    
    heart_rate=get_heart_rate(record)
    rrM=get_rr_stdM(record)
    rr=get_rr_std(record)
    qrsM=get_qrs_durationM(record) 
    qrs=get_qrs_features(record)
    qrsA=get_qrs_summary(record)["mean_QRS_amplitude"]   
    qtM=get_qt_intervalM(record)
    qt=get_qt_interval(record)
    qtc=get_qtc_bazett(record)
    prM=get_pr_intervalM(record)
    pr=get_pr_interval(record)
    q=get_q_wave_features(record)
    qA=get_q_wave_summary(record)["mean_Q_amplitude"]
    qD=get_q_wave_summary(record)["mean_Q_duration_ms"]
    p=get_p_wave_features(record)
    pA=get_p_wave_summary(record)["mean_P_amplitude"]
    pD=get_p_wave_summary(record)["mean_P_duration_ms"]
    r=get_r_wave_features(record)
    rA=get_r_wave_summary(record)["mean_R_amplitude"]
    rD=get_r_wave_summary(record)["mean_R_duration_ms"]
    s=get_s_wave_features(record)
    sA=get_s_wave_summary(record)["mean_S_amplitude"]
    sD=get_s_wave_summary(record)["mean_S_duration_ms"]
    t=get_t_wave_features(record)
    tA=get_t_wave_summary(record)["mean_T_amplitude"]
    tD=get_t_wave_summary(record)["mean_T_duration_ms"]
    st=get_st_segment_features(record)
    br=get_brd_bri_patterns(record) 
    sokolow=get_sokolow_lyon_index(record)
    

    
    record_features={"heart_rate":heart_rate, "rrM":rrM, "rr":rr, "qrsM":qrsM, "qrs":qrs, "qtM":qtM, "qt":qt, "qtc":qtc, "prM":prM, "pr":pr, "q_wave":q, "qA":qA, "qD":qD, "p_wave":p, "pA":pA, "pD":pD,"r_wave":r, "rA":rA, "rD":rD, "s_wave":s, "sA":sA, "sD":sD,"t_wave":t, "tA":tA, "tD":tD, "st_segment":st, "br": br, "sokolow":sokolow} 
# Almacenar las características
    features[patient_id]={file:record_features}

    return features

def possible_infarct(patient_id, archivo, caracteristicas):
    patient_features = caracteristicas[patient_id]

# Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

# Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    st_results = archivo_features["st_segment"]

# Ahora, 'st_results' es una lista de diccionarios con los resultados de ST, por lo que puedes acceder a la característica 'ST_elevation_contiguous' así:
    st_elevated_contiguous=False
    for result in st_results:
        st_elevation = result["ST_elevation_contiguous"]
        if st_elevation==True:
            st_elevated_contiguous=True
            break
    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    t_wave = archivo_features["t_wave"]

    # Ahora, 'st_results' es una lista de diccionarios con los resultados de ST, por lo que puedes acceder a la característica 'ST_elevation_contiguous' así:
    t_inverted=False
    for result in t_wave:
        t = result["T_inverted"]
        if t==True:
            t_inverted=True
            break
    
    q_wave =  archivo_features["q_wave"]
    q_pathological=False
    for result in q_wave:
        q = result["Q_pathological"]
        if q==True:
            q_pathological=True
            break
    
    qrs =  archivo_features["qrs"]
    qrs_wide=False
    for result in qrs:
        qrs_= result["is_qrs_wide"]
        if qrs_==True:
            qrs_wide=True
            break
    

    possible=0
    if(st_elevated_contiguous):
        possible=possible+50
    if(t_inverted):
        possible=possible+18
    if(q_pathological):
        possible=possible+22
    if(qrs_wide):
        possible=possible+10
    
    return possible

def possible_arrhythmia(patient_id, archivo, caracteristicas):
    
    rr_irregular = caracteristicas[patient_id][archivo]["rrM"][1]
    patient_features = caracteristicas[patient_id]

    # Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    p_results = archivo_features["p_wave"]

    # Ahora, 'st_results' es una lista de diccionarios con los resultados de ST, por lo que puedes acceder a la característica 'ST_elevation_contiguous' así:
    p_altered=False
    for result in p_results:
        p = result["p_is_altered"]
        if p==True:
            p_altered=True
            break
    
    bad_HR = caracteristicas[patient_id][archivo]["heart_rate"]>150
    pr_prolonged=caracteristicas[patient_id][archivo]["prM"]>200

    possible=0
    if(rr_irregular):
        possible=possible+40
    if(p_altered):
        possible=possible+25
    if(bad_HR):
        possible=possible+20
    if(pr_prolonged):
        possible=possible+15
    
    return possible

def possible_branch_block(patient_id, archivo, caracteristicas):
    
    patient_features = caracteristicas[patient_id]

    # Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    qrs_results = archivo_features["qrs"]

    # Ahora, 'st_results' es una lista de diccionarios con los resultados de ST, por lo que puedes acceder a la característica 'ST_elevation_contiguous' así:
    qrs_wide=False
    for result in qrs_results:
        qrs = result["is_qrs_wide"]
        if qrs==True:
            qrs_wide=True
            break

    if (caracteristicas[patient_id][archivo]["br"]['brd_pattern']):
        bb_pattern=True
    else:
        bb_pattern=caracteristicas[patient_id][archivo]["br"]['bri_pattern']

    possible=0
    if(qrs_wide):
        possible=possible+60
    if(bb_pattern):
        possible=possible+40
    
    possible=min(possible, 100)
    
    return possible

def possible_ventricular_hypertrophy(patient_id, archivo, caracteristicas):
    
    patient_features = caracteristicas[patient_id]

    # Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    r_results = archivo_features["r_wave"]

    # Ahora, 'st_results' es una lista de diccionarios con los resultados de ST, por lo que puedes acceder a la característica 'ST_elevation_contiguous' así:
    r_high=False
    for result in r_results:
        r = result["R_high_in_V1_V6"]
        if r==True:
            r_high=True
            break
    rV1_high=False
    for result in r_results:
        r = result["RV1high"]
        if r==True:
            rV1_high=True
            break
    
    sokolow_positive =   caracteristicas[patient_id][archivo]["sokolow"]["Index_positive"]
    
    

    possible=0
    if(r_high):
        possible=possible=40

    if(rV1_high):
        possible=possible+40
    
    if(sokolow_positive): 
        possible=possible+20
    
    
    return possible

def possible_long_QT_syndrome(patient_id, archivo, caracteristicas):

    qt = caracteristicas[patient_id][archivo]["qtc"]["QTc_prolonged_count"]
    qt_prolonged=False
    if(qt>0):
        qt_prolonged=True
    
    patient_features = caracteristicas[patient_id]

    # Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    t_results = archivo_features["t_wave"]
    
    t_bifid=False
    
    for result in t_results:
        r = result["T_bifid"]
        if r==True:
            t_bifid=True
            break
    possible=0
    if(qt_prolonged):
        possible=possible+60

    if(t_bifid):
        possible=possible+40
   
    
    return possible

def possible_brugada_syndrome(patient_id, archivo, caracteristicas):
    patient_features = caracteristicas[patient_id]

    # Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    st_results = archivo_features["st_segment"]

    st_elevated=False
    
    for result in st_results:
        r = result["Elevated_V1_V3"]
        if r==True:
            st_elevated=True
            break
    
    pr=caracteristicas[patient_id][archivo]["pr"]["pr_depression_in_inferior_leads"]

    qrs_results = archivo_features["qrs"]

    electric=False
    
    for result in qrs_results:
        r = result["electric_alternance"]
        if r==True:
            electric=True
            break

    possible=0
    if(st_elevated):
        possible=possible+55

    if(pr):
        possible=possible+30
    
    if(electric):
        possible=possible+15
    possible=min(possible, 100)
   
    
    return possible

def possible_pericarditis(patient_id, archivo, caracteristicas):

    patient_features = caracteristicas[patient_id]

    # Acceder a las características específicas del archivo
    archivo_features = patient_features[archivo]

    # Acceder a la lista de resultados de ST (que contiene diccionarios con las características ST)
    st_results = archivo_features["st_segment"]

    st_elevated=False
    
    for result in st_results:
        r = result["DiffuseSTElevation"]
        if r==True:
            st_elevated=True
            break

    t_results = archivo_features["t_wave"]

    t_negative=False
    
    for result in t_results:
        r = result["T_negative_right_precordial"]
        if r==True:
            t_negative=True
            break

    type1=False
    
    for result in st_results:
        r = result["Type1"]
        if r==True:
            type1=True
            break

    
    
    

    possible=0
    if(st_elevated):
        possible=possible+50

    if(t_negative):
        possible=possible+30
    
    if(type1):
        possible=possible+20
    possible=min(possible, 100)
   
    
    return possible

def get_disease_df(allFeatures):   
    diseases = {
        "arrhythmia": possible_arrhythmia,
        "branch_block": possible_branch_block,
        "brugada_syndrome": possible_brugada_syndrome,
        "infarct": possible_infarct,
        "long_QT_syndrome": possible_long_QT_syndrome,
        "pericarditis": possible_pericarditis,
        "ventricular_hypertrophy": possible_ventricular_hypertrophy
    }

    # Lista donde guardaremos los resultados
    rows = []

    # Recorrer todos los pacientes y señales
    for patient_id in allFeatures:
        for archivo in allFeatures[patient_id]:
            row = {
                "Patient": patient_id,
                "File": archivo
            }
            
            # Calcular todos los scores
            scores = {}
            for disease, func in diseases.items():
                try:
                    score = func(patient_id, archivo, allFeatures)
                except:
                    score = 0
                scores[disease] = score
                row[disease] = score
            
            # Obtener la enfermedad con el puntaje más alto
            max_disease = max(scores, key=scores.get)
            row["Max_Label"] = max_disease
            
            

            rows.append(row)

    # Crear el DataFrame final
    df = pd.DataFrame(rows)
    
    return df

def features_dict_to_df(features_dict):
    rows = []
    
    for patient_id, archivos in features_dict.items():
        for archivo, feature_dict in archivos.items():
            row = {
                "Patient": patient_id,
                "File": archivo
            }
            # Agregar solo elementos simples (números o booleanos)
            for key, value in feature_dict.items():
                if isinstance(value, (int, float, bool)):
                    row[key] = value
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df