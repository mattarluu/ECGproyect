import wfdb
import numpy as np
import pandas as pd
import os
import neurokit2 as nk
from scipy.signal import find_peaks

#function to obtain the heart rate of a signal
def get_heart_rate(record):
    
    signal = record.p_signal[:, 0]
    fs = record.fs
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    hr = nk.ecg_rate(rpeaks, sampling_rate=fs)
    return np.mean(hr)

#function to obtain the standard deviation of the mean time between beats of a signal, and calculate if it is irregular
def get_rr_stdM(record, threshold_factor=0.2):
    signal = record.p_signal[:, 0]  
    fs = record.fs
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)

    rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / fs
    
    rr_std = np.std(rr_intervals)
    
    rr_mean = np.mean(rr_intervals)
    
    is_irregular = rr_std > threshold_factor * rr_mean  
    
    return rr_std, is_irregular

#function to obtain the times between beats of a signal
def get_rr_std(record):
    signal = record.p_signal[:, 0]
    fs = record.fs
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    rr_intervals = np.diff(rpeaks["ECG_R_Peaks"]) / fs
    return rr_intervals

#function to obtain the mean of qrs durations of a asignal
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

#function to obtain the features of the qrs durations of a asignal
def get_qrs_features(record, qrs_duration_threshold=120, qrs_amplitude_variation_threshold=0.2):

    signal = record.p_signal[:, 0]
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []
    
    qrs_amplitude_previous = None  
    alternancia_electrica_detectada = False 

    for i, r_idx in enumerate(rpeaks["ECG_R_Peaks"]):
        try:
            q_idx = delineate[1]["ECG_Q_Peaks"][i]
            s_idx = delineate[1]["ECG_S_Peaks"][i]
        
            qrs_duration = (s_idx - q_idx) * 1000 / fs 
            
            is_qrs_wide = qrs_duration > qrs_duration_threshold

            q_value = abs(cleaned[q_idx])
            r_value = abs(cleaned[r_idx])
            s_value = abs(cleaned[s_idx])
            qrs_amplitude = q_value + r_value + s_value  

            if qrs_amplitude_previous is not None:
                amplitude_variation = abs(qrs_amplitude - qrs_amplitude_previous)
                if amplitude_variation > qrs_amplitude_variation_threshold:
                    alternancia_electrica_detectada = True

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

#function to obtain the summary of qrs features of a asignal
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

#function to obtain the qt interval durations of a asignal    
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

#function to obtain the mean of qt interval durations of a asignal
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

#Function to obtain the qtc duration values of a signal, and if they are prolonged
def get_qtc_bazett(record, threshold_ms=500):
    signal = record.p_signal[:, 0]
    fs = record.fs

    
    qt_intervals = get_qt_interval(record)  
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    
    r_locs = rpeaks["ECG_R_Peaks"]
    rr_intervals = np.diff(r_locs) / fs 
    
    min_len = min(len(qt_intervals), len(rr_intervals))
    qt_intervals = qt_intervals[:min_len]
    rr_intervals = rr_intervals[:min_len]
    
    qtc_values = [(qt / np.sqrt(rr)) * 1000 for qt, rr in zip(qt_intervals, rr_intervals)]  # en ms
    count_prolonged = sum(1 for qtc in qtc_values if qtc > threshold_ms)
    
    return {
        "QTc_values_ms": qtc_values,
        "QTc_prolonged_count": count_prolonged
    }

#Function to obtain the mean of the durations of the p wave of a signal
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

#Function to obtain the durations of the p wave of a signal
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

#Function to obtain the mean of the durations of the pr intervals of a signal
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

#Function to obtain the durations of the pr intervals of a signal, and if they have depression
def get_pr_interval(record):
    signal = record.p_signal[:, 0]  
    fs = record.fs
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    pr_intervals = []
    pr_depression_inferior = False

    p_onsets = delineate[1]["ECG_P_Onsets"]
    q_peaks = delineate[1]["ECG_Q_Peaks"]

    lead_names = record.sig_name
    inferior_leads = ["II", "III", "aVF"]
    inferior_indices = [i for i, name in enumerate(lead_names) if name in inferior_leads]

    for i in range(min(len(p_onsets), len(q_peaks))):
        try:
            onset = p_onsets[i]
            q_peak = q_peaks[i]

            if onset is None or q_peak is None or q_peak <= onset:
                continue

            pr = (q_peak - onset) / fs
            pr_intervals.append(pr)

            for idx in inferior_indices:
                pr_segment = cleaned[onset:q_peak, idx]
                baseline = cleaned[onset - 40:onset, idx] if onset >= 40 else cleaned[:onset, idx]
                baseline_mean = np.mean(baseline)
                pr_mean = np.mean(pr_segment)

                if (baseline_mean - pr_mean) > 0.1:  
                    pr_depression_inferior = True
                    break 

        except (IndexError, TypeError):
            continue

    return {
        "pr_intervals_s": pr_intervals if len(pr_intervals) > 0 else np.nan,
        "pr_depression_in_inferior_leads": pr_depression_inferior
    }
    
#Function to obtain the features of the q wave of a signal
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

            duration = (r_idx - q_idx) * 1000 / fs  
            relative_amplitude = q_value / r_value if r_value != 0 else 0

            q_pathological = False
        
            if q_value > 0.3:
                q_pathological = True
            elif duration > 40:
                q_pathological = True

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

#function to obtain the summary of  the features of the q waves of a asignal
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

#Function to obtain the features of the p wave of a signal
def get_p_wave_features(record, p_amplitude_threshold=0.1, p_duration_threshold=150):
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

            duration_p = (offset_p - onset_p) * 1000 / fs

            r_idx = rpeaks["ECG_R_Peaks"][i]
            r_value = abs(cleaned[r_idx]) if cleaned[r_idx] != 0 else 1e-6
            relative_amplitude_p = p_value / r_value

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

    if p_missing:
        results.append({"p_missing": True})

    return results

#function to obtain the summary of  the features of the p waves of a asignal
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

#Function to obtain the features of the r wave of a signal
def get_r_wave_features(record, threshold_r_amplitude=0.5, r_v1_threshold=7):
 
    signal = record.p_signal  
    fs = record.fs  

    cleaned = nk.ecg_clean(signal[:, 0], sampling_rate=fs)

    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)

    v1_signal = signal[:, 0]  # Derivation V1
    v2_signal = signal[:, 1]  # Derivation V2
    v3_signal = signal[:, 2]  # Derivation V3
    v4_signal = signal[:, 3]  # Derivation V4
    v5_signal = signal[:, 4]  # Derivation V5
    v6_signal = signal[:, 5]  # Derivation V6
    
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)
    
    results = []
    for i in range(len(delineate[1]["ECG_R_Onsets"])):
        try:
            r_idx = rpeaks["ECG_R_Peaks"][i]
            onset_r = delineate[1]["ECG_R_Onsets"][i]
            offset_r = delineate[1]["ECG_R_Offsets"][i]
            r_value = abs(cleaned[r_idx])

            duration_r = (offset_r - onset_r) * 1000 / fs

            r_amplitude_v1 = abs(v1_signal[r_idx])
            r_amplitude_v2 = abs(v2_signal[r_idx])
            r_amplitude_v3 = abs(v3_signal[r_idx])
            r_amplitude_v4 = abs(v4_signal[r_idx])
            r_amplitude_v5 = abs(v5_signal[r_idx])
            r_amplitude_v6 = abs(v6_signal[r_idx])
            
            r_high = any(amplitude > threshold_r_amplitude for amplitude in [r_amplitude_v1, r_amplitude_v2, r_amplitude_v3, r_amplitude_v4, r_amplitude_v5, r_amplitude_v6])
            
            r_v1_high = r_amplitude_v1 > r_v1_threshold  

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

#function to obtain the summary of  the features of the r waves of a asignal
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

#Function to obtain the features of the s wave of a signal
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

            duration_s = (s_idx - r_idx) * 1000 / fs
         
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

#function to obtain the summary of  the features of the s waves of a asignal
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

#Function to obtain the features of the t wave of a signal
def get_t_wave_features(record):
    signal = record.p_signal[:, 0] 
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

            t_negative_right_precordial = False
            if record.sig_name[0] in ["V1", "V2"] and cleaned[t_idx] < 0:
                t_negative_right_precordial = True

            t_segment = cleaned[onset_t:offset_t]
            peaks, _ = find_peaks(t_segment, distance=int(0.05 * fs)) 
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

#function to obtain the summary of  the features of the t waves of a asignal
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

#Function to obtain the features of the st segments of a signal
def get_st_segment_features(record):
    
    signals = record.p_signal
    fs = record.fs
    leads = record.sig_name

    signal = signals[:, 0]
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    delineate = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt", show=False)

    results = []

    for i, r_idx in enumerate(rpeaks["ECG_R_Peaks"]):
        try:
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

#Function to obtain if a signal has branch lock patterns 
def get_brd_bri_patterns(record, threshold_s_depth=0.2, threshold_r_prime_amplitude=0.1):
    signal = record.p_signal
    fs = record.fs
    cleaned = nk.ecg_clean(signal[:, 0], sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
  
    v1_signal = signal[:, record.sig_name.index('v1')]  
    v5_signal = signal[:, record.sig_name.index('v5')] 
    v6_signal = signal[:, record.sig_name.index('v6')]  
    
    brd_pattern = detect_rsr_pattern(v1_signal, fs, threshold_r_prime_amplitude)
    
    bri_pattern = detect_s_wave_depth(v5_signal, v6_signal, fs, threshold_s_depth)
    
    results = {
        "brd_pattern": brd_pattern,
        "bri_pattern": bri_pattern
    }
    
    return results

#Function to obtain if a signal has rsr pattern
def detect_rsr_pattern(signal, fs, threshold_r_prime_amplitude):

    if signal.size == 0:
        raise ValueError("La señal proporcionada está vacía.")

    if fs <= 0:
        raise ValueError(f"La frecuencia de muestreo 'fs' no es válida: {fs}")
    
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    except Exception as e:
        raise RuntimeError(f"Error al limpiar la señal: {e}")
    
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    
    try:
        delineate = nk.ecg_delineate(signal, sampling_rate=fs, method="dwt", show=False)
    except Exception as e:
        raise RuntimeError(f"Error al delinear la señal ECG: {e}")
    
    r_peaks = rpeaks["ECG_R_Peaks"]  
    s_peaks = delineate[1]["ECG_S_Peaks"]  
    
    brd_detected = False
    min_len = min(len(r_peaks) - 1, len(s_peaks))  
    for i in range(1, min_len): 
        r_idx = r_peaks[i]
        s_idx = s_peaks[i]
  
        if r_idx > s_idx and r_idx - s_idx > 30:  
            try:
                r_prime_idx = r_peaks[i + 1] 
                if abs(signal[r_prime_idx]) > threshold_r_prime_amplitude: 
                    brd_detected = True
                    break
            except IndexError:
                continue
    
    return brd_detected

#Function to obtain the s wave depth of a signal
def detect_s_wave_depth(v5_signal, v6_signal, fs, threshold_s_depth):
  
    delineate_v5 = nk.ecg_delineate(v5_signal, sampling_rate=fs, method="dwt", show=False)
    delineate_v6 = nk.ecg_delineate(v6_signal, sampling_rate=fs, method="dwt", show=False)
    
    s_peaks_v5 = delineate_v5[1]["ECG_S_Peaks"]
    s_peaks_v6 = delineate_v6[1]["ECG_S_Peaks"]
    
    bri_detected = False
    
    min_len = min(len(s_peaks_v5), len(s_peaks_v6))
    
    for i in range(min_len):
        s_v5 = s_peaks_v5[i]
        s_v6 = s_peaks_v6[i]

        if s_v5 < len(v5_signal) and s_v6 < len(v6_signal):
            if abs(v5_signal[s_v5]) < -threshold_s_depth and abs(v6_signal[s_v6]) < -threshold_s_depth:
                bri_detected = True
                break
    
    return bri_detected

#Function to obtain the sokolow lyon index of a signal
def get_sokolow_lyon_index(record, threshold=35):

    signal = record.p_signal  
    fs = record.fs  

    try:
        idx_v1 = record.sig_name.index('v1')
        idx_v5 = record.sig_name.index('v5')
        idx_v6 = record.sig_name.index('v6')
    except ValueError as e:
        raise ValueError(f"No se encontraron derivaciones necesarias en el registro: {e}")

    v1_signal = signal[:, idx_v1]
    v5_signal = signal[:, idx_v5]
    v6_signal = signal[:, idx_v6]

    cleaned = nk.ecg_clean(v5_signal, sampling_rate=fs)
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)

    try:
        delineate_v1 = nk.ecg_delineate(v1_signal, rpeaks["ECG_R_Peaks"], sampling_rate=fs, method="dwt", show=False)
    except IndexError:
        delineate_v1 = nk.ecg_delineate(v1_signal, rpeaks["ECG_R_Peaks"], sampling_rate=fs, method="peak", show=False)

    s_peaks_v1 = delineate_v1[1].get("ECG_S_Peaks", [])
    r_peaks = rpeaks["ECG_R_Peaks"]

    s_amplitudes_v1 = [abs(v1_signal[i]) for i in s_peaks_v1 if i < len(v1_signal)]
    r_amplitudes_v5 = [abs(v5_signal[i]) for i in r_peaks if i < len(v5_signal)]
    r_amplitudes_v6 = [abs(v6_signal[i]) for i in r_peaks if i < len(v6_signal)]

    s_amplitude_v1 = max(s_amplitudes_v1) if s_amplitudes_v1 else 0
    r_amplitude_v5 = max(r_amplitudes_v5) if r_amplitudes_v5 else 0
    r_amplitude_v6 = max(r_amplitudes_v6) if r_amplitudes_v6 else 0

    r_amplitude_max = max(r_amplitude_v5, r_amplitude_v6)
    sokolow_lyon_index = s_amplitude_v1 + r_amplitude_max

    is_positive = sokolow_lyon_index > threshold

    results = {
        "S_in_V1_amplitude_mm": s_amplitude_v1,
        "R_in_V5_amplitude_mm": r_amplitude_v5,
        "R_in_V6_amplitude_mm": r_amplitude_v6,
        "Sokolow_Lyon_index_mm": sokolow_lyon_index,
        "Index_positive": is_positive
    }

    return results

#Function to obtain all features of a signal
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

#Function to detect a possible infarct in a signal
def possible_infarct(patient_id, archivo, caracteristicas):
    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

    st_results = archivo_features["st_segment"]

    st_elevated_contiguous=False
    for result in st_results:
        st_elevation = result["ST_elevation_contiguous"]
        if st_elevation==True:
            st_elevated_contiguous=True
            break

    t_wave = archivo_features["t_wave"]

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

#Function to detect a possible arrhythmia in a signal
def possible_arrhythmia(patient_id, archivo, caracteristicas):
    
    rr_irregular = caracteristicas[patient_id][archivo]["rrM"][1]
    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

    p_results = archivo_features["p_wave"]

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

#Function to detect a possible branch block in a signal
def possible_branch_block(patient_id, archivo, caracteristicas):
    
    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

    qrs_results = archivo_features["qrs"]

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

#Function to detect a possible ventricular hypertrophy in a signal
def possible_ventricular_hypertrophy(patient_id, archivo, caracteristicas):
    
    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

    r_results = archivo_features["r_wave"]

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

#Function to detect a possible long-QT syndrome in a signal
def possible_long_QT_syndrome(patient_id, archivo, caracteristicas):

    qt = caracteristicas[patient_id][archivo]["qtc"]["QTc_prolonged_count"]
    qt_prolonged=False
    if(qt>0):
        qt_prolonged=True
    
    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

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

#Function to detect a possible brugada syndrome in a signal
def possible_brugada_syndrome(patient_id, archivo, caracteristicas):
    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

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

#Function to detect a possible pericarditis in a signal
def possible_pericarditis(patient_id, archivo, caracteristicas):

    patient_features = caracteristicas[patient_id]

    archivo_features = patient_features[archivo]

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

#Function to get a dataframe of the probablities of each disease of a signal
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

    rows = []

    for patient_id in allFeatures:
        for archivo in allFeatures[patient_id]:
            row = {
                "Patient": patient_id,
                "File": archivo
            }
      
            scores = {}
            for disease, func in diseases.items():
                try:
                    score = func(patient_id, archivo, allFeatures)
                except:
                    score = 0
                scores[disease] = score
                row[disease] = score
         
            max_disease = max(scores, key=scores.get)
            row["Max_Label"] = max_disease

            rows.append(row)

    df = pd.DataFrame(rows)
    
    return df

#Function to transofrom the dictionary of features in a dataframe
def features_dict_to_df(features_dict):
    rows = []
    
    for patient_id, archivos in features_dict.items():
        for archivo, feature_dict in archivos.items():
            row = {
                "Patient": patient_id,
                "File": archivo
            }
            for key, value in feature_dict.items():
                if isinstance(value, (int, float, bool)):
                    row[key] = value
            rows.append(row)
    
    df = pd.DataFrame(rows)
    return df