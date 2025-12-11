import warnings
import data_split
import numpy as np
import neurokit2 as nk

def basicStats(x: np.ndarray, prefix: str) -> dict:
    '''Calculate basic statistical features of the signal x'''
    x = np.asarray(x).squeeze()   # Ensure x is a numpy array and flatten to 1D
    feats = {}

    feats[f"{prefix}_mean"] = float(np.mean(x))
    feats[f"{prefix}_median"] = float(np.median(x))
    feats[f"{prefix}_std"] = float(np.std(x))
    feats[f"{prefix}_min"] = float(np.min(x))
    feats[f"{prefix}_max"] = float(np.max(x))
    feats[f"{prefix}_range"] = float(np.max(x) - np.min(x))
    feats[f"{prefix}_delta"] = float(x[-1] - x[0])
    return feats


def slopeFeature(x: np.ndarray, prefix: str, location: str, sensor: str) -> dict:
   '''Calculate the slope (a - rate of change) of the signal x over time based on sensor frequency'''
   x = np.asarray(x).squeeze()   
   feats = {}

   freq = data_split.getSensorFrequency(location, sensor)
   time = np.arange(len(x)) / freq

   slope, _ = np.polyfit(time, x, 1)    # Linear regression. Slope = a
   feats[f"{prefix}_slope"] = float(slope)
   return feats


def edaScrFeatures(eda: np.ndarray, prefix: str, location: str) -> dict:
    """Extract SCR features for wrist (E4) or chest (RespiBAN) EDA."""
    eda = np.asarray(eda, dtype=float).squeeze()
    feats = {}

    freq = data_split.getSensorFrequency(location, "EDA")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        signals, info = nk.eda_process(eda, sampling_rate=freq)

    scr_amplitudes = np.asarray(signals["SCR_Amplitude"].values, dtype=float)
    scr_amplitudes = scr_amplitudes[~np.isnan(scr_amplitudes)]
    scr_amplitudes = scr_amplitudes[scr_amplitudes > 0]

    if scr_amplitudes.size == 0:
        feats[f"{prefix}_scr_count"] = 0
        feats[f"{prefix}_scr_mean_amp"] = 0.0
        feats[f"{prefix}_scr_max_amp"] = 0.0
        feats[f"{prefix}_scr_sum_amp"] = 0.0
        return feats

    feats[f"{prefix}_scr_count"] = int(scr_amplitudes.size)
    feats[f"{prefix}_scr_mean_amp"] = float(np.mean(scr_amplitudes))
    feats[f"{prefix}_scr_max_amp"] = float(np.max(scr_amplitudes))
    feats[f"{prefix}_scr_sum_amp"] = float(np.sum(scr_amplitudes))

    return feats


def accFeatures(acc: np.ndarray, prefix: str, location: str) -> dict:
    """Extract the most informative features from 3-axis accelerometer data (ACC)"""
    acc = np.asarray(acc, dtype=float)
    feats = {}

    ZERO_CROSSING_THRESHOLD = 0.005  # g

    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("ACC must be a array of [ax, ay, az].")

    mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2 + acc[:,2]**2)  # Magnitude

    if location == "wrist":
        mag = mag / 64.0  

    feats[f"{prefix}_mag_mean"] = float(np.mean(mag))
    feats[f"{prefix}_mag_std"] = float(np.std(mag))
    feats[f"{prefix}_mag_energy"] = float(np.sum(mag**2))
    feats[f"{prefix}_mag_mad"] = float(np.mean(np.abs(mag - np.mean(mag))))  # Mean Absolute Deviation
    feats[f"{prefix}_mag_range"] = float(np.max(mag) - np.min(mag))
    
    diff = np.diff(mag)
    diff[np.abs(diff) < ZERO_CROSSING_THRESHOLD] = 0
    zero_crossings = np.sum(diff[:-1] * diff[1:] < 0)
    feats[f"{prefix}_mag_zero_crossings"] = int(zero_crossings)

    return feats


def ecgFeatures(ecg: np.ndarray, location: str) -> dict:
    '''Extract ECG-based features for stress detection'''
    ecg = np.asarray(ecg, dtype=float).squeeze()
    feats = {}

    freq = data_split.getSensorFrequency(location, "ECG")
    signals, info = nk.ecg_process(ecg, sampling_rate=freq)

    heart_rate = np.asarray(signals["ECG_Rate"].values, dtype=float)
    heart_rate = heart_rate[~np.isnan(heart_rate)]
    heart_rate = heart_rate[heart_rate > 0]

    if heart_rate.size == 0:
        feats["chest_ECG_hr_mean"] = 0.0
        feats["chest_ECG_hr_std"] = 0.0
    else:
        feats["chest_ECG_hr_mean"] = float(np.mean(heart_rate))
        feats["chest_ECG_hr_std"] = float(np.std(heart_rate))
    r_peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)  
    if r_peaks.size >= 2:
        rr = np.diff(r_peaks) / float(freq) * 1000.0  # ms
        feats["chest_ECG_rr_mean"] = float(np.mean(rr))
        feats["chest_ECG_rr_std"] = float(np.std(rr))
    else:
        feats["chest_ECG_rr_mean"] = 0.0
        feats["chest_ECG_rr_std"] = 0.0

    if r_peaks.size >= 3:
        try:
            peaks_dict = {"ECG_R_Peaks": r_peaks}
            hrv_df = nk.hrv_time(peaks_dict, sampling_rate=freq, show=False)    
            hrv_row = hrv_df.iloc[0]

            feats["chest_ECG_hrv_rmssd"] = float(hrv_row.get("HRV_RMSSD", 0.0))   # Root Mean Square of Successive Differences
            feats["chest_ECG_hrv_sdnn"] = float(hrv_row.get("HRV_SDNN", 0.0))     # Standard Deviation of NN intervals (RR is NN, after filtering)
            feats["chest_ECG_hrv_pnn50"] = float(hrv_row.get("HRV_pNN50", 0.0))   # Proportion of NN50 divided by total number of NN intervals
        except Exception:
            feats["chest_ECG_hrv_rmssd"] = 0.0
            feats["chest_ECG_hrv_sdnn"] = 0.0
            feats["chest_ECG_hrv_pnn50"] = 0.0
    else:
        feats["chest_ECG_hrv_rmssd"] = 0.0
        feats["chest_ECG_hrv_sdnn"] = 0.0
        feats["chest_ECG_hrv_pnn50"] = 0.0

    return feats


def bvpFeatures(bvp: np.ndarray, location: str) -> dict:
    '''Extract BVP/PPG-based features for stress detection'''
    bvp = np.asarray(bvp, dtype=float).squeeze()
    feats = {}

    freq = data_split.getSensorFrequency(location, "BVP")
    signals, info = nk.ppg_process(bvp, sampling_rate=freq)

    heart_rate = np.asarray(signals["PPG_Rate"].values, dtype=float)
    heart_rate = heart_rate[~np.isnan(heart_rate)]
    heart_rate = heart_rate[heart_rate > 0]

    if heart_rate.size == 0:
        feats["wrist_BVP_hr_mean"] = 0.0
        feats["wrist_BVP_hr_std"] = 0.0
    else:
        feats["wrist_BVP_hr_mean"] = float(np.mean(heart_rate))
        feats["wrist_BVP_hr_std"] = float(np.std(heart_rate))
    ppg_clean = np.asarray(signals["PPG_Clean"].values, dtype=float)
    r_peaks = np.asarray(info["PPG_Peaks"], dtype=int) 

    if r_peaks.size > 0:
        peak_values = ppg_clean[r_peaks]
        feats["wrist_BVP_ppg_amp_mean"] = float(np.mean(peak_values))
        feats["wrist_BVP_ppg_amp_std"] = float(np.std(peak_values))
    else:
        feats["wrist_BVP_ppg_amp_mean"] = 0.0
        feats["wrist_BVP_ppg_amp_std"] = 0.0

    return feats


def pttFeatures(ecg, ppg) -> dict:
    ecg = np.asarray(ecg, dtype=float).squeeze()
    ppg = np.asarray(ppg, dtype=float).squeeze()
    feats = {}

    fs_ecg = data_split.getSensorFrequency("chest", "ECG")
    fs_ppg = data_split.getSensorFrequency("wrist", "BVP")

    # R peaks (ECG)
    ecg_clean = nk.ecg_clean(ecg, sampling_rate=fs_ecg)
    r_peaks = nk.ecg_peaks(ecg_clean, sampling_rate=fs_ecg)[1]["ECG_R_Peaks"]

    # PPG peaks 
    signals_ppg, info_ppg = nk.ppg_process(ppg, sampling_rate=fs_ppg)
    ppg_peaks = info_ppg["PPG_Peaks"]

    if len(r_peaks) < 2 or len(ppg_peaks) < 2:
        feats[f"PTT_mean"] = 0
        feats[f"PTT_std"] = 0
        feats[f"PTT_median"] = 0
        return feats

    # Convert to time (seconds)
    t_r = r_peaks / fs_ecg
    t_p = ppg_peaks / fs_ppg

    # Match R with next PPG peak
    ptt = []
    for r in t_r:
        nxt = t_p[t_p > r]
        if len(nxt):
            ptt.append(nxt[0] - r)

    ptt = np.array(ptt)

    # Clean invalid values
    ptt = ptt[(ptt > 0.05) & (ptt < 0.5)]  # 50–500 ms

    if len(ptt) == 0:
        feats[f"PTT_mean"] = 0
        feats[f"PTT_std"] = 0
        feats[f"PTT_median"] = 0
        return feats

    feats[f"PTT_mean"] = float(np.mean(ptt))
    feats[f"PTT_std"] = float(np.std(ptt))
    feats[f"PTT_median"] = float(np.median(ptt))

    return feats


def emgFeatures(emg: np.ndarray) -> dict:
    '''Extract EMG features related to muscle activation / tension'''
    emg = np.asarray(emg, dtype=float).squeeze()
    feats = {}

    if emg.ndim != 1:
        raise ValueError("EMG must be a 1D array.")
    
    fs = data_split.getSensorFrequency("chest", "EMG")
    emg_clean = nk.signal_filter(emg, 
                           sampling_rate=fs,
                           lowcut=20, highcut=250,
                           method="butterworth", order=4)
    
    rms = np.sqrt(np.mean(emg_clean ** 2))
    feats["chest_EMG_rms"] = float(rms)

    mean_abs = np.mean(np.abs(emg_clean))
    feats["chest_EMG_mean_abs"] = float(mean_abs)

    energy = np.sum(emg_clean ** 2)
    feats["chest_EMG_energy"] = float(energy)

    return feats


def extractFeatures(label: str, wrist_data: dict, chest_data: dict, index: int) -> dict:
    '''Extract features from wrist and chest data for a given label and window index.'''
    features: dict = {}

    # Store signals for PTT calculation
    ecg_signal = None
    bvp_signal = None

    for sensor in wrist_data.keys():
        window = wrist_data[sensor][label][index]

        if sensor == "ACC":
            features.update(accFeatures(window, "wrist_ACC", "wrist"))
        elif sensor == "BVP":
            bvp_signal = window 
            features.update(bvpFeatures(window, location="wrist"))
        elif sensor == "EDA":
            features.update(basicStats(window, "wrist_EDA"))
            features.update(slopeFeature(window, "wrist_EDA", "wrist", "EDA"))
            features.update(edaScrFeatures(window, "wrist_EDA", "wrist"))
        elif sensor == "TEMP":
            features.update(basicStats(window, "wrist_TEMP"))
            features.update(slopeFeature(window, "wrist_TEMP", "wrist", "TEMP"))

    for sensor in chest_data.keys():
        window = chest_data[sensor][label][index]

        if sensor == "ECG":
            ecg_signal = window  
            features.update(ecgFeatures(window, location="chest"))
        elif sensor == "EMG":
            features.update(emgFeatures(window))
        elif sensor == "EDA":
            features.update(basicStats(window, "chest_EDA"))
            features.update(slopeFeature(window, "chest_EDA", "chest", "EDA"))
            features.update(edaScrFeatures(window, "chest_EDA", "chest"))
        elif sensor == "Temp":
            features.update(basicStats(window, "chest_Temp"))
            features.update(slopeFeature(window, "chest_Temp", "chest", "Temp"))
        elif sensor == "Resp":
            pass
    
    if ecg_signal is not None and bvp_signal is not None:
        features.update(pttFeatures(ecg_signal, bvp_signal))
    
    features["label"] = label
    
    return features


def main():
    dataset: list[dict] = []
    wrist_data, chest_data = data_split.main()

    # Calculate total number of samples
    total_samples = sum(
        len(wrist_data["EDA"][label]) 
        for label in data_split.VALID_LABELS.values()
    )
    
    current_sample = 0
    print(f"\nExtracting features from {total_samples} samples...")

    for label_category in data_split.VALID_LABELS.values():
        if len(wrist_data["EDA"][label_category]) != len(chest_data["EDA"][label_category]):
            raise ValueError(f"Mismatch in number of windows for label '{label_category}' between wrist and chest EDA sensors.")
        
        for i in range(len(wrist_data['EDA'][label_category])):
            current_sample += 1
            progress = (current_sample / total_samples) * 100
            print(f"\rFeature extraction: {progress:.1f}% ({current_sample}/{total_samples}) - {label_category}", end="", flush=True)
            
            extracted_features = extractFeatures(label_category, wrist_data, chest_data, i)
            dataset.append(extracted_features)

    print()  
    print(f"\nDataset size: {len(dataset)} samples")
    print("First sample:", dataset[0])
    print("Feature vector length:", len(dataset[0]) - 1)
    return dataset


if __name__ == "__main__":
    main()
