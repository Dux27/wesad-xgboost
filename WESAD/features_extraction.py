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
   feats[f"{prefix}_slope"] = slope
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
    feats = {}

    ZERO_CROSSING_THRESHOLD = 0.005  # g

    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[1] != 3:
        raise ValueError("ACC must be a Nx3 array of [ax, ay, az].")

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


def extractFeatures(label: str, wrist_data: dict, chest_data: dict, index: int) -> dict:
    '''Extract features from wrist and chest data for a given label and window index.'''
    features: dict = {}

    for sensor in wrist_data.keys():
        window = wrist_data[sensor][label][index]

        if sensor == "ACC":
            features.update(accFeatures(window, "wrist_ACC", "wrist"))
        elif sensor == "BVP":
            pass
        elif sensor == "EDA":
            features.update(basicStats(window, "wrist_EDA"))
            features.update(slopeFeature(window, "wrist_EDA", "wrist", "EDA"))
            features.update(edaScrFeatures(window, "wrist_EDA", "wrist"))

        elif sensor == "TEMP":
            features.update(basicStats(window, "wrist_TEMP"))

    for sensor in chest_data.keys():
        window = chest_data[sensor][label][index]

        if sensor == "ACC":
            features.update(accFeatures(window, "chest_ACC", "chest"))
        elif sensor == "ECG":
            pass
        elif sensor == "EMG":
            pass    
        elif sensor == "EDA":
            features.update(basicStats(window, "chest_EDA"))
            features.update(slopeFeature(window, "chest_EDA", "chest", "EDA"))
            features.update(edaScrFeatures(window, "chest_EDA", "chest"))
            
        elif sensor == "Temp":
            features.update(basicStats(window, "chest_Temp"))
        elif sensor == "Resp":
            pass
    
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
    return dataset


if __name__ == "__main__":
    main()
