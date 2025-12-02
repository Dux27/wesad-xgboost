import warnings
import data_split
import numpy as np
import neurokit2 as nk

def preprocessChestSignals(chest_data: dict) -> dict:
    """Preprocess chest sensor signals by applying sensor-specific transformations."""
    for sensor in chest_data.keys():
        for label in data_split.VALID_LABELS.values():
            data_array = np.array(chest_data[sensor][label], dtype=float)
            
            if sensor == "EDA":
                data_array = ((data_array / 4096) * 3.0) / 0.12
            elif sensor == "ECG":
                pass  
            elif sensor == "EMG":
                pass  
            elif sensor == "Resp":
                pass  
            elif sensor == "Temp":
                pass  
            
            chest_data[sensor][label] = data_array

    return chest_data
    

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


def edaScrFeatures(eda: np.ndarray, prefix: str) -> dict:
    '''Extract EDA-specific features related to Skin Conductance Responses (SCR)'''
    eda = np.asarray(eda).astype(float).squeeze()
    feats = {}

    freq = data_split.getSensorFrequency("wrist", "EDA")
    with warnings.catch_warnings():     # Suppress warning about low frequency
        warnings.simplefilter("ignore")
        signals, info = nk.eda_process(eda, sampling_rate=freq)

    scr_amplitudes = signals["SCR_Amplitude"].values  # pandas to numpy

    # odfiltrowanie NaN / zer
    scr_amplitudes = scr_amplitudes[~np.isnan(scr_amplitudes)]  # Operator '~' negates the boolean array
    scr_amplitudes = scr_amplitudes[scr_amplitudes > 0]

    feats[f"{prefix}_scr_count"] = int(scr_amplitudes.size) if scr_amplitudes.size > 0 else 0
    feats[f"{prefix}_scr_mean_amp"] = float(np.mean(scr_amplitudes)) if scr_amplitudes.size > 0 else 0.0
    feats[f"{prefix}_scr_max_amp"] = float(np.max(scr_amplitudes)) if scr_amplitudes.size > 0 else 0.0
    feats[f"{prefix}_scr_sum_amp"] = float(np.sum(scr_amplitudes)) if scr_amplitudes.size > 0 else 0.0
    return feats


def extractFeatures(label: str, wrist_data: dict, chest_data: dict, index: int) -> dict:
    features: dict = {}

    for sensor in wrist_data.keys():
        if sensor == "ACC":
            pass
        elif sensor == "BVP":
            pass
        elif sensor == "EDA":
            eda_signal = wrist_data[sensor][label][index]
            features.update(basicStats(eda_signal, "wrist_EDA"))
            features.update(slopeFeature(eda_signal, "wrist_EDA", "wrist", "EDA"))
            features.update(edaScrFeatures(eda_signal, "wrist_EDA"))

        elif sensor == "TEMP":
            pass

    for sensor in chest_data.keys():
        if sensor == "ECG":
            pass
        elif sensor == "EMG":
            pass    
        elif sensor == "EDA":
            eda_signal = chest_data[sensor][label][index]
            features.update(basicStats(eda_signal, "chest_EDA"))
            features.update(slopeFeature(eda_signal, "chest_EDA", "chest", "EDA"))
            features.update(edaScrFeatures(eda_signal, "chest_EDA"))
            
        elif sensor == "Resp":
            pass
        elif sensor == "Temp":
            pass
    
    features["label"] = label
    
    return features


def main():
    dataset: list[dict] = []
    wrist_data, chest_data = data_split.main()

    for label_category in data_split.VALID_LABELS.values():
        if len(wrist_data["EDA"][label_category]) != len(chest_data["EDA"][label_category]):
            raise ValueError(f"Mismatch in number of windows for label '{label_category}' between wrist and chest EDA sensors.")
        
        for i in range(len(wrist_data['EDA'][label_category])):
            extracted_features = extractFeatures(label_category, wrist_data, chest_data, i)
            dataset.append(extracted_features)

    print(f"Dataset size: {len(dataset)} samples")
    print(dataset[0]) # Print first sample as example
    return dataset


if __name__ == "__main__":
    main()
