import data_split
import numpy as np

wrist_data, chest_data = data_split.main()  

print(f"Wrist number of sensors: {len(wrist_data)}")
print(f"Chest number of sensors: {len(chest_data)}")

# for sensor in wrist_data.keys():
#     for label, windows in wrist_data[sensor].items():
#         for window in windows:
#             print(f"Wrist Sensor: {sensor}, Label: {label}, Window shape: {window.shape}")

# print(f"{len(wrist_data['EDA'])}")

# for label_category in data_split.VALID_LABELS.values():
#     if len(wrist_data["EDA"][label_category]) != len(chest_data["EDA"][label_category]):
#         raise ValueError(f"Mismatch in number of windows for label '{label_category}' between wrist and chest EDA sensors.")
    
#     for i in range(len(wrist_data['EDA'][label_category])):
#         print(f"Wrist EDA Window {i} for label '{label_category}': shape {wrist_data['EDA'][label_category][i].shape}")
#         print(f"Chest EDA Window {i} for label '{label_category}': shape {chest_data['EDA'][label_category][i].shape}")

# x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# freq = 4
# t = np.arange(len(x))
# print(t)

# print(wrist_data['EDA']['baseline'][0])
# print()
# print(chest_data['EDA']['baseline'][0])

# def preprocessChestSignals(chest_data: dict) -> dict:
#     """Preprocess chest sensor signals by applying sensor-specific transformations."""
#     for sensor in chest_data.keys():
#         for label in data_split.VALID_LABELS.values():
#             if sensor == "ECG":
#                 pass
#             elif sensor == "EMG":
#                 pass    
#             elif sensor == "EDA":
#                 eda_array = np.array(chest_data[sensor][label])
#                 preprocessed = ((eda_array / 4096) * 3.0) / 0.12
#                 chest_data[sensor][label] = preprocessed.tolist()
#             elif sensor == "Resp":
#                 pass
#             elif sensor == "Temp":
#                 pass

#     return chest_data

# chest_data = preprocessChestSignals(chest_data)
# print(chest_data['EDA']['baseline'][0])