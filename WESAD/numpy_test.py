import numpy as np

tab1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])   # target length
tab2 = np.array([1, 3, 6, 7, 8, 9])               # to be stretched

target_len = len(tab1)
indices = np.linspace(0, len(tab2) - 1, target_len)
cliped_indices = np.clip(np.round(indices).astype(int), 0, len(tab2) - 1)

print("Indices:", indices)
print("Clipped indices:", cliped_indices)

aligned = tab2[cliped_indices]
print("Aligned:", aligned)
