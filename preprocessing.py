import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

filePath = sys.argv[1]
labels = np.load(filePath+"/labels.npy")
train_size = .7
labels_per_class = [5, 25, 50]
n_repeats = 5

indices = np.arange(len(labels))

train_idx, _ = train_test_split(
    indices,
    train_size=train_size,
    stratify=labels,
    random_state=42,  # for reproducibility
    shuffle=True,
)


sublabels = labels[train_idx]
cl_val, counts = np.unique(sublabels, return_counts=True)
print(counts)
max_k = max(labels_per_class)
short_classes = cl_val[counts < max_k]
if len(short_classes) > 0:
    print(f"Warning: classes {short_classes} have fewer than {max_k} "
          f"training samples; labelled sets for those classes will be smaller than requested.")

suffix = os.path.basename(filePath.rstrip("/"))
os.makedirs(suffix, exist_ok=True)
np.save("%s/train_idx.npy"%suffix, train_idx)

for repet in range(n_repeats):
    labelled_by_k = {k: [] for k in labels_per_class}
    
    for el in cl_val:
        idx = np.where(sublabels == el)[0]
        idx = shuffle(idx, random_state=repet)
        for k in labelled_by_k.keys():
            labelled_by_k[k].extend(idx[:k])

    for k in labelled_by_k:
        np.save("%s/labelled_samples_%d_%d.npy"%(suffix, k, repet), np.array(labelled_by_k[k]) )
