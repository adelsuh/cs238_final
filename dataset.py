from torch.utils.data import Dataset
import numpy as np
from dateutil import rrule
from datetime import datetime
from bisect import bisect_left

class MahjongDataset(Dataset):
    def __init__(self, lengths_file, features_dir, action_type, action_idx):
        self.features_dir = features_dir
        self.lengths = np.cumsum(np.load(lengths_file)[action_idx], dtype=np.int16)
        self.days = [dt.strftime("%Y%m%d") for dt in rrule.rrule(rrule.DAILY, dtstart=datetime.strptime("20230101", '%Y%m%d'),
                      until=datetime.strptime("20230531", '%Y%m%d'))]
        self.action_type = action_type

    def __len__(self):
        return self.lengths[-1]

    def __getitem__(self, idx):
        i = bisect_left(self.lengths, idx)
        prev = 0 if i == 0 else self.lengths[i-1]
        data = np.load(self.features_dir+self.action_type+self.days[i]+".npz", dtype=np.int16)
        return (data["sparse"][idx-prev-1], data["dense"][idx-prev-1]), data["labels"][idx-prev-1]