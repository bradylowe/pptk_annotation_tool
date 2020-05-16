import numpy as np

class Mask:
    # This seems like a cleaner way to implement masks to allow for union and intersection but some of the oddities of creation
    def __init__(self, size, val=False):
        if not val:
            self.bools = np.zeros(size, dtype=bool)
        else:
            self.bools = np.ones(size, dtype=bool)

    def __len__(self):
        return len(self.bools)

    def __getitem__(self, item):
        return self.bools[item]

    def __setitem__(self, key, value):
        self.bools[key] = value

    def intersection(self, other_mask):
        self.bools[other_mask == False] = False

    def union(self, other_mask):
        self.bools[other_mask] = True

    def set(self, mask):
        self.bools = mask

    def setr(self, indices):
        if len(indices):
            self.false()
            self.bools[indices] = True

    def setr_subset(self, indices, subset_mask):
        if len(indices) and np.sum(subset_mask):
            absolute = np.arange(len(subset_mask))[subset_mask]
            indices = absolute[indices]
            self.setr(indices)

    def compliment(self):
        self.bools = self.bools == False

    def false(self):
        self.bools[:] = False

    def true(self):
        self.bools[:] = True

    def count(self):
        return np.sum(self.bools)

    def resolve(self):
        return np.arange(len(self.bools))[self.bools]


