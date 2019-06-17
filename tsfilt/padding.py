import abc
import numpy as np

def get_padder(method, params=dict()):
    padders = {
        "zero": ZeroPadder,
        "same": SamePadder,
        "identical": IdenticalPadder,
    }
    return padders[method](**params)

class BasePadder():
    def __init__(self, padding_size):
        self.padding_size = padding_size

    @abc.abstractmethod
    def transform(self, seq):
        pass
    
    @abc.abstractmethod
    def inv_transform(self, seq):
        pass

class ZeroPadder(BasePadder):
    PAD_VALUE = 0.0
    def __init__(self, padding_size):
        super(ZeroPadder, self).__init__(padding_size)

    def transform(self, seq):
        return np.hstack([
            self.PAD_VALUE * np.ones(self.padding_size),
            seq,
            self.PAD_VALUE * np.ones(self.padding_size),
        ])

    def inv_transform(self, seq):
        ps = self.padding_size
        return  seq[ps:-ps]

class SamePadder(BasePadder):
    def __init__(self, padding_size):
        super(SamePadder, self).__init__(padding_size)

    def transform(self, seq):
        start_val = seq[0]
        end_val = seq[-1]

        return np.hstack([
            start_val * np.ones(self.padding_size),
            seq,
            end_val * np.ones(self.padding_size),
        ])

    def inv_transform(self, seq):
        ps = self.padding_size
        return  seq[ps:-ps]

class IdenticalPadder(BasePadder):
    def transform(self, seq):
        return seq

    def inv_transform(self, seq):
        return seq

