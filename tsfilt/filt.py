import abc
import numpy as np
from scipy.stats import norm
from .util import window
from .padding import get_padder


class BaseSpatialFilter():
    """Base class for time series filtering.
    """
    def __init__(self, win_size=3, padding="same", n_iter=1):
        assert win_size % 2 == 1, "window size must be odd value."
        assert padding in ("zero", "same", "identical"),\
            "padding method has to be `zero`, `same` or `identical`."

        self.win_size = win_size
        self.padder = get_padder(padding, {"padding_size": win_size // 2})
        self.med_idx = win_size // 2
        self.n_iter = n_iter

    def fit(self, seq):
        self.seq_ = seq  # keep original signal internal
        # padding
        self.seq_padded_ = self.padder.transform(seq)

        return self
    
    def transform(self, seq):
        for i in range(self.n_iter):
            x = self.seq_padded_
            # do filtering for each sub-sequence
            filt = []
            for xs in window(x, n=self.win_size):
                filt.append(self._filt(xs))
            x = np.hstack(filt)
            self.seq_padded_ = self.padder.transform(x)

        return x

    def fit_transform(self, seq):
        self.fit(seq)
        return self.transform(seq)

    @abc.abstractmethod
    def _filt(self, sub_seq):
        """Execute filtering for sub sequence.
        """
        pass

class IdenticalFilter():
    """
    """
    def fit(self, seq):
        return self
    
    def transform(self, seq):
        return seq

    def fit_transform(self, seq):
        return seq

class BoxFilter(BaseSpatialFilter):
    """Box Filtering Class.
    """
    def __init__(self, win_size, padding="same", n_iter=1):
        super(BoxFilter, self).__init__(win_size, padding, n_iter)
        self.weight = 1.0 / win_size * np.ones(win_size)

    def _filt(self, sub_seq):
        prod = self.weight.reshape(1, -1) @ sub_seq.reshape(-1, 1)
        return prod[0, 0]

class GaussianFilter(BaseSpatialFilter):
    """Gaussian Filtering Class.
    """
    def __init__(self, win_size, padding="same", n_iter=1, sigma_d=None):
        super(GaussianFilter, self).__init__(win_size, padding, n_iter)
        if sigma_d is None:
            sigma_d = self._suggest_sigma_d()
        self.sigma_d = sigma_d
        self.weight = norm.pdf(np.arange(win_size), loc=self.med_idx, scale=self.sigma_d)
        self.weight /= self.weight.sum()

    def _filt(self, sub_seq):
        prod = self.weight.reshape(1, -1) @ sub_seq.reshape(-1, 1)
        return prod[0, 0]

    def _suggest_sigma_d(self):
        RATIO = 4
        return self.win_size / (RATIO * 2)

class BilateralFilter(GaussianFilter):
    """Bilateral Filtering Class.
    """
    def __init__(self, win_size, padding="same", n_iter=1,\
            sigma_d=None, sigma_i=None):
        super(BilateralFilter, self).__init__(win_size, padding, n_iter, sigma_d)
        self.sigma_i = sigma_i

    def _filt(self, sub_seq):
        if self.sigma_i is None:
            self.sigma_i = self._suggest_sigma_i()

        w = norm.pdf(sub_seq, loc=sub_seq[self.med_idx], scale=self.sigma_i)
        weight = self.weight * w
        weight /= weight.sum()

        prod = weight.reshape(1, -1) @ sub_seq.reshape(-1, 1)
        return prod[0, 0]

    def _suggest_sigma_i(self):
        """Suggest sigma param.
        Estimate noise standard deviation.
        """
        x = self.seq_
        # 1% of total range
        return (x.max() - x.min()) / 100.0

class NonLocalMeanFilter(BaseSpatialFilter):
    """NonLocalMeanFilter Class.
    """
    def __init__(self, win_size, padding="same", n_iter=1,\
        beta=1.0, th=None, delta=0.5, sigma=None):
        super(NonLocalMeanFilter, self).__init__(win_size, padding, n_iter)
        self.beta = beta
        self.th = th
        self.sigma = sigma
        self.delta = delta

    def _filt(self, sub_seq):
        if self.sigma is None:
            self.sigma = self._suggest_sigma()
        
        self.th = self._suggest_th(sub_seq)

        x_all = self.seq_padded_
        weight = np.empty_like(self.seq_)

        for i, xs in enumerate(window(x_all, n=self.win_size)):
            d = np.linalg.norm((xs - sub_seq), ord=2)
            if d < self.th:
                weight[i] = np.exp(-d / (2 * self.beta * self.sigma * self.win_size))
            else:
                weight[i] = 0
        
        if weight.sum() == 0:
            return sub_seq[self.med_idx]

        weight /= weight.sum()
        prod = weight.reshape(1, -1) @ self.seq_.reshape(-1, 1)

        self.w_ = weight

        return prod[0, 0]

    def _suggest_sigma(self):
        """Suggest sigma param.
        Estimate noise standard deviation.
        """
        x = self.seq_
        # 1% of total range
        return (x.max() - x.min()) / 100.0

    def _suggest_th(self, sub_seq):
        max_val = sub_seq.max()
        min_val = sub_seq.min()
        return self.delta * (max_val - min_val) * self.win_size

class ChainFilter():
    def __init__(self, filters, keep_intermediate=False):
        self.filters = filters
        self.keep_intermediate = keep_intermediate
        if keep_intermediate:
            self.filt_res_ = list()

    def fit_transform(self, seq):
        for filt in self.filters:
            seq = filt.fit_transform(seq)
            if self.keep_intermediate:
                self.filt_res_.append(seq)

        return seq