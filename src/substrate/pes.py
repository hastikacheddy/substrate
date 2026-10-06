"""Locating the reactant and product wells on a 1D potential-energy surface."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

#: a minimum only counts as a well if the surface rises at least this far (eV) on the way to any lower point;
#: this keeps numerical or fitting noise from being mistaken for chemistry
MIN_PROMINENCE_EV = 5e-3


def prominent_minima(v: np.ndarray, min_prominence: float = MIN_PROMINENCE_EV) -> np.ndarray:
    return find_peaks(-np.asarray(v), prominence=min_prominence)[0]


def locate_wells(v: np.ndarray, min_prominence: float = MIN_PROMINENCE_EV) -> tuple[int, int, int] | None:
    """Indices (left well, barrier top, right well): the two lowest prominent minima and the highest
    point between them. None when the surface has fewer than two wells."""
    v = np.asarray(v)
    minima = prominent_minima(v, min_prominence)
    if len(minima) < 2:
        return None
    i_left, i_right = sorted(int(i) for i in minima[np.argsort(v[minima])[:2]])
    i_top = i_left + int(np.argmax(v[i_left:i_right + 1]))
    return i_left, i_top, i_right
