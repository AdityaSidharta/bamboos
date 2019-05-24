"""Wrapper Functions for pandas, numpy, and scikit learn"""

__version__ = "0.1.3"
__author__ = "adityasidharta"

from bamboos.date import date_single, date_double
from bamboos.encode import (
    fit_binary,
    fit_categorical,
    fit_label,
    fit_onehot,
    transform_binary,
    transform_categorical,
    transform_label,
    transform_onehot,
)
from bamboos.color import color