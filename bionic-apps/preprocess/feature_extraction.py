from enum import Enum, auto

import numpy as np
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# from statsmodels.tsa.ar_model import AutoReg

TIME_AXIS = -1


class FeatureType(Enum):
    HUGINES = auto()
    RAW = auto()


def MAV(series):
    return np.sum(np.abs(series), axis=TIME_AXIS) / series.shape[TIME_AXIS]


# Hudgins set: RMS, WL, ZC, SSC
def wave_len(series):
    return np.sum(np.diff(series, axis=TIME_AXIS), axis=TIME_AXIS)


def zero_crossings(series):
    return np.sum(np.abs(np.diff(np.sign(series), axis=TIME_AXIS)), axis=TIME_AXIS)


def slope_sign_change(series):
    return np.sum(np.abs(np.diff(np.sign(np.diff(series, axis=TIME_AXIS)), axis=TIME_AXIS)), axis=TIME_AXIS)


def RMS(series):
    return np.sqrt(np.mean(np.power(series, 2), axis=TIME_AXIS))


# def AR6(series):
#     coeffs = []
#     for chn in range(len(series)):
#         channel = series[chn, :]
#         model = AutoReg(channel, lags=6)
#         model_fit = model.fit()
#         coeffs.append(model_fit.params)
#     return coeffs
#
#
# def Katz_fractal(series):
#     n = series.shape[TIME_AXIS] - 1
#     logn = np.log(n)
#     sqdiffs = np.power(np.diff(series), 2)
#     L = np.sum(np.sqrt(4E-6 + sqdiffs), axis=TIME_AXIS)  # (1/500)^2 + sqdiff
#     d = np.max(
#         np.power((np.arange(1, series.shape[TIME_AXIS]) * 0.002), 2) + np.power(series[..., 0] - series[..., 1:], 2))
#     dims = logn / (logn + np.log(d / L))
#     return dims


def generate_features(x, f_type=FeatureType.HUGINES, norm=False):
    if f_type is FeatureType.RAW:
        return x

    if f_type is FeatureType.HUGINES:
        feature_union_list = [wave_len, zero_crossings, slope_sign_change, RMS]

    else:
        raise NotImplementedError
    feature_gen = FeatureUnion([(fun.__name__, FunctionTransformer(fun)) for fun in feature_union_list])
    if norm:
        feature_gen = make_pipeline(
            feature_gen,
            StandardScaler()
        )
    x = feature_gen.fit_transform(x)
    return x
