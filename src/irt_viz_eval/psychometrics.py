"""Psychometric curve functions for benchmark stimulus generation."""

from __future__ import annotations

import numpy as np


D_SCALE = 1.7


def theta_grid(theta_min: float = -4.0, theta_max: float = 4.0, step: float = 0.05) -> np.ndarray:
    count = int(round((theta_max - theta_min) / step)) + 1
    return np.linspace(theta_min, theta_max, count)


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dichotomous_probability(
    theta: np.ndarray,
    a: float = 1.0,
    b: float = 0.0,
    c: float = 0.0,
    d: float = 1.0,
) -> np.ndarray:
    """1PL/2PL/3PL/4PL probability curve."""
    core = logistic(D_SCALE * a * (theta - b))
    return c + (d - c) * core


def dichotomous_information(
    theta: np.ndarray,
    a: float = 1.0,
    b: float = 0.0,
    c: float = 0.0,
    d: float = 1.0,
) -> np.ndarray:
    """Binary item information via the derivative of the response function."""
    core = logistic(D_SCALE * a * (theta - b))
    probability = dichotomous_probability(theta, a=a, b=b, c=c, d=d)
    derivative = (d - c) * D_SCALE * a * core * (1.0 - core)
    denom = np.clip(probability * (1.0 - probability), 1e-9, None)
    return np.square(derivative) / denom


def pcm_category_probabilities(theta: np.ndarray, thresholds: list[float], a: float = 1.0) -> np.ndarray:
    """Partial-credit style category probabilities.

    Returns an array shaped (len(theta), number_of_categories). Thresholds define
    the adjacent category transition locations.
    """
    m = len(thresholds)
    logits = [np.zeros_like(theta)]
    running = np.zeros_like(theta)
    for threshold in thresholds:
        running = running + D_SCALE * a * (theta - threshold)
        logits.append(running.copy())
    stacked = np.vstack(logits).T
    stacked = stacked - stacked.max(axis=1, keepdims=True)
    numerator = np.exp(stacked)
    return numerator / numerator.sum(axis=1, keepdims=True)


def grm_category_probabilities(theta: np.ndarray, thresholds: list[float], a: float = 1.0) -> np.ndarray:
    """Graded response model category probabilities."""
    cumulative = [logistic(D_SCALE * a * (theta - threshold)) for threshold in thresholds]
    probs = []
    probs.append(1.0 - cumulative[0])
    for idx in range(1, len(thresholds)):
        probs.append(cumulative[idx - 1] - cumulative[idx])
    probs.append(cumulative[-1])
    return np.vstack(probs).T


def expected_score(category_probs: np.ndarray) -> np.ndarray:
    scores = np.arange(category_probs.shape[1], dtype=float)
    return category_probs @ scores


def ideal_point_probability(
    theta: np.ndarray,
    location: float = 0.0,
    width: float = 1.0,
    lower: float = 0.02,
    upper: float = 0.95,
) -> np.ndarray:
    """Simple non-monotonic ideal-point endorsement curve.

    This is intentionally labeled as an ideal-point approximation rather than a
    full GGUM implementation. It gives the benchmark a transparent, reproducible
    non-monotonic stress test.
    """
    peak = np.exp(-0.5 * np.square((theta - location) / width))
    return lower + (upper - lower) * peak


def ideal_point_information(
    theta: np.ndarray,
    location: float = 0.0,
    width: float = 1.0,
    lower: float = 0.02,
    upper: float = 0.95,
) -> np.ndarray:
    probability = ideal_point_probability(theta, location=location, width=width, lower=lower, upper=upper)
    derivative = probability * (-(theta - location) / np.square(width))
    denom = np.clip(probability * (1.0 - probability), 1e-9, None)
    return np.square(derivative) / denom


def nearest_value(theta: np.ndarray, y: np.ndarray, target_theta: float) -> float:
    idx = int(np.argmin(np.abs(theta - target_theta)))
    return float(y[idx])


def curve_peak(theta: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    idx = int(np.argmax(y))
    return float(theta[idx]), float(y[idx])
