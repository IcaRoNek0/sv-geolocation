"""Convert visual vehicle/year probabilities into a bounded county prior."""
import numpy as np


def coverage_prior(vehicle_probs, year_probs, coverage, adcodes):
    allowed = set(adcodes)
    prior = {a: 0.0 for a in adcodes}
    supported_mass = 0.0
    for vehicle, probability in vehicle_probs.items():
        years = coverage.get(vehicle.upper())
        if not years or probability <= 0:
            continue
        # Missing years use the vehicle's known union; never eliminate counties
        # solely because a trajectory export is incomplete.
        union = set().union(*(set(codes) for codes in years.values())) & allowed
        if not union:
            continue
        distribution = year_probs or {"unknown": 1.0}
        total_year = sum(distribution.values())
        for year, py in distribution.items():
            codes = set(years.get(year, union)) & allowed
            codes = codes or union
            mass = probability * py / max(total_year, 1e-12)
            for a in codes:
                prior[a] += mass / len(codes)
            supported_mass += mass
    if supported_mass:
        prior = {a: p / supported_mass for a, p in prior.items()}
    return prior, float(supported_mass)


def blend_vehicle_prior(env, prior, confidence, supported_mass,
                        strength=0.25, min_confidence=0.6):
    """At most 35% posterior adjustment; low-confidence/no-coverage is identity.

    Thresholds are experimental and require validation on visible/absent/unknown
    cars. A confident classifier output does not establish that a car is visible.
    """
    if not 0 <= strength <= 0.35 or not 0 <= min_confidence <= 1:
        raise ValueError("Invalid vehicle fusion strength/confidence")
    values = np.array(list(env.values()), dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("Invalid environmental probabilities")
    values /= values.sum()
    base = dict(zip(env, values.tolist()))
    if strength == 0 or confidence < min_confidence or supported_mass < 0.5:
        return base, False
    q = np.array([prior.get(a, 0.0) for a in env], dtype=float)
    if not np.isfinite(q).all() or (q < 0).any() or q.sum() <= 0:
        return base, False
    q /= q.sum()
    q = 0.9 * q + 0.1 / len(q)
    adjusted = values * q
    adjusted /= adjusted.sum()
    result = (1 - strength) * values + strength * adjusted
    return dict(zip(env, result.tolist())), True
