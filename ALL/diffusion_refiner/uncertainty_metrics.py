"""Quantitative uncertainty metrics: Brier, ECE, Correlation, AURC."""

import numpy as np
from scipy.stats import pearsonr


def brier_score(probs, targets):
    p = np.asarray(probs, dtype=np.float64).flatten()
    t = np.asarray(targets, dtype=np.float64).flatten()
    return float(np.mean((p - t) ** 2))


def expected_calibration_error(probs, targets, n_bins=10):
    p = np.asarray(probs, dtype=np.float64).flatten()
    t = np.asarray(targets, dtype=np.float64).flatten()
    bounds = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p > bounds[i]) & (p <= bounds[i + 1])
        if mask.sum() == 0: continue
        ece += (mask.sum() / len(p)) * abs(t[mask].mean() - p[mask].mean())
    return float(ece)


def error_variance_correlation(mean_pred, variance, targets):
    err = np.abs(np.asarray(mean_pred).flatten() - np.asarray(targets).flatten())
    var = np.asarray(variance).flatten()
    corr, pval = pearsonr(var, err)
    return float(corr), float(pval)


def area_under_risk_coverage(mean_pred, variance, targets, n_points=100):
    mp = np.asarray(mean_pred, dtype=np.float64).flatten()
    vr = np.asarray(variance, dtype=np.float64).flatten()
    tg = np.asarray(targets, dtype=np.float64).flatten()
    
    idx = np.argsort(-vr)
    mp, tg = mp[idx], tg[idx]
    
    covs = np.linspace(0.1, 1.0, n_points)
    risks = []
    for c in covs:
        n = max(1, int(len(mp) * c))
        pb = (mp[:n] >= 0.5).astype(np.float64)
        tb = (tg[:n] >= 0.5).astype(np.float64)
        inter = (pb * tb).sum()
        denom = pb.sum() + tb.sum()
        dice = 2 * inter / (denom + 1e-7) if denom > 0 else 1.0
        risks.append(1.0 - dice)
    return float(np.trapz(risks, covs)), covs, risks


def compute_all_uncertainty_metrics(all_probs, all_vars, all_targets):
    p = np.concatenate([np.asarray(x, dtype=np.float64).flatten() for x in all_probs])
    v = np.concatenate([np.asarray(x, dtype=np.float64).flatten() for x in all_vars])
    t = np.concatenate([np.asarray(x, dtype=np.float64).flatten() for x in all_targets])
    
    valid = ~(np.isnan(p) | np.isnan(t) | np.isinf(p) | np.isinf(t) |
              np.isnan(v) | np.isinf(v))
    p, v, t = p[valid], v[valid], t[valid]
    
    corr, pval = error_variance_correlation(p, v, t)
    aurc, covs, risks = area_under_risk_coverage(p, v, t)
    
    return {
        'brier': brier_score(p, t),
        'ece': expected_calibration_error(p, t),
        'correlation': corr, 'correlation_pvalue': pval,
        'aurc': aurc, 'coverages': covs, 'risks': risks,
    }


def print_uncertainty_report(results, method_name="Method"):
    print(f"\n{'='*55}")
    print(f"  {method_name}")
    print(f"{'='*55}")
    print(f"  Brier (↓):  {results['brier']:.4f}")
    print(f"  ECE   (↓):  {results['ece']:.4f}")
    print(f"  Corr  (↑):  {results['correlation']:.4f}  (p={results['correlation_pvalue']:.4f})")
    print(f"  AURC  (↓):  {results['aurc']:.4f}")
    print(f"{'='*55}\n")