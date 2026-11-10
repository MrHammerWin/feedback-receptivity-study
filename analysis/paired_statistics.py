"""Exact paired binary summaries without treating repeated policies as observations."""
import math


def binomial_cdf(k, n, p):
    return sum(math.comb(n, j) * p**j * (1-p)**(n-j) for j in range(k+1))


def clopper_pearson(k, n, alpha=.05):
    if n == 0:
        return [None, None]
    def bisect(target, cdf_k):
        lo, hi = 0., 1.
        for _ in range(100):
            mid = (lo+hi)/2
            if binomial_cdf(cdf_k, n, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    lower = 0. if k == 0 else bisect(1-alpha/2, k-1)
    upper = 1. if k == n else bisect(alpha/2, k)
    return [lower, upper]


def paired_binary(a, b):
    if len(a) != len(b):
        raise ValueError('Pairs must share the same observational units')
    n = len(a)
    win = sum(bool(x) and not bool(y) for x, y in zip(a, b, strict=True))
    loss = sum(not bool(x) and bool(y) for x, y in zip(a, b, strict=True))
    discordant = win+loss
    p = min(1., 2*binomial_cdf(min(win, loss), discordant, .5)) if discordant else 1.
    # Each marginal cell has coverage97.5%; Bonferroni gives >=95% joint
    # coverage. Unlike an empirical bootstrap, zero discordance is not CI[0,0].
    wl, wu = clopper_pearson(win, n, .025)
    ll, lu = clopper_pearson(loss, n, .025)
    return {'n': n, 'a_only': win, 'b_only': loss,
            'difference': (win-loss)/n if n else None,
            'interval95': [wl-lu, wu-ll] if n else [None,None],
            'interval_method': 'conservative exact discordant-cell Bonferroni',
            'mcnemar_exact_two_sided_p': p}


def observed_success(episode, cutoff):
    steps = episode['trajectory']
    if any(step.get('success') and step['end_logical_calls'] <= cutoff for step in steps):
        return True
    if episode['usage']['calls'] >= cutoff:
        return False
    if episode['status'] in {'success','terminated','action_horizon','model_error'}:
        return bool(episode['success'])
    # Compute cap smaller than requested cutoff and campaign interruption
    # provide no observation of the larger-cutoff continuation.
    return None


if __name__ == '__main__':
    assert clopper_pearson(0,10)[0] == 0
    assert abs(clopper_pearson(0,10)[1] - (1-.025**.1)) < 1e-10
    assert paired_binary([True]*20,[True]*20)['interval95'][1] > 0
    print('exact binary checks passed')
