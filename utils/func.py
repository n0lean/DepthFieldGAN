import numpy as np


def radius_coverage(fake_points, modes, sigma, eps=0.1):
    result = np.zeros(modes.shape[0] + 1)
    for pt in fake_points:
        dist = np.linalg.norm(modes - pt, axis=1)
        mode_num = np.argmin(dist)
        if 3 * sigma > dist[mode_num]:
            result[mode_num] += 1
        else:
            result[-1] += 1
    c = 0
    # dataset_len = len(fake_points)
    dataset_len = np.sum(result[:-1])
    if dataset_len == 0:
        return 0, result[-1]/len(fake_points), result
    for key, val in enumerate(result):
        if key == len(result) - 1:
            continue
        if val / dataset_len > 1 / len(modes) * eps:
            c += 1
    return c, result[-1]/len(fake_points), result


def js_div_uniform(p, num_cat=1000):
    """ Computes the JS-divergence between p and the uniform distribution.

    """
    phat = np.bincount(p, minlength=num_cat)
    phat = (phat + 0.0) / np.sum(phat)
    pu = (phat * .0 + 1.) / num_cat
    pref = (phat + pu) / 2.
    JS = np.sum(np.log(pu / pref) * pu)
    JS += np.sum(np.log(pref / pu) * pref)
    JS = JS / 2.

    return JS