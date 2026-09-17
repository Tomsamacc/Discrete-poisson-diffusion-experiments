import os

import numpy as np
import torch
from scipy import stats


DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def generate_n_points(n, dist):
    """Draw n samples. Pass a dist object, not a name string.

    torch:  generate_n_points(n, torch.distributions.Poisson(20.))
    scipy:  generate_n_points(n, scipy.stats.poisson(20))
    own:    generate_n_points(n, lambda n: mix_of_two_poissons(n))
    """
    if isinstance(dist, torch.distributions.Distribution):
        return dist.sample((n,)).detach().cpu().numpy()
    if hasattr(dist, "rvs"):
        return np.asarray(dist.rvs(size=n))
    if callable(dist):
        return np.asarray(dist(n))
    raise TypeError(f"unsupported dist: {type(dist)}")


def save_dataset(x, path):
    x = np.asarray(x).reshape(-1)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, x)
    elif ext in (".txt", ".csv"):
        np.savetxt(path, x, fmt="%.10g")
    else:
        raise ValueError(f"use .npy, .txt, or .csv, got {path}")
    return path


def generate_dataset(n, dist, path):
    x = generate_n_points(n, dist)
    save_dataset(x, path)
    return x


def poisson_mix(n, w=0.5, lam0=5.0, lam1=50.0, seed=None):
    rng = np.random.default_rng(seed)
    comp = rng.random(n) < w
    x = np.empty(n, dtype=np.int64)
    x[comp] = rng.poisson(lam0, size=int(comp.sum()))
    x[~comp] = rng.poisson(lam1, size=int((~comp).sum()))
    return x


def write_split(name, dist_train, dist_val, n_train=100000, n_val=50000):
    train_path = os.path.join(DATA, name, "train.npy")
    val_path = os.path.join(DATA, name, "val.npy")
    xtr = generate_dataset(n_train, dist_train, train_path)
    xva = generate_dataset(n_val, dist_val, val_path)
    print(f"{name} train {xtr.shape} mean={xtr.mean():.4g} -> {train_path}")
    print(f"{name} val   {xva.shape} mean={xva.mean():.4g} -> {val_path}")


if __name__ == "__main__":
    write_split(
        "gamma_ltj",
        lambda n: stats.gamma.rvs(a=1, scale=10, size=n, random_state=0),
        lambda n: stats.gamma.rvs(a=1, scale=10, size=n, random_state=1),
    )
    write_split(
        "nb",
        lambda n: stats.nbinom.rvs(5, 0.2, size=n, random_state=0),
        lambda n: stats.nbinom.rvs(5, 0.2, size=n, random_state=1),
    )
    write_split(
        "poismix_mod",
        lambda n: poisson_mix(n, w=0.5, lam0=5.0, lam1=50.0, seed=0),
        lambda n: poisson_mix(n, w=0.5, lam0=5.0, lam1=50.0, seed=1),
    )
