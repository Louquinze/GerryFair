import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
"""


def keep_efficient(pts):
    'returns Pareto efficient row subset of pts'
    # sort points by decreasing sum of coordinates
    pts = pts[pts.sum(1).argsort()]
    # initialize a boolean mask for undominated points
    # to avoid creating copies each iteration
    undominated = np.ones(pts.shape[0], dtype=bool)
    for i in range(pts.shape[0]):
        # process each point in turn
        n = pts.shape[0]
        if i >= n:
            break
        # find all points not dominated by i
        # since points are sorted by coordinate sum
        # i cannot dominate any points in 1,...,i-1
        undominated[i+1:n] = (pts[i+1:] <= pts[i]).any(1)
        # keep points undominated so far
        pts = pts[undominated[:n]]
    return pts


def creat_pareto(tag="default"):
    df = pd.read_csv(f"res/{tag}.csv")
    mrx = df[["error", "fairness_violation"]].to_numpy()

    pareto = keep_efficient(mrx)

    h = plt.plot(mrx[:, 0], mrx[:, 1], '.b', markersize=6, label='Non Pareto-optimal')
    h = plt.plot(pareto[:, 0], pareto[:, 1], '.r', markersize=12, label='Non Pareto-optimal')
    _ = plt.title(f'experiment: {tag}', fontsize=14)
    plt.xlabel('error', fontsize=12)
    plt.ylabel('fairness_violation', fontsize=12)
    plt.show()


def creat_tracery(tag="default"):
    df = pd.read_csv(f"res/{tag}.csv")
    mrx = df[["error", "fairness_violation"]].to_numpy()

    h = plt.plot(mrx[:, 0], mrx[:, 1], label='Tracery')
    plt.hlines(y=[0.01], xmin=[0], xmax=[max(mrx[:, 0])], colors='purple', linestyles='--', lw=2,
               label='Multiple Lines')
    _ = plt.title(f'experiment: {tag}', fontsize=14)
    plt.xlabel('error', fontsize=12)
    plt.ylabel('fairness_violation', fontsize=12)
    plt.show()
