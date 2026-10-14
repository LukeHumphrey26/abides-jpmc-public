import numpy as np


order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [2.9, 1.2],
            "frozen": False,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [100.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [200.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [300.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [400.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [500.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [600.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [700.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [800.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [900.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1000.0, 0.15],
            "frozen": True,
        },
    ],
    "weights": [
        0.2,
        0.7,
        0.06,
        0.004,
        0.0329,
        0.001,
        0.0006,
        0.0004,
        0.0005,
        0.0003,
        0.0003,
    ],
}


class OrderSizeModel:
    """Mixture-model order sizes, sampled with NumPy.

    Replaces the original pomegranate 0.14 ``GeneralMixtureModel`` (which does not
    install on Python 3.12). Same mixture as ``order_size`` above: pick a component
    by weight, then draw from it. ``LogNormalDistribution`` parameters are the
    (mu, sigma) of the underlying normal; ``NormalDistribution`` parameters are
    (mean, std). Distributionally equivalent, but NOT draw-for-draw identical to
    pomegranate for a given seed.
    """

    def __init__(self) -> None:
        dists = order_size["distributions"]
        self.names = [d["name"] for d in dists]
        self.params = [tuple(d["parameters"]) for d in dists]
        w = np.asarray(order_size["weights"], dtype=float)
        self.weights = w / w.sum()

    def sample(self, random_state: np.random.RandomState) -> float:
        k = random_state.choice(len(self.weights), p=self.weights)
        a, b = self.params[k]
        if self.names[k] == "LogNormalDistribution":
            x = random_state.lognormal(mean=a, sigma=b)
        else:
            x = random_state.normal(loc=a, scale=b)
        return round(x)
