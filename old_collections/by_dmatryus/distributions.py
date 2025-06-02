from abc import ABCMeta

import pandas as pd
import scipy.stats as ss


class Distribution(metaclass=ABCMeta):
    def __init__(self, name, dist_func: callable, dist_params: dict = None):
        self.name = name
        self.dist_func = dist_func
        self.dist_params = dist_params

    def sample(self, size: int):
        return self.dist_func(size=size, **self.dist_params)


class OnDescribeDistribution(Distribution):
    def read_params(
        self,
        count: int = None,
        max_: float = None,
        mean_: float = None,
        min_: float = None,
        std: float = None,
        q25: float = None,
        q50: float = None,
        q75: float = None,
        describe_series: pd.Series = None,
    ):
        self.dist_params = {
            "count": count
            or (describe_series.get("count") if describe_series is not None else None),
            "max_": max_
            or (describe_series.get("max_") if describe_series is not None else None),
            "mean_": mean_
            or (describe_series.get("mean_") if describe_series is not None else None),
            "min_": min_
            or (describe_series.get("min_") if describe_series is not None else None),
            "std": std
            or (describe_series.get("std") if describe_series is not None else None),
            "q25": q25
            or (describe_series.get("q25") if describe_series is not None else None),
            "q50": q50
            or (describe_series.get("q50") if describe_series is not None else None),
            "q75": q75
            or (describe_series.get("q75") if describe_series is not None else None),
        }

    def __init__(
        self,
        name: str,
        dist_func: callable,
        count: int = None,
        max_: float = None,
        mean_: float = None,
        min_: float = None,
        std: float = None,
        q25: float = None,
        q50: float = None,
        q75: float = None,
        describe_series: pd.Series = None,
    ):
        super().__init__(name, dist_func)
        self.read_params(
            count=count,
            max_=max_,
            mean_=mean_,
            min_=min_,
            std=std,
            q25=q25,
            q50=q50,
            q75=q75,
            describe_series=describe_series,
        )

    def param_mutation(self):
        return

    def sample(self, size: int = None):
        self.param_mutation()
        if size:
            self.dist_params["size"] = size
        return self.dist_func(**self.dist_params)


# ------- Discrete distributions -------
class BernoulliDist(OnDescribeDistribution):
    def __init__(
        self, count: int, mean_: float = None, describe_series: pd.Series = None
    ):
        super().__init__(
            "bernoulli",
            ss.bernoulli.rvs,
            count=count,
            mean_=mean_,
            describe_series=describe_series,
        )

    def param_mutation(self):
        self.dist_params = {
            "p": self.dist_params["mean_"],
            "size": self.dist_params["count"],
        }


class GeomDist(OnDescribeDistribution):
    def __init__(
        self, count: int, mean_: float = None, describe_series: pd.Series = None
    ):
        super().__init__(
            "geom",
            ss.geom.rvs,
            count=count,
            mean_=mean_,
            describe_series=describe_series,
        )

    def param_mutation(self, **params) -> dict:
        return {"p": self.dist_params["mean_"], "size": self.dist_params["count"]}


class LogserDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("logser", ss.logser.rvs)

    def param_mutation(self, **params) -> dict:
        return {"p": params["mean_"], "size": params["count"]}

    def sample(
        self, count: int, mean_: float = None, describe_series: pd.Series = None
    ):
        return super().sample(count=count, mean_=mean_, describe_series=describe_series)


# class


class RandintDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("randint", ss.randint.rvs)

    def param_mutation(self, **params) -> dict:
        return {"low": params["min_"], "high": params["max_"], "size": params["count"]}

    def sample(
        self,
        count: int,
        min_: float = None,
        max_: float = None,
        describe_series: pd.Series = None,
    ):
        return super().sample(
            count=count, min_=min_, max_=max_, describe_series=describe_series
        )


# ------- Continuous distributions -------


class AnglitDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("anglit", ss.anglit.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class ArcsintDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("arcsin", ss.arcsine.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class CauchyDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("cauchy", ss.cauchy.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class CosineDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("cosine", ss.cosine.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class ExponDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("expon", ss.expon.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class GibratDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("gibrat", ss.gibrat.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class GumbelRDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("gumbel_r", ss.gumbel_r.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class Gumbel_lDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("gumbel_l", ss.gumbel_l.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class HalfcauchyDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("halfcauchy", ss.halfcauchy.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class HalflogisticDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("halflogistic", ss.halflogistic.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class HalfnormDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("halfnorm", ss.halfnorm.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class HypsecantDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("hypsecant", ss.hypsecant.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class KstwobignDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("kstwobign", ss.kstwobign.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class LaplaceDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("laplace", ss.laplace.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class LevyDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("levy", ss.levy.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class Levy_lDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("levy_l", ss.levy_l.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class LogidticDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("logistic", ss.logistic.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class LognormDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("lognorm", ss.lognorm.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"], "s": params["std"]}

    def sample(self, count: int, std: float):
        return super().sample(count=count, std=std)


class MaxwellDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("maxwell", ss.maxwell.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class MoyalDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("moyal", ss.moyal.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class NormDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("norm", ss.norm.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class ReyleighDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("reyleigh", ss.rayleigh.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class SemicircularDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("semicircular", ss.semicircular.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class UniformDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("uniform", ss.uniform.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)


class WaldDist(OnDescribeDistribution):
    def __init__(self):
        super().__init__("wald", ss.wald.rvs)

    def param_mutation(self, **params) -> dict:
        return {"size": params["count"]}

    def sample(self, count: int):
        return super().sample(count=count)
