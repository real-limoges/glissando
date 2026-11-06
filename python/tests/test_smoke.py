"""Sanity checks: module imports, distributions construct."""

import glissando


def test_module_exposes_expected_classes():
    for name in (
        "GamlssModel",
        "Gaussian",
        "Poisson",
        "Binomial",
        "Gamma",
        "NegativeBinomial",
        "Beta",
        "StudentT",
        "BCCG",
        "BCT",
        "BCPE",
        "Weibull",
        "Ocat",
        "Censored",
        "Truncated",
        "Hurdle",
    ):
        assert hasattr(glissando, name), f"glissando is missing class {name!r}"


def test_stateless_distributions_construct():
    glissando.Gaussian()
    glissando.Poisson()
    glissando.Gamma()
    glissando.NegativeBinomial()
    glissando.Beta()
    glissando.StudentT()
    glissando.BCCG()
    glissando.BCT()
    glissando.BCPE()
    glissando.Weibull()


def test_binomial_takes_n_trials():
    bin_ = glissando.Binomial([10.0, 10.0, 10.0])
    assert bin_ is not None
