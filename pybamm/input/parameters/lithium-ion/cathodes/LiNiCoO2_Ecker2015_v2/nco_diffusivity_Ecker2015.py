from pybamm import exp, constants


def nco_diffusivity_Ecker2015(sto, T):
    """
    NCO diffusivity as a function of stochiometry [1, 2, 3].

    References
    ----------
    .. [1] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery i. determination of parameters." Journal of the
    Electrochemical Society 162.9 (2015): A1836-A1848.
    .. [2] Ecker, Madeleine, et al. "Parameterization of a physico-chemical model of
    a lithium-ion battery ii. model validation." Journal of The Electrochemical
    Society 162.9 (2015): A1849-A1857.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    y0 = -12.66
    A1 = 1.611
    sigma1 = 0.07818
    A2 = 1.614
    sigma2 = 0.09367

    log_D_ref = y0 + (
        A1 * exp(-((sto - 0.6191) / sigma1) ** 2) + 
        A2 * exp(-((sto - 1) / sigma2) ** 2)
    )
    D_ref = 10 ** log_D_ref
    E_D_s = 8.06e4
    arrhenius = exp(-E_D_s / (constants.R * T)) * exp(E_D_s / (constants.R * 296.15))

    return D_ref * arrhenius
