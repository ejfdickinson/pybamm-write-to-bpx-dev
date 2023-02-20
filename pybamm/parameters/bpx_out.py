# Utility functions for converting pybamm.ParameterValues to BPX JSON format
# Original author: Edmund Dickinson (ED), About:Energy Ltd, February 2023
# Uploaded to PyBaMM development, Feb 2023

import pybamm
import json

from math import log10, floor
from scipy import optimize

ELECTRODES = [
    "Negative",
    "Positive"
]

def round_float(x, sf):
    '''
    Round x to sf significant figures.
    '''
    # Note(ED): rounding direction is unpredictable at decimal midpoints. Not considered a big issue for this use-case.

    if sf < 1:
        raise ValueError("expected significant figures >= 1 in round_float().")
    if x == 0:
        return x

    dp = sf-1-floor(log10(abs(x)))
    return round(x,dp)

def tidy_bpx(val):
    '''
    Handle different types in dict -> JSON conversion
    '''
    # TODO: handle pybamm.Interpolant cases
    if callable(val):
        # TODO: handle function cases properly!

        # Note(ED):
        # Basically the issue is that a PyBaMM function can be any valid Python code (dynamically typed)
        # but BPX has many restrictions on allowed functions.
        # BPX functions are parsed to Python using a Function() class in create_from_BPX().
        # Basic idea could be to enforce some restrictions on callable PyBaMM functions that the code can parse.
        # For a simple function, string-substitute scalar values and temperature into the return script.
        # Consider using sequential (interpreted) substitution, swap out 'T' with nominal temperature
        # and swap out array constants with numeric values.
        # Numerical derivatives could be used to convert T-dependent functions to
        # required activation energy, entropic coefficient.

        # PLACEHOLDER to just write the function name as str while functions are not handled
        return val.__name__
    
    elif isinstance(val,float):
        # Round floats to 5 s.f. for output
        return round_float(val,5)
    else:
        return val

def calc_avol(volfrac, rp):
    '''
    Volumetric surface area from active material volume fraction and particle radius
    '''
    return (3*volfrac/rp)

def calc_b_effcy(poro, brugg):
    '''
    Transport efficiency from porosity and Bruggeman coefficient
    '''
    return (poro ** brugg)

def calc_eff_cond(cond, volfrac, brugg):
    '''
    Effective conductivity from conductivity, volume fraction and Bruggeman coefficient
    '''
    return (cond * calc_b_effcy(volfrac, brugg))

def calc_xLi_SOC0(xLi_SOC100, U_els, nLi_sat, E_eod):
    '''
    Calculates SOC = 0% lithiation extents (xLi_pos_max and xLi_neg_min) given SOC = 100% lithiation extents,
    OCP functions and necessary gravimetric information.

    Parameters
    ---
    xLi_SOC100 - tuple of xLi_pos_min, xLi_neg_max (SOC = 100% stoichiometry)
    U_els - tuple of pos/neg callables returning pybamm.Scalar
    nLi_sat - tuple of pos/neg theoretical saturated Li concentrations (mol.m-2) = csat * L * active volume fraction
    E_EOD - end-of-discharge voltage
    ---

    Return
    ---
    xLi_SOC0 - tuple of xLi_pos_max, xLi_neg_min (SOC = 0% stoichiometry)
    ---
    '''
    # Note(ED): initial guesses are hard-coded but could be improved
    # Note(ED): as fsolve presumably uses a numerical derivative, need to double-check accuracy. Worked for basic tests.
    # Note(ED): very likely to fail if Upos or Uneg becomes singular, even for sto < 0 or sto > 1. Encourage use of well-behaved functions as input.

    xLi_pos_min, xLi_neg_max = xLi_SOC100
    Upos, Uneg = U_els
    nLi_sat_pos, nLi_sat_neg = nLi_sat

    # Initial guesses - positive, negative
    x0 = [0.9, 0.1]

    def func(x1):
        xLi_pos_max, xLi_neg_min = tuple(x1)

        return [
            Upos(xLi_pos_max).value - Uneg(xLi_neg_min).value - E_eod,                              # equality of OCPs at E = E_eod
            nLi_sat_pos * (xLi_pos_max - xLi_pos_min) - nLi_sat_neg * (xLi_neg_max - xLi_neg_min)   # equality of Li contents in each electrode
        ]
    
    xLi_SOC0 = tuple(optimize.fsolve(func, x0))
    return xLi_SOC0

def calc_internal(parameter_values):
    '''
    Bespoke conversions of ParameterValues object data to a dict of BPX data values.
    '''
    # TODO: functional parameterisation of any of the looked-up ParameterValues cases
    # can fail if the input is a callable rather than a value. Need to handle for any
    # fields where this can be true.

    param_internal = {}
    
    # Electrode area
    h_el = parameter_values["Electrode height [m]"]
    w_el = parameter_values["Electrode width [m]"]
    param_internal["A_el"] = h_el * w_el

    # Separator transport efficiency
    poro_sep = parameter_values["Separator porosity"]
    brugg_sep = parameter_values["Separator Bruggeman coefficient (electrolyte)"]
    param_internal["b_effcy_sep"] = calc_b_effcy(poro_sep, brugg_sep)
    
    # Limiting stoichiometries
    c0_neg = parameter_values["Initial concentration in negative electrode [mol.m-3]"]
    c0_pos = parameter_values["Initial concentration in positive electrode [mol.m-3]"]

    csat_neg = parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
    csat_pos = parameter_values["Maximum concentration in positive electrode [mol.m-3]"]

    Lel_neg = parameter_values["Negative electrode thickness [m]"]
    Lel_pos = parameter_values["Positive electrode thickness [m]"]

    volfrac_neg = parameter_values["Negative electrode active material volume fraction"]
    volfrac_pos = parameter_values["Positive electrode active material volume fraction"]

    nLi_sat_neg = csat_neg * Lel_neg * volfrac_neg
    nLi_sat_pos = csat_pos * Lel_pos * volfrac_pos

    # Initial conditions are SOC = 100%
    # TODO: If "Initial concentration" is not 100%, this will be wrong. If this can ever be the case, fix.
    param_internal["xLi_min_pos"] = c0_pos/csat_pos
    param_internal["xLi_max_neg"] = c0_neg/csat_neg

    # TODO: Ensure that the OCP entries in parameter_values are callable functions. If not, this will fail.
    param_internal["xLi_max_pos"], param_internal["xLi_min_neg"] = calc_xLi_SOC0(
        (param_internal["xLi_min_pos"], param_internal["xLi_max_neg"]),
        (parameter_values["Positive electrode OCP [V]"], parameter_values["Negative electrode OCP [V]"]),
        (nLi_sat_pos, nLi_sat_neg),
        parameter_values["Lower voltage cut-off [V]"]
    )

    for electrode in ELECTRODES:
        # Add "_pos" or "_neg" to names of electrode-specific properties
        subscr = "_" + electrode[0:3].lower()

        # Volumetric surface area
        volfrac = parameter_values[electrode + " electrode active material volume fraction"]
        rp = parameter_values[electrode + " particle radius [m]"]
        param_internal["avol" + subscr] = calc_avol(volfrac, rp)

        # Transport efficiency
        poro = parameter_values[electrode + " electrode porosity"]    
        brugg_el = parameter_values[electrode + " electrode Bruggeman coefficient (electrolyte)"]
        param_internal["b_effcy" + subscr] = calc_b_effcy(poro, brugg_el)

        # Conductivity
        cond = parameter_values[electrode + " electrode conductivity [S.m-1]"]
        brugg_solid = parameter_values[electrode + " electrode Bruggeman coefficient (electrode)"]
        param_internal["cond_eff" + subscr] = calc_eff_cond(cond, volfrac, brugg_solid)

    return param_internal

def write_to_bpx(parameter_values, fp,
        bpx_version=0.3, title='', description='', references='',
        model='DFN', dens_cell=0, Cpsp_cell=0, kave_cell=0
    ):
    '''
    Creates a BPX file from a PyBaMM ParameterValues object.
    Header information is provided as supplementary function input.
    Inclusion of Validation data is not supported.

    Parameters
    ---
    parameter_values - a pybamm.ParameterValues object
    fp - file-like object to write
    bpx_version - BPX version to write for backwards compatibility (default: latest version)
    title - string input to "Title" field of header
    description - string input to "Description" field of header
    references - string input to "References" field of header
    model - string input "Model" field of header
    dens_cell - cell average density (kg/m3)
    Cpsp_cell - cell average specific heat (J/kg/K)
    kave_cell - cell average thermal conductivity (W/m/K)
    '''
    # IMPORTANT: function in progress. Does not currently output a valid BPX.
    # IMPORTANT: Arguably, this should be a method of the ParameterValues class.
    #            In this case, the first argument should be "self".
    
    # TODO: add support for insertion of "Validation" data

    # TODO: make rounding in output optional / user-controllable
    # Note(ED): may wish to improve consistency of scientific notation and rounding in output
    #     
    # TODO: test and as required implement functionality for lookup tables
    # TODO: handle cases when parameters are expressed as functions
    # TODO(ED): add support for outstanding list of temperature-dependent BPX parameters
    # The following require some function analysis:
        # Electrolyte
            # Diffusivity activation energy [J.mol-1]
            # Conductivity activation energy [J.mol-1]
        # Negative electrode/Positive electrode
            # Reaction rate constant [mol.m-2.s-1]
            # Reaction rate constant activation energy [J.mol-1]
            # Diffusivity activation energy [J.mol-1]     

    ## PYBAMM <=> BPX NAME MAPS
    # TODO: if desired, re-order to match BPX demo files
    PYBAMM_TO_BPX_GENERAL = {
        # PyBaMM parameter name         : (BPX class, BPX parameter name)
        "Cell cooling surface area [m2]": ("Cell", "External surface area [m2]"),
        "Cell volume [m3]"              : ("Cell", "Volume [m3]"),
        "Number of electrodes connected in parallel to make a cell": ("Cell", "Number of electrode pairs connected in parallel to make a cell"),
        "Lower voltage cut-off [V]"     : ("Cell", "Lower voltage cut-off [V]"),
        "Upper voltage cut-off [V]"     : ("Cell", "Upper voltage cut-off [V]"),
        "Nominal cell capacity [A.h]"   : ("Cell", "Nominal cell capacity [A.h]"),
        "Ambient temperature [K]"       : ("Cell", "Ambient temperature [K]"),
        "Initial temperature [K]"       : ("Cell", "Initial temperature [K]"),
        "Reference temperature [K]"     : ("Cell", "Reference temperature [K]"),
        "Initial concentration in electrolyte [mol.m-3]": ("Electrolyte", "Initial concentration [mol.m-3]"), 
        "Cation transference number"    : ("Electrolyte", "Cation transference number"),
        "Electrolyte diffusivity [m2.s-1]": ("Electrolyte", "Diffusivity [m2.s-1]"),
        "Electrolyte conductivity [S.m-1]": ("Electrolyte", "Conductivity [S.m-1]"),
        "Separator thickness [m]"       : ("Separator", "Thickness [m]"),
        "Separator porosity"            : ("Separator", "Porosity")    
    }

    PYBAMM_TO_BPX_ELECTRODES = {
        # PyBaMM parameter name         : BPX parameter name
        # PyBaMM parameter names contain substitutes as follows:
        # "ID_upper" for "Positive"/"Negative"
        # "id_lower" for "positive"/"negative
        "ID_upper electrode thickness [m]": "Thickness [m]",
        "ID_upper electrode porosity"   : "Porosity",
        "Maximum concentration in id_lower electrode [mol.m-3]": "Maximum concentration [mol.m-3]",
        "ID_upper particle radius [m]"  : "Particle radius [m]",
        "ID_upper electrode diffusivity [m2.s-1]": "Diffusivity [m2.s-1]",
        "ID_upper electrode OCP [V]"    : "OCP [V]",
        "ID_upper electrode OCP entropic change [V.K-1]": "Entropic change coefficient [V.K-1]",
    }

    INTERNAL_TO_BPX_GENERAL = {
        # Internal computed value: (BPX class, BPX parameter name)
        "A_el"          : ("Cell", "Electrode area [m2]"),
        "b_effcy_sep"   : ("Separator", "Transport efficiency")
    }

    INTERNAL_TO_BPX_ELECTRODES = {
        # Internal computed value: BPX parameter name
        # Internal computed value names contain substitutes as follows:
        # "id" for "pos"/"neg"
        "avol_id"          : "Surface area per unit volume [m-1]",
        "b_effcy_id"       : "Transport efficiency",
        "cond_eff_id"      : "Conductivity [S.m-1]",
        "xLi_min_id"       : "Minimum stoichiometry",
        "xLi_max_id"       : "Maximum stoichiometry"
    }

    CELL_THERMAL_OPTS = [
        # TODO: if desired, use kwargs instead
        # BPX Cell parameters that are set from function input
        "Density [kg.m-3]",
        "Specific heat capacity [J.K-1.kg-1]",
        "Thermal conductivity [W.m-1.K-1]"
    ]

    VALID_MODELS = [
        "DFN",
        "SPMe",
        "SPM"
    ]
    
    ## INPUT VALIDATION
    if model not in VALID_MODELS:
        # TODO: better error handling
        raise ValueError("Valid model types in write_to_bpx(): DFN, SPMe, SPM.")

    ## MAIN
    # Assemble the BPX as a dict in place, from blank dictionary
    dict_bpx = {}

    ## HEADER INFORMATION
    # Required fields
    dict_header = {
        "BPX"  : bpx_version,
        "Model": model
    }
    # Only add optional fields if present
    if title:
        dict_header["Title"] = title
    if description:
        dict_header["Description"] = description
    if references:
        dict_header["References"] = references
        
    dict_bpx["Header"] = dict_header

    ## PARAMETER INFORMATION

    # Calculate BPX variables not represented explicitly by PyBaMM    
    param_internal = calc_internal(parameter_values)
    
    # Create a local parameters dict to assemble to BPX
    dict_params = {
        "Cell":                 {},
        "Electrolyte":          {},
        "Negative electrode":   {},
        "Positive electrode":   {},
        "Separator":            {}
    }

    # Non-electrode parameters
    # Parameters which can directly map
    for str_PyBaMM,(class_bpx, str_bpx) in PYBAMM_TO_BPX_GENERAL.items():
        val = tidy_bpx(parameter_values[str_PyBaMM])
        dict_params[class_bpx][str_bpx] = val

    # Parameters computed internally
    for str_internal,(class_bpx, str_bpx) in INTERNAL_TO_BPX_GENERAL.items():
        val = tidy_bpx(param_internal[str_internal])
        dict_params[class_bpx][str_bpx] = val

    # Handle parameters specified at input
    for param,str_param in zip((dens_cell, Cpsp_cell, kave_cell),CELL_THERMAL_OPTS):
        # Only add optional fields if present
        if param:
            dict_params["Cell"][str_param] = param

    # Electrode parameters
    for electrode in ELECTRODES:
        class_bpx = electrode + " electrode"

        # Parameters which can directly map
        for str_PyBaMM,str_bpx in PYBAMM_TO_BPX_ELECTRODES.items():
            str_PyBaMM = str_PyBaMM.replace('ID_upper',electrode).replace('id_lower',electrode.lower())
            val = tidy_bpx(parameter_values[str_PyBaMM])
            dict_params[class_bpx][str_bpx] = val

        # Parameters computed internally
        for str_internal,str_bpx in INTERNAL_TO_BPX_ELECTRODES.items():
            str_internal = str_internal.replace('id',electrode[0:3].lower())
            val = tidy_bpx(param_internal[str_internal])
            dict_params[class_bpx][str_bpx] = val                             

    dict_bpx["Parameterisation"] = dict_params

    ## JSON OUTPUT
    # Write the compiled dict to file
    # TODO: check for overwrite and get user confirmation
    # TODO: check for success and report failure
    with open(fp,'w') as target:
        json.dump(dict_bpx,target,indent=6)