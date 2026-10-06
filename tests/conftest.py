from pathlib import Path

import pytest

from substrate import Pipeline, Quantity, Scale, ScientificSystem, default_registry

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
PROTON, DEUTERON = 1.007276, 2.013553


def double_well(barrier=0.5, half_sep=0.4, d_e=0.05, temperature=300.0, mass=PROTON, sigma=None, **extra):
    params = {
        "mass": Quantity(mass, "amu"),
        "barrier_height": Quantity(barrier, "eV", sigma),
        "half_separation": Quantity(half_sep, "angstrom"),
        "reaction_energy": Quantity(d_e, "eV"),
        "temperature": Quantity(temperature, "K"),
    }
    params.update({k: Quantity(v, "1") for k, v in extra.items()})
    return ScientificSystem("proton", Scale.QUANTUM, "quantum.double_well_1d", parameters=params)


def evb(R=2.5, depth=4.6, alpha=2.2, r_eq=0.96, coupling=0.6, offset=0.05, mass=PROTON, temperature=300.0,
        sigmas=None, **extra):
    """Two-state valence-bond system at the electronic-structure scale."""
    values = {
        "donor_acceptor_distance": (R, "angstrom"), "morse_depth": (depth, "eV"), "morse_alpha": (alpha, "1/angstrom"),
        "morse_r_eq": (r_eq, "angstrom"), "coupling": (coupling, "eV"), "diabatic_offset": (offset, "eV"),
        "particle_mass": (mass, "amu"), "temperature": (temperature, "K"),
    }
    values.update({k: (v, "angstrom" if k == "scan_min_bond_length" else "1") for k, v in extra.items()})
    sigmas = sigmas or {}
    return ScientificSystem(
        "evb", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state",
        parameters={k: Quantity(v, u, sigmas.get(k)) for k, (v, u) in values.items()},
    )


@pytest.fixture
def pipeline():
    return Pipeline(default_registry())


def evb2d(mass=PROTON, temperature=300.0, heavy=15.9949, sigmas=None, **overrides):
    """Flexible-O...O valence-bond system. `overrides` replace a value and keep its unit (or pass (value, unit))."""
    values = {
        "morse_depth": (4.6, "eV"), "morse_alpha": (2.2, "1/angstrom"), "morse_r_eq": (0.96, "angstrom"),
        "coupling": (0.6, "eV"), "coupling_decay": (3.0, "1/angstrom"), "reference_distance": (2.5, "angstrom"),
        "oo_depth": (0.4, "eV"), "oo_alpha": (2.5, "1/angstrom"), "oo_equilibrium": (2.7, "angstrom"),
        "diabatic_offset": (0.05, "eV"), "particle_mass": (mass, "amu"), "heavy_atom_mass": (heavy, "amu"),
        "temperature": (temperature, "K"),
    }
    for name, value in overrides.items():
        values[name] = value if isinstance(value, tuple) else (value, values.get(name, (None, "1"))[1])
    sigmas = sigmas or {}
    return ScientificSystem(
        "OHO", Scale.ELECTRONIC_STRUCTURE, "electronic.evb_two_state_2d",
        parameters={k: Quantity(v, u, sigmas.get(k)) for k, (v, u) in values.items()},
    )


_CONTEXT_UNITS = {"k_on": "1/(M s)", "k_off": "1/s", "k_release": "1/s", "substrate": "M", "product": "M",
                  "k_on_product": "1/(M s)",
                  # biological scale
                  "enzyme_total": "M", "external_substrate": "M", "transport_rate": "1/s", "drain_vmax": "M/s",
                  "drain_km": "M"}
DEFAULT_CONTEXT = dict(k_on=1e7, k_off=1e3, k_release=1e4, substrate=1e-4)


def with_context(system, sigmas=None, **context):
    """Add experiment-level biophysical `context.*` parameters to a system (in place) and return it."""
    sigmas = sigmas or {}
    for name, value in (context or DEFAULT_CONTEXT).items():
        system.parameters[f"context.{name}"] = Quantity(value, _CONTEXT_UNITS[name], sigmas.get(name), "input")
    return system


def enzyme(k_f=1e5, k_r=1e4, k_on=1e7, k_off=1e3, k_release=1e4, substrate=1e-4, product=0.0,
           k_on_product=None, temperature=300.0):
    """A biophysical enzyme-cycle system built directly (no lower scales)."""
    from substrate.engines.biophysical import enzyme_cycle_structure
    system = ScientificSystem(
        "enzyme", Scale.BIOPHYSICAL, "biophysical.enzyme_cycle",
        parameters={"k_f": Quantity(k_f, "1/s"), "k_r": Quantity(k_r, "1/s"), "temperature": Quantity(temperature, "K")},
        structure=enzyme_cycle_structure(rebinding=k_on_product is not None),
    )
    context = dict(k_on=k_on, k_off=k_off, k_release=k_release, substrate=substrate, product=product)
    if k_on_product is not None:
        context["k_on_product"] = k_on_product
    return with_context(system, **context)


BIOPHYSICAL_CONTEXT = dict(k_on=1e7, k_off=1e3, k_release=1e4, substrate=1e-4, k_on_product=1e6)
BIOLOGICAL_CONTEXT = dict(enzyme_total=1e-6, external_substrate=1e-3, transport_rate=5.0, drain_vmax=2e-3, drain_km=1e-4)


def with_biology(system, sigmas=None, **overrides):
    """Add the biophysical and biological `context.*` parameters needed to reach the biological scale."""
    return with_context(system, sigmas=sigmas, **{**BIOPHYSICAL_CONTEXT, **BIOLOGICAL_CONTEXT, **overrides})


def zundel_system(program="pyscf", theory="hf", basis="6-31g*", scan_distance=2.8, n_x=21, n_r=11, mass=PROTON,
                  mirror=True, **extra):
    """The Zundel cation as a quantum-chemistry electronic-structure system (see substrate.molecules)."""
    from substrate.molecules import zundel_cation
    values = {
        "particle_mass": (mass, "amu"), "heavy_atom_mass": (15.9949, "amu"), "temperature": (300.0, "K"),
        "scan_distance": (scan_distance, "angstrom"), "n_x": (n_x, "1"), "n_r": (n_r, "1"),
        "x_extent": (0.7, "angstrom"), "distance_min": (2.3, "angstrom"), "distance_max": (3.0, "angstrom"),
    }
    values.update({k: (v, "angstrom") if k.startswith(("x_ext", "distance")) else (v, "1") for k, v in extra.items()})
    return ScientificSystem(
        "Zundel cation", Scale.ELECTRONIC_STRUCTURE, "electronic.qc_scan_2d",
        parameters={k: Quantity(v, u) for k, (v, u) in values.items()},
        structure={"molecule": zundel_cation(), "method": {"theory": theory, "basis": basis, "program": program},
                   "scan": {"mirror_symmetric": mirror}},
    )
