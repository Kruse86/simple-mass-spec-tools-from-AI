#!/usr/bin/env python3
"""
CADB v6.0 — Core forward model (no GUI)

Pipeline:
    Metabolite concentrations
        -> per-residue reaction probability (competing kinetic processes)
        -> Monte Carlo over independent virtual protein copies
        -> summed mass shift PER MOLECULE
        -> instrument response (mass-dependent broadening)
        -> predicted intact-mass distribution

Key corrections vs. v5.2 (see accompanying notes):
  1. Monte Carlo now runs over independent protein copies and sums all
     modifications per copy into ONE mass before histogramming (v5.2 histogrammed
     individual per-residue events as if each were a separate molecule — wrong).
  2. reactive_fraction() uses a single Henderson-Hasselbalch expression; the
     pKa>7/<7 branch in v5.2 was a genuine formula bug, not a simplification.
  3. Arginine is no longer gated by amine-type deprotonation — dicarbonyl adduction
     on Arg proceeds via the guanidinium group in its normal protonation state.
  4. Competing metabolites for a single residue are resolved via competing-rate
     kinetics (rates summed, outcome chosen proportionally to rate), removing the
     list-order bias present in v5.2's sequential loop.
  5. Instrument broadening (dropped in the v5.x OOP rewrite) is reinstated as a
     mass-dependent Gaussian convolution.
  6. Optional per-molecule exposure-time sampling (steady-state turnover model).

All kinetic parameters (kon, Km) are ORDER-OF-MAGNITUDE PLACEHOLDERS unless the
`evidence` field says otherwise. Treat outputs as illustrative until calibrated
against literature or experimental rate data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import random
import numpy as np

# ====================== CONSTANTS ======================

AA_MASS = {
    "A": 71.03711, "R": 156.10111, "N": 114.04293, "D": 115.02694,
    "C": 103.00919, "E": 129.04259, "Q": 128.05858, "G": 57.02146,
    "H": 137.05891, "I": 113.08406, "L": 113.08406, "K": 128.09496,
    "M": 131.04049, "F": 147.06841, "P": 97.05276, "S": 87.03203,
    "T": 101.04768, "W": 186.07931, "Y": 163.06333, "V": 99.06841,
}
WATER = 18.01056

# pKa of the conjugate acid of each nucleophilic side chain (used only for
# residues whose mechanism genuinely depends on the deprotonated/neutral form).
SIDE_CHAIN_PKA = {"K": 10.5, "H": 6.0, "C": 8.3, "Y": 10.1}
# Arg is deliberately excluded — see note above.


# ====================== CHEMISTRY DATA ======================

@dataclass
class Metabolite:
    name: str
    formula: str
    exact_mass: float
    mass_shift: float          # fixed mass added by ONE adduction event (Da)
    conc_mM: float
    kon: float                 # effective 2nd-order rate constant, M^-1 s^-1 (placeholder unless noted)
    targets: List[str]         # amino acid one-letter codes this metabolite can modify
    description: str
    mechanism: str = "Schiff base / rearrangement"
    reversible: bool = False
    evidence: str = "Illustrative — not literature-calibrated"
    reference: str = ""


DEFAULT_METABOLITES: List[Metabolite] = [
    Metabolite(
        name="Methylglyoxal", formula="C3H4O2", exact_mass=72.021, mass_shift=54.0106,
        conc_mM=0.0003,  # ~300 nM free plasma MGO — order-of-magnitude, not patient-specific
        kon=0.5, targets=["K", "R"],
        description="Highly reactive alpha-oxoaldehyde; dominant irreversible AGE precursor",
        mechanism="Hydroimidazolone formation (Arg) / Nε-carboxyethyl-lysine (Lys)",
        evidence="Order-of-magnitude placeholder, calibrated only to give plausible "
                 "(not saturating) occupancy over a ~7-day exposure window; NOT fit to "
                 "measured rate constants.",
    ),
    Metabolite(
        name="Glyoxal", formula="C2H2O2", exact_mass=58.005, mass_shift=58.0055,
        conc_mM=0.0001, kon=0.3, targets=["K", "R"],
        description="Dialdehyde, lipid-peroxidation/glycoxidation product",
        mechanism="Hydroimidazolone / CML formation",
        evidence="Order-of-magnitude placeholder — see Methylglyoxal note",
    ),
    Metabolite(
        name="Glucose", formula="C6H12O6", exact_mass=180.063, mass_shift=162.0528,
        conc_mM=5.5, kon=0.0005, targets=["K"],
        description="Reducing sugar; slow but high-concentration glycation via Schiff base -> Amadori",
        mechanism="Schiff base -> Amadori rearrangement -> AGE",
        reversible=True,
        evidence="Concentration is physiological (fasting, mM); kon is an order-of-"
                 "magnitude placeholder, not a literature value.",
    ),
    Metabolite(
        name="Pyruvate", formula="C3H4O3", exact_mass=88.016, mass_shift=70.0055,
        conc_mM=0.08, kon=0.02, targets=["K"],
        description="alpha-keto acid, minor glycation contributor",
        evidence="Order-of-magnitude placeholder",
    ),
]

# Scenario multipliers act on metabolite CONCENTRATION (i.e. on plasma exposure),
# not on mass shift or on kon — see chemistry discussion.
SCENARIOS: Dict[str, Dict[str, float]] = {
    "Normal":                 {"Methylglyoxal": 1.0, "Glyoxal": 1.0, "Glucose": 1.0, "Pyruvate": 1.0},
    "Hyperglycemia":          {"Methylglyoxal": 1.5, "Glyoxal": 1.2, "Glucose": 3.0, "Pyruvate": 1.0},
    "Inflammatory Acidosis":  {"Methylglyoxal": 3.5, "Glyoxal": 2.8, "Glucose": 1.0, "Pyruvate": 1.5},
    "Combined":               {"Methylglyoxal": 2.8, "Glyoxal": 2.2, "Glucose": 2.5, "Pyruvate": 1.2},
}


# ====================== RESIDUES / PROTEIN ======================

# Default solvent-accessibility placeholders by residue type, used only until
# real structure-derived values (e.g. SASA from a PDB/AlphaFold model) are
# supplied. These are NOT measured values — they exist so that not every
# lysine/arginine in the sequence is treated as 100% and identically exposed,
# which is the single biggest lever on how fast the model saturates.
DEFAULT_ACCESSIBILITY = {"K": 0.5, "R": 0.3, "H": 0.3, "C": 0.3, "Y": 0.3}


@dataclass
class Residue:
    position: int
    aa: str
    modified_by: Optional[str] = None  # metabolite name, or None
    accessibility: float = field(default=0.0)  # 0-1, set from DEFAULT_ACCESSIBILITY below

    def __post_init__(self):
        if self.accessibility == 0.0:
            self.accessibility = DEFAULT_ACCESSIBILITY.get(self.aa, 0.0)

    def reactive_fraction(self, pH: float) -> float:
        """Fraction of this residue's side chain in its reactive (typically
        deprotonated/neutral) form, via Henderson-Hasselbalch. Single formula
        used for ALL residues that need it — no pKa-dependent branching.
        Arginine is handled separately (see note in module docstring).
        """
        if self.aa == "R":
            # Guanidinium reacts with dicarbonyls in its normally-protonated
            # state; not gated by deprotonation. Treat as fully available,
            # subject only to accessibility.
            return 1.0
        pKa = SIDE_CHAIN_PKA.get(self.aa)
        if pKa is None:
            return 0.0
        return 1.0 / (1.0 + 10 ** (pKa - pH))

    @property
    def is_modified(self) -> bool:
        return self.modified_by is not None


class Protein:
    def __init__(self, header: str, sequence: str):
        self.header = header
        self.sequence = "".join(c for c in sequence.upper() if c.isalpha())
        self.residues: List[Residue] = [
            Residue(i + 1, aa) for i, aa in enumerate(self.sequence)
        ]

    @classmethod
    def from_fasta(cls, filename: str) -> "Protein":
        with open(filename) as f:
            lines = f.readlines()
        header = lines[0].strip().lstrip(">") if lines and lines[0].startswith(">") else "protein"
        seq_lines = lines[1:] if lines and lines[0].startswith(">") else lines
        seq = "".join(line.strip() for line in seq_lines)
        return cls(header, seq)

    def native_mass(self) -> float:
        return WATER + sum(AA_MASS.get(aa, 0.0) for aa in self.sequence)

    def fresh_copy(self) -> "Protein":
        """Return a new Protein instance sharing the sequence but with
        independent (unmodified) residue state, for Monte Carlo sampling."""
        return Protein(self.header, self.sequence)


# ====================== KINETIC MODELS ======================

class KineticModel:
    """Returns a per-second (or per-dt) reaction RATE, not a probability,
    so that competing metabolites can be combined correctly."""

    def rate(self, metabolite: Metabolite, residue: Residue, pH: float) -> float:
        raise NotImplementedError


class SecondOrderModel(KineticModel):
    """Default: pseudo-first-order in metabolite (metabolite in large excess
    relative to any single residue), i.e. standard mass-action kinetics.
    This is the chemically appropriate default for spontaneous, non-enzymatic
    modification (Michaelis-Menten below is an optional empirical alternative,
    not a mechanistic claim of enzyme-like saturation)."""

    def rate(self, metabolite: Metabolite, residue: Residue, pH: float) -> float:
        frac = residue.reactive_fraction(pH)
        conc_M = metabolite.conc_mM * 1e-3
        return metabolite.kon * conc_M * frac * residue.accessibility  # s^-1


class MichaelisModel(KineticModel):
    """Empirical saturating approximation. Useful only as a phenomenological
    fit option; does not imply an enzyme-substrate mechanism."""

    def __init__(self, Km_mM: float = 1.0):
        self.Km_mM = Km_mM

    def rate(self, metabolite: Metabolite, residue: Residue, pH: float) -> float:
        frac = residue.reactive_fraction(pH)
        S = frac * residue.accessibility
        v = (metabolite.kon * S * metabolite.conc_mM) / (self.Km_mM + metabolite.conc_mM)
        return v  # s^-1 (approx.)


# ====================== REACTION ENGINE ======================

class ReactionEngine:
    """Resolves competition between multiple metabolites for each residue
    using competing-rate kinetics rather than sequential trial-and-lock."""

    def __init__(self, model: Optional[KineticModel] = None):
        self.model = model or SecondOrderModel()

    def react_one_molecule(
        self,
        protein: Protein,
        metabolites: List[Metabolite],
        pH: float,
        exposure_seconds: float,
    ) -> List[dict]:
        """Simulate ONE virtual copy of the protein for exposure_seconds.
        Returns the list of modification events that occurred on this molecule."""
        events = []
        for residue in protein.residues:
            applicable = [m for m in metabolites if residue.aa in m.targets]
            if not applicable:
                continue

            rates = np.array([self.model.rate(m, residue, pH) for m in applicable])
            total_rate = rates.sum()
            if total_rate <= 0:
                continue

            p_react_at_all = 1.0 - np.exp(-total_rate * exposure_seconds)
            if random.random() < p_react_at_all:
                # competing-risk: choose which metabolite won, proportional to rate
                weights = rates / total_rate
                chosen = np.random.choice(len(applicable), p=weights)
                metabolite = applicable[chosen]
                residue.modified_by = metabolite.name
                events.append({
                    "position": residue.position,
                    "aa": residue.aa,
                    "metabolite": metabolite.name,
                    "mass_shift": metabolite.mass_shift,
                })
        return events


# ====================== INSTRUMENT RESPONSE ======================

class InstrumentModel:
    """Mass-dependent Gaussian broadening approximating linear-mode MALDI-TOF
    resolution loss at high mass. This is a first-order approximation of
    resolving power, matrix/adduct noise and unresolved biological
    micro-heterogeneity lumped together — NOT a full isotope/charge-state
    simulation."""

    def __init__(self, resolving_power: float = 400.0, floor_sigma_da: float = 40.0):
        # resolving_power ~ M/FWHM; linear MALDI-TOF for intact proteins is
        # typically in the low hundreds, much lower than reflector mode.
        self.resolving_power = resolving_power
        self.floor_sigma_da = floor_sigma_da

    def broaden(self, masses: np.ndarray) -> np.ndarray:
        fwhm = masses / self.resolving_power
        sigma = np.maximum(fwhm / 2.355, self.floor_sigma_da)
        return masses + np.random.normal(0.0, sigma)


# ====================== POPULATION SIMULATOR ======================

@dataclass
class PopulationResult:
    native_mass: float
    molecule_masses: np.ndarray          # one entry per virtual molecule, pre-instrument
    observed_masses: np.ndarray          # after instrument broadening
    modification_log: List[List[dict]]   # per-molecule event lists (for QC/export)


class Simulator:
    def __init__(self, engine: Optional[ReactionEngine] = None,
                 instrument: Optional[InstrumentModel] = None):
        self.engine = engine or ReactionEngine()
        self.instrument = instrument or InstrumentModel()

    def run_population(
        self,
        protein: Protein,
        metabolites: List[Metabolite],
        pH: float = 7.4,
        n_molecules: int = 5000,
        exposure_days: float = 7.0,
        use_turnover: bool = False,
        half_life_days: float = 19.0,
    ) -> PopulationResult:
        """Monte Carlo over N independent virtual copies of the protein.
        Each copy accumulates zero or more modifications; its total mass is
        native_mass + sum(mass_shift over its own events). Only THEN do we
        histogram — one point per molecule, matching what MALDI actually
        measures (a population of intact proteoforms)."""
        native_mass = protein.native_mass()
        molecule_masses = np.empty(n_molecules)
        log: List[List[dict]] = []

        for i in range(n_molecules):
            if use_turnover:
                # first-order clearance -> exponential age distribution
                mean_life_s = (half_life_days / np.log(2)) * 86400
                exposure_s = np.random.exponential(mean_life_s)
            else:
                exposure_s = exposure_days * 86400

            copy = protein.fresh_copy()
            events = self.engine.react_one_molecule(copy, metabolites, pH, exposure_s)
            total_shift = sum(e["mass_shift"] for e in events)
            molecule_masses[i] = native_mass + total_shift
            log.append(events)

        observed = self.instrument.broaden(molecule_masses)

        return PopulationResult(
            native_mass=native_mass,
            molecule_masses=molecule_masses,
            observed_masses=observed,
            modification_log=log,
        )


def scenario_metabolites(scenario: str, base: Optional[List[Metabolite]] = None) -> List[Metabolite]:
    """Return metabolite list with concentrations scaled by scenario multipliers
    (multipliers act on concentration, i.e. plasma exposure — not on mass shift
    and not on kon)."""
    base = base or DEFAULT_METABOLITES
    mult = SCENARIOS.get(scenario, SCENARIOS["Normal"])
    out = []
    for m in base:
        factor = mult.get(m.name, 1.0)
        out.append(Metabolite(
            name=m.name, formula=m.formula, exact_mass=m.exact_mass,
            mass_shift=m.mass_shift, conc_mM=m.conc_mM * factor, kon=m.kon,
            targets=m.targets, description=m.description, mechanism=m.mechanism,
            reversible=m.reversible, evidence=m.evidence, reference=m.reference,
        ))
    return out
