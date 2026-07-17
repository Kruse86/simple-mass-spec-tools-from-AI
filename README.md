# CADB v6.0

Forward kinetic model of carbonyl-driven albumin proteoform formation, predicting
intact-mass distributions comparable to (DTT-reduced, sinapinic-acid, linear
positive-mode) MALDI-TOF spectra.

## Files
- `core.py` — all modelling logic (chemistry, kinetics, Monte Carlo population
  simulation, instrument response). No GUI dependency; importable and unit-testable
  on its own.
- `gui.py` — Tkinter interface wrapping `core.py`. Run with `python3 gui.py`
  (requires `python3-tk`: `sudo apt install python3-tk` if missing).

## What changed from v5.2, and why

1. **Population vs. single-molecule conflation (the important fix).**
   v5.2 histogrammed individual *residue-modification events* as if each were
   a separate protein molecule's mass. That doesn't simulate a proteoform
   population — it just re-plots the reaction list. `Simulator.run_population()`
   now runs N independent virtual copies of the protein, sums *all* modifications
   that occur on a given copy into one total mass, and only histograms those N
   per-molecule totals. This is what actually corresponds to "a population of
   albumin molecules with heterogeneous modification patterns," which is what
   MALDI is measuring.

2. **`reactive_fraction()` formula bug.** v5.2 branched between two different
   expressions depending on whether pKa was above or below 7. The correct
   Henderson–Hasselbalch fraction of the reactive/neutral species is the same
   expression regardless of pKa; there was no chemical reason for the branch.

3. **Arginine decoupled from amine-type pKa gating.** Methylglyoxal/glyoxal
   hydroimidazolone formation on Arg proceeds via the guanidinium group in its
   normal protonation state — it is not gated by deprotonation the way Lys
   Schiff-base chemistry is. Treating Arg identically to Lys was a mechanistic
   error, not a simplification.

4. **Competing-risk resolution between metabolites.** v5.2 looped through
   metabolites sequentially and locked a residue on the first success, biasing
   outcomes toward whichever metabolite happened to be first in the list.
   `ReactionEngine.react_one_molecule()` now sums per-metabolite rates for each
   residue and, when a reaction occurs, chooses which metabolite "won" with
   probability proportional to its rate — standard competing-exponentials
   treatment.

5. **Instrument broadening reinstated.** Present in the earlier procedural
   scripts, silently dropped during the v5.x OOP rewrite. `InstrumentModel`
   applies mass-dependent Gaussian broadening (`FWHM = mass / resolving_power`),
   which is a closer (if still first-order) approximation of linear-mode
   MALDI-TOF resolution loss than a fixed-width Gaussian.

6. **Optional turnover model.** Real circulating albumin is a steady-state
   mixture of molecule ages (plasma half-life ≈ 19 days), not one fixed
   exposure time. `run_population(..., use_turnover=True)` samples each virtual
   molecule's exposure time from an exponential distribution with that mean
   life, instead of applying the same fixed exposure to every copy.

7. **Parameter provenance.** Every `Metabolite` carries an `evidence` field.
   As of v6 every default metabolite's `kon`/concentration is explicitly
   labelled as an order-of-magnitude placeholder, calibrated only so the
   population doesn't saturate or stay at 0% modified over a plausible exposure
   window — **not** fit to literature or experimental rate constants. Do not
   quote these numbers as calibrated kinetics in a manuscript without replacing
   them.

## Known remaining limitations (not fixed in this pass — flagging for later)

- Residue accessibility is a per-amino-acid-type constant (`DEFAULT_ACCESSIBILITY`
  in `core.py`), not derived from structure. Replacing this with real SASA values
  (e.g. from a PDB model or AlphaFold prediction via FreeSASA/DSSP) is the
  single highest-value next step for quantitative credibility.
- No charge-state (1+/2+) conversion or isotope envelope — output is neutral
  monoisotopic-equivalent mass only.
- Each metabolite contributes one lumped, irreversible mass shift; reversible
  Schiff-base intermediates and multi-step AGE cross-linking are not modelled
  explicitly.
- Scenario multipliers (`SCENARIOS` in `core.py`) act on metabolite
  *concentration*, which is chemically correct, but the multiplier values
  themselves are illustrative, not derived from measured plasma metabolomics
  in the relevant patient groups.

## Quick start

```bash
python3 gui.py
```
Paste a sequence (or click "Load HSA Example"), pick a scenario/pH/kinetic
model, set population size, and run. Export the histogram as PNG or the raw
per-molecule data as CSV (native mass, observed mass, modification count,
which metabolites) for downstream stats (centroid, FWHM, skewness comparisons
against your real spectra).
