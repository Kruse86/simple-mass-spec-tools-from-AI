#!/usr/bin/env python3
"""
CADB v6.0 — GUI

Wraps core.py (Protein / Simulator / ReactionEngine / InstrumentModel) in a
Tkinter interface. All modelling logic lives in core.py; this file is
presentation + I/O only.
"""

import csv
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core import (
    Protein, Simulator, ReactionEngine, InstrumentModel,
    SecondOrderModel, MichaelisModel,
    DEFAULT_METABOLITES, SCENARIOS, scenario_metabolites,
)

EXAMPLE_HSA = (
    "MKWVTFISLLFLFSSAYSRGVFRRDAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTC"
    "VADESAENCDKSLHTLFGDKLCTVATLRETYGEMADCCAKQEPERNECFLQHKDDNPNLPRLVRPEVDVMCTAFHDN"
    "EETFLKKYLYEIARRHPYFYAPELLFFAKRYKAAFTECCQAADKAACLLPKLDELRDEGKASSAKQRLKCASLQKFG"
    "ERAFKAWAVARLSQRFPKAEFAEVSKLVTDLTKVHTECCHGDLLECADDRADLAKYICENQDSISSKLKECCEKPLL"
    "EKSHCIAEVENDEMPADLPSLAADFVESKDVCKNYAEAKDVFLGMFLYEYARRHPDYSVVLLLRLAKTYETTLEKCC"
    "AAADPHECYAKVFDEFKPLVEEPQNLIKQNCELFEQLGEYKFQNALLVRYTKKVPQVSTPTLVEVSRNLGKVGSKCC"
    "KHPEAKRMPCAEDYLSVVLNQLCVLHEKTPVSDRVTKCCTESLVNRRPCFSALEVDETYVPKEFNAETFTFHADICT"
    "LSEKERQIKKQTALVELVKHKPKATKEQLKAVMDDFAAFVEKCCKADDKETCFAEEGKKLVAASQAALGL"
)


class CADBGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CADB v6.0 — Carbonyl Adduct Proteoform Simulator")
        self.root.geometry("1260x900")
        self.last_result = None
        self.last_protein = None

        tk.Label(root, text="CADB v6.0 — Carbonyl-Driven Albumin Proteoform Simulator",
                 font=("Arial", 16, "bold")).pack(pady=10)
        tk.Label(root, text="Illustrative forward model — kinetic parameters are order-of-"
                             "magnitude placeholders, not literature-calibrated. See metabolite "
                             "evidence fields.", font=("Arial", 9, "italic"), fg="#a03").pack()

        tk.Label(root, text="Amino acid sequence (FASTA body, no header needed):").pack(anchor="w", padx=25, pady=(10, 0))
        self.seq_text = scrolledtext.ScrolledText(root, height=8, width=145, font=("Consolas", 10))
        self.seq_text.pack(padx=25, pady=6)

        ctrl = tk.Frame(root)
        ctrl.pack(pady=8)

        tk.Label(ctrl, text="Kinetic model:").grid(row=0, column=0, padx=4, sticky="e")
        self.model_var = tk.StringVar(value="Second-Order (mass-action)")
        ttk.Combobox(ctrl, values=["Second-Order (mass-action)", "Michaelis-Menten (empirical)"],
                     textvariable=self.model_var, state="readonly", width=26).grid(row=0, column=1, padx=4)

        tk.Label(ctrl, text="Scenario:").grid(row=0, column=2, padx=4, sticky="e")
        self.scenario_var = tk.StringVar(value="Normal")
        ttk.Combobox(ctrl, values=list(SCENARIOS.keys()), textvariable=self.scenario_var,
                     state="readonly", width=20).grid(row=0, column=3, padx=4)

        tk.Label(ctrl, text="pH:").grid(row=0, column=4, padx=4, sticky="e")
        self.pH = tk.DoubleVar(value=7.4)
        tk.Scale(ctrl, from_=6.8, to=7.6, resolution=0.05, variable=self.pH,
                 orient=tk.HORIZONTAL, length=140).grid(row=0, column=5, padx=4)

        tk.Label(ctrl, text="Population size:").grid(row=1, column=0, padx=4, pady=6, sticky="e")
        self.n_molecules = tk.IntVar(value=5000)
        tk.Entry(ctrl, textvariable=self.n_molecules, width=8).grid(row=1, column=1, sticky="w")

        self.turnover_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text="Use 19-day turnover (exponential age) instead of fixed exposure",
                       variable=self.turnover_var, command=self._toggle_exposure).grid(row=1, column=2, columnspan=3, sticky="w")

        tk.Label(ctrl, text="Exposure (days):").grid(row=1, column=5, padx=4, sticky="e")
        self.exposure_days = tk.DoubleVar(value=7.0)
        self.exposure_entry = tk.Entry(ctrl, textvariable=self.exposure_days, width=6)
        self.exposure_entry.grid(row=1, column=6, sticky="w")

        btns = tk.Frame(root)
        btns.pack(pady=8)
        tk.Button(btns, text="Run Simulation", command=self.run_simulation,
                  bg="#2196F3", fg="white", font=("Arial", 11, "bold"), width=18).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Load HSA Example", command=self.load_example,
                  bg="#FF9800", fg="white", width=18).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Export Plot (PNG)", command=self.export_png,
                  bg="#4CAF50", fg="white", width=16).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Export Data (CSV)", command=self.export_csv,
                  bg="#4CAF50", fg="white", width=16).pack(side=tk.LEFT, padx=6)
        tk.Button(btns, text="Clear", command=self.clear, bg="#f44336", fg="white", width=10).pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(root, text="", font=("Arial", 11))
        self.status.pack(pady=6)

        self.fig, self.ax = plt.subplots(figsize=(11.5, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=25, pady=8)

    def _toggle_exposure(self):
        self.exposure_entry.configure(state="disabled" if self.turnover_var.get() else "normal")

    def load_example(self):
        self.seq_text.delete("1.0", tk.END)
        self.seq_text.insert(tk.END, EXAMPLE_HSA)

    def clear(self):
        self.seq_text.delete("1.0", tk.END)
        self.status.config(text="")
        self.ax.clear()
        self.canvas.draw()
        self.last_result = None

    def run_simulation(self):
        seq = self.seq_text.get("1.0", tk.END).strip()
        if len(seq) < 20:
            messagebox.showwarning("Input", "Enter a protein sequence (>= 20 residues).")
            return

        protein = Protein("User protein", seq)
        n_lys = protein.sequence.count("K")
        n_arg = protein.sequence.count("R")

        model = (SecondOrderModel() if self.model_var.get().startswith("Second-Order")
                 else MichaelisModel())
        engine = ReactionEngine(model)
        sim = Simulator(engine=engine, instrument=InstrumentModel())

        metabolites = scenario_metabolites(self.scenario_var.get(), DEFAULT_METABOLITES)

        n_mol = max(200, int(self.n_molecules.get()))
        use_turnover = self.turnover_var.get()

        self.status.config(text="Running Monte Carlo population simulation...")
        self.root.update_idletasks()

        result = sim.run_population(
            protein, metabolites,
            pH=self.pH.get(),
            n_molecules=n_mol,
            exposure_days=self.exposure_days.get(),
            use_turnover=use_turnover,
        )

        self.last_result = result
        self.last_protein = protein

        n_events = np.array([len(e) for e in result.modification_log])
        frac_unmod = (n_events == 0).mean()
        mean_shift = result.observed_masses.mean() - result.native_mass

        exposure_desc = "19-day turnover (exponential age)" if use_turnover else f"{self.exposure_days.get():.1f} d fixed exposure"
        self.status.config(
            text=(f"n={n_mol} molecules | Native: {result.native_mass:.1f} Da | "
                  f"Mean observed shift: {mean_shift:+.1f} Da | "
                  f"Unmodified fraction: {frac_unmod:.1%} | "
                  f"K={n_lys}, R={n_arg} | {exposure_desc}")
        )

        self.plot(result)

    def plot(self, result):
        self.ax.clear()
        self.ax.hist(result.observed_masses, bins=140, alpha=0.85, color="#3498db",
                     edgecolor="black", linewidth=0.3, label="Simulated proteoform population\n(instrument-broadened)")
        self.ax.axvline(result.native_mass, color="red", linestyle="--", alpha=0.8)
        self.ax.annotate(f"Native\n{result.native_mass:.1f} Da",
                          xy=(result.native_mass, 0), xytext=(0, 40),
                          textcoords="offset points", ha="center", color="red", fontsize=11,
                          arrowprops=dict(arrowstyle="->", color="red"))
        self.ax.set_xlabel("Molecular mass (Da)", fontsize=12)
        self.ax.set_ylabel("Molecule count", fontsize=12)
        self.ax.set_title(f"Predicted intact-mass proteoform distribution — {self.scenario_var.get()} "
                           f"(pH {self.pH.get():.2f})", fontsize=13)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc="upper right", fontsize=9)
        self.canvas.draw()

    def export_png(self):
        if self.last_result is None:
            messagebox.showwarning("Export", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.fig.savefig(path, dpi=400, bbox_inches="tight")
            messagebox.showinfo("Export", f"Saved: {path}")

    def export_csv(self):
        if self.last_result is None:
            messagebox.showwarning("Export", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        result = self.last_result
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["molecule_index", "true_mass_da", "observed_mass_da", "n_modifications", "metabolites"])
            for i, (true_m, obs_m, events) in enumerate(
                zip(result.molecule_masses, result.observed_masses, result.modification_log)
            ):
                metas = ";".join(e["metabolite"] for e in events)
                writer.writerow([i, f"{true_m:.3f}", f"{obs_m:.3f}", len(events), metas])
        messagebox.showinfo("Export", f"Saved: {path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = CADBGUI(root)
    root.mainloop()
