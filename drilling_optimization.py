import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import math
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import numpy as np

# ---------------- Core Classes (physics + wear + cost) ----------------

class DrillBit:
    def __init__(self, bit_type: str, diameter_inch: float, wear_percent: float):
        self.bit_type = bit_type
        self.diameter = diameter_inch  # inches
        # store wear as fraction 0..1
        self.initial_wear = max(0.0, min(0.99, wear_percent / 100.0))
        # simple bit replacement cost baseline by bit type (USD)
        t = bit_type.lower()
        if "pdc" in t:
            self.replace_cost = 15000.0
        elif "tricone" in t:
            self.replace_cost = 8000.0
        elif "diamond" in t:
            self.replace_cost = 25000.0
        else:
            self.replace_cost = 10000.0

    def efficiency_from_wear(self, wear_frac):
        # efficiency falls off as wear increases, floor at 0.25
        return max(0.25, 1.0 - wear_frac)


class Mud:
    def __init__(self, density_ppg: float, viscosity_cp: float, flow_gpm: float):
        self.density = density_ppg
        self.viscosity = viscosity_cp
        self.flow = flow_gpm

    def hydraulic_efficiency(self):
        """
        Simplified mud hydraulics efficiency:
        - Higher viscosity reduces efficiency
        - Higher flow improves efficiency
        - Density moderate values are best (too low or too high reduce efficiency)
        Returns efficiency in (0.15 .. 1.0)
        """
        dens_score = 1.0 - abs(self.density - 11.0) / 10.0  # best near 11 ppg
        visc_score = max(0.0, 1.2 - self.viscosity / 80.0)  # high viscosity penalized
        flow_score = 0.5 + (self.flow / 2000.0)  # more flow helps
        eff = dens_score * visc_score * flow_score
        return max(0.15, min(1.0, eff))

    def pressure_loss(self, depth_ft, bore_diameter_inch):
        hydrostatic = 0.052 * self.density * depth_ft
        # small demo friction term
        area = math.pi * ((bore_diameter_inch * 0.0254) ** 2) / 4.0  # m^2 approximate
        if area <= 0: area = 1e-6
        friction = (self.viscosity / 100.0) * ((self.flow / max(area, 1e-6)) ** 2) * (0.00000001 * (25 / (bore_diameter_inch + 0.1)))
        return hydrostatic + friction


class Wellbore:
    def __init__(self, depth_ft: float, rock_hardness: float, pore_pressure_psi: float):
        self.depth = depth_ft
        # map hardness 1->0.6, 10->3.0
        self.rock_hardness = rock_hardness
        self.cs_factor = 0.6 + (rock_hardness - 1) * ((3.0 - 0.6) / 9.0)
        self.pore_pressure = pore_pressure_psi


class DrillingOptimizer:
    def __init__(self, bit: DrillBit, mud: Mud, well: Wellbore):
        self.bit = bit
        self.mud = mud
        self.well = well

    def _bit_wear_rate(self, WOB_tons, RPM, mud_eff):
        # normalized factors
        wob_factor = WOB_tons / 20.0
        rpm_factor = RPM / 150.0
        cs = self.well.cs_factor
        base = 0.0008
        wear_rate = base * wob_factor * rpm_factor * cs / max(0.2, mud_eff)
        # bit-type modifier
        type_mod = 1.0
        t = self.bit.bit_type.lower()
        if "pdc" in t: type_mod = 1.0
        elif "tricone" in t: type_mod = 1.1
        elif "diamond" in t: type_mod = 0.8
        return wear_rate * type_mod  # fraction per hour

    def _rop_model(self, WOB_tons, RPM, effective_wear, mud_eff):
        k_wob = 0.8
        # ROP base
        rop_base = k_wob * (WOB_tons ** 0.7) * (RPM ** 0.3)
        rock_penalty = 1.0 / self.well.cs_factor
        wear_eff = self.bit.efficiency_from_wear(effective_wear)
        rop = rop_base * rock_penalty * wear_eff * mud_eff * 0.8
        return max(0.01, rop)

    def _estimate_bit_life_hours(self, current_wear_frac, wear_rate_per_hr, fail_wear_frac=0.7):
        # estimate hours until wear reaches failure threshold (e.g., 70% wear)
        if wear_rate_per_hr <= 0:
            return float('inf')
        remaining = max(0.0, fail_wear_frac - current_wear_frac)
        if remaining <= 0:
            return 0.0
        return remaining / wear_rate_per_hr

    def _estimate_cost(self, WOB_tons, RPM, simulate_hours, rop_ft_per_hr):
        """
        Simple cost model:
         - rig operating cost per hour (USD)
         - bit replacement cost amortized by its estimated life (hours)
         - mud cost proportional to flow rate and density
         - extra consumables proportional to WOB/RPM
        Returns estimated total cost per simulated hour and cost_per_ft (USD/ft)
        """
        rig_cost_per_hr = 2000.0  # USD/hr (example)
        # mud consumable cost proportional to flow and density
        mud_cost_per_hr = 0.02 * self.mud.flow * self.mud.density
        # extra wear-related consumables
        consumables = 10.0 * (WOB_tons / 10.0) + 0.5 * RPM
        # estimate bit life (hours) by current operating stress (approx)
        wear_rate = self._bit_wear_rate(WOB_tons, RPM, self.mud.hydraulic_efficiency())
        est_bit_life_hours = self._estimate_bit_life_hours(self.bit.initial_wear, wear_rate, fail_wear_frac=0.7)
        # amortize replacement cost across estimated bit life (if infinite, treat large)
        if est_bit_life_hours <= 0 or math.isinf(est_bit_life_hours):
            bit_amort_per_hr = 0.0
        else:
            bit_amort_per_hr = self.bit.replace_cost / est_bit_life_hours
        total_cost_per_hr = rig_cost_per_hr + mud_cost_per_hr + consumables + bit_amort_per_hr
        # cost per foot (avoid division by zero)
        eps = 1e-6
        cost_per_ft = total_cost_per_hr / max(eps, rop_ft_per_hr)
        return total_cost_per_hr, cost_per_ft

    def evaluate_candidate(self, WOB, RPM, simulate_hours=1.0):
        mud_eff = self.mud.hydraulic_efficiency()
        init_wear = self.bit.initial_wear
        wear_rate = self._bit_wear_rate(WOB, RPM, mud_eff)
        wear_after = min(0.99, init_wear + wear_rate * simulate_hours)
        rop = self._rop_model(WOB, RPM, wear_after, mud_eff)
        # bit life estimate in hours
        bit_life = self._estimate_bit_life_hours(init_wear, wear_rate, fail_wear_frac=0.7)
        # cost estimates
        total_cost_per_hr, cost_per_ft = self._estimate_cost(WOB, RPM, simulate_hours, rop)
        # convert cost per ft into a "cost efficiency" metric where higher is better for Pareto:
        cost_eff = 1.0 / (cost_per_ft + 1e-9)
        return {
            "WOB": WOB,
            "RPM": RPM,
            "ROP": rop,
            "WearRate_per_hr": wear_rate,
            "WearAfter": wear_after,
            "BitLife_hours": bit_life,
            "MudEff": mud_eff,
            "CostPerHr": total_cost_per_hr,
            "CostPerFt": cost_per_ft,
            "CostEff": cost_eff
        }

    def optimize_multiobj(self, wob_range=None, rpm_range=None, simulate_hours=1.0):
        if wob_range is None:
            wob_range = list(range(5, 31, 5))  # tons
        if rpm_range is None:
            rpm_range = list(range(60, 201, 20))

        candidates = []
        for WOB in wob_range:
            for RPM in rpm_range:
                c = self.evaluate_candidate(WOB, RPM, simulate_hours)
                candidates.append(c)

        # Pareto front (non-dominated) - now maximizing ROP, BitLife, MudEff, CostEff
        pareto = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is c:
                    continue
                better_or_equal = (
                    (other["ROP"] >= c["ROP"]) and
                    (other["BitLife_hours"] >= c["BitLife_hours"]) and
                    (other["MudEff"] >= c["MudEff"]) and
                    (other["CostEff"] >= c["CostEff"])
                )
                strictly_better = (
                    (other["ROP"] > c["ROP"]) or
                    (other["BitLife_hours"] > c["BitLife_hours"]) or
                    (other["MudEff"] > c["MudEff"]) or
                    (other["CostEff"] > c["CostEff"])
                )
                if better_or_equal and strictly_better:
                    dominated = True
                    break
            if not dominated:
                pareto.append(c)

        # sort pareto by ROP descending for display
        pareto_sorted = sorted(pareto, key=lambda x: x["ROP"], reverse=True)
        # also find single-objective best (max ROP)
        best_by_rop = max(candidates, key=lambda x: x["ROP"])
        return candidates, pareto_sorted, best_by_rop

# ---------------- GUI Application (with Pareto & Cost) ----------------

class DrillingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drilling Optimization - Multi-Objective (Pareto + Cost)")
        self.root.geometry("1000x720")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Tabs
        self.input_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.charts_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.input_tab, text="Input Parameters")
        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.charts_tab, text="Charts (Extra)")

        # storage
        self.candidates = []
        self.pareto = []
        self.best_by_rop = None

        # build UI
        self._build_input_tab()
        self._build_results_tab()
        self._build_charts_tab()

    def _build_input_tab(self):
        frame = ttk.Frame(self.input_tab, padding=12)
        frame.pack(fill="both", expand=True)

        labels = [
            "Drill Bit Type (PDC/Tricone/Diamond)",
            "Diameter (inch)",
            "Initial Bit Wear (%)",
            "Mud Density (ppg)",
            "Mud Viscosity (cP)",
            "Mud Flow Rate (gpm)",
            "Well Depth (ft)",
            "Rock Hardness (1-10)",
            "Pore Pressure (psi)",
            "Simulation Hours (for wear)"
        ]
        defaults = ["PDC", "8.5", "10", "11.0", "40", "600", "8000", "4", "3000", "1.0"]

        self.inputs = []
        for i, (label, default) in enumerate(zip(labels, defaults)):
            ttk.Label(frame, text=label + ":").grid(row=i, column=0, sticky="w", pady=4)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=22)
            entry.grid(row=i, column=1, pady=4, padx=6)
            self.inputs.append(var)

        # Grid ranges (optional controls)
        ttk.Label(frame, text="WOB min,max,step (tons)").grid(row=0, column=2, sticky="w", padx=10)
        self.wob_range_var = tk.StringVar(value="5,30,5")
        ttk.Entry(frame, textvariable=self.wob_range_var, width=18).grid(row=0, column=3, padx=6)

        ttk.Label(frame, text="RPM min,max,step").grid(row=1, column=2, sticky="w", padx=10)
        self.rpm_range_var = tk.StringVar(value="60,200,20")
        ttk.Entry(frame, textvariable=self.rpm_range_var, width=18).grid(row=1, column=3, padx=6)

        ttk.Button(frame, text="Run Optimization (Pareto)", command=self.run_optimizer).grid(row=12, column=0, columnspan=2, pady=12)
        ttk.Button(frame, text="Export Full Grid + Pareto CSV", command=self.export_all_csv).grid(row=12, column=2, columnspan=2, pady=12)

        # quick presets
        preset_frame = ttk.LabelFrame(frame, text="Presets")
        preset_frame.grid(row=13, column=0, columnspan=4, pady=10, sticky="ew")
        ttk.Button(preset_frame, text="Soft", command=self.load_preset_soft).pack(side="left", padx=6, pady=6)
        ttk.Button(preset_frame, text="Medium", command=self.load_preset_medium).pack(side="left", padx=6, pady=6)
        ttk.Button(preset_frame, text="Hard", command=self.load_preset_hard).pack(side="left", padx=6, pady=6)

    def _build_results_tab(self):
        frame = ttk.Frame(self.results_tab, padding=8)
        frame.pack(fill="both", expand=True)

        # Best-by-ROP area
        ttk.Label(frame, text="Best (Max ROP) Candidate", font=("Arial", 12, "bold")).pack(anchor="w")
        columns = ("Parameter", "Value")
        self.best_table = ttk.Treeview(frame, columns=columns, show="headings", height=4)
        self.best_table.heading("Parameter", text="Parameter")
        self.best_table.heading("Value", text="Value")
        self.best_table.pack(fill="x", pady=6)

        # Pareto area
        ttk.Label(frame, text="Pareto Front (non-dominated solutions)", font=("Arial", 12, "bold")).pack(anchor="w", pady=(8,0))
        pareto_cols = ("WOB", "RPM", "ROP (ft/hr)", "BitLife (hrs)", "MudEff", "CostPerFt")
        self.pareto_table = ttk.Treeview(frame, columns=pareto_cols, show="headings", height=8)
        for c in pareto_cols:
            self.pareto_table.heading(c, text=c)
            self.pareto_table.column(c, width=110, anchor="center")
        self.pareto_table.pack(fill="x", pady=6)

        # Summary label
        self.summary_var = tk.StringVar(value="Run optimization to compute Pareto front.")
        ttk.Label(frame, textvariable=self.summary_var, foreground="blue").pack(pady=4)

        # Chart canvas
        chart_frame = ttk.Frame(frame)
        chart_frame.pack(fill="both", expand=True, pady=6)
        self.fig, self.ax = plt.subplots(figsize=(8,5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # Buttons for plotting
        plot_btn_frame = ttk.Frame(frame)
        plot_btn_frame.pack(fill="x", pady=6)
        ttk.Button(plot_btn_frame, text="Plot Pareto 3D", command=self.plot_pareto_3d).pack(side="left", padx=6)
        ttk.Button(plot_btn_frame, text="Plot Pareto 2D (ROP vs BitLife)", command=self.plot_pareto_2d).pack(side="left", padx=6)

    def _build_charts_tab(self):
        frame = ttk.Frame(self.charts_tab, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Extra charts (future): torque, pressure profile, bit-life curves").pack()

    # Presets
    def load_preset_soft(self):
        vals = ["PDC", "8.5", "5", "9.5", "35", "500", "3500", "2", "1500", "1.0"]
        for var, v in zip(self.inputs, vals):
            var.set(v)

    def load_preset_medium(self):
        vals = ["PDC", "12.25", "20", "11.0", "42", "650", "7500", "5", "2800", "1.0"]
        for var, v in zip(self.inputs, vals):
            var.set(v)

    def load_preset_hard(self):
        vals = ["Tricone", "17.5", "40", "13.2", "55", "850", "12000", "8", "5000", "1.0"]
        for var, v in zip(self.inputs, vals):
            var.set(v)

    # Utility: parse range string "min,max,step" -> list
    def _parse_range(self, s, cast=int, default=(5,30,5)):
        try:
            parts = [cast(x.strip()) for x in s.split(",")]
            if len(parts) == 3:
                mn, mx, step = parts
                if step <= 0 or mn > mx:
                    return default
                return list(range(mn, mx+1, step))
            else:
                return default
        except Exception:
            return default

    # Run the multi-objective optimizer
    def run_optimizer(self):
        try:
            bit_type = self.inputs[0].get()
            diameter = float(self.inputs[1].get())
            wear_pct = float(self.inputs[2].get())
            mud_density = float(self.inputs[3].get())
            viscosity = float(self.inputs[4].get())
            flow_rate = float(self.inputs[5].get())
            depth = float(self.inputs[6].get())
            rock_hardness = float(self.inputs[7].get())
            pore_pressure = float(self.inputs[8].get())
            sim_hours = float(self.inputs[9].get())

            bit = DrillBit(bit_type, diameter, wear_pct)
            mud = Mud(mud_density, viscosity, flow_rate)
            well = Wellbore(depth, rock_hardness, pore_pressure)
            optimizer = DrillingOptimizer(bit, mud, well)

            wob_range = self._parse_range(self.wob_range_var.get(), int, default=(5,30,5))
            rpm_range = self._parse_range(self.rpm_range_var.get(), int, default=(60,200,20))

            # run multiobjective optimization
            candidates, pareto_sorted, best_by_rop = optimizer.optimize_multiobj(wob_range=wob_range, rpm_range=rpm_range, simulate_hours=sim_hours)

            self.candidates = candidates
            self.pareto = pareto_sorted
            self.best_by_rop = best_by_rop

            # update best table
            for r in self.best_table.get_children():
                self.best_table.delete(r)
            best_info = [
                ("WOB (tons)", best_by_rop["WOB"]),
                ("RPM", best_by_rop["RPM"]),
                ("ROP (ft/hr)", f"{best_by_rop['ROP']:.3f}"),
                ("BitLife (hrs est)", f"{best_by_rop['BitLife_hours']:.2f}"),
                ("MudEff", f"{best_by_rop['MudEff']:.3f}"),
                ("CostPerFt (USD/ft)", f"{best_by_rop['CostPerFt']:.3f}")
            ]
            for p, v in best_info:
                self.best_table.insert("", "end", values=(p, v))

            # update pareto table
            for r in self.pareto_table.get_children():
                self.pareto_table.delete(r)
            for sol in self.pareto:
                self.pareto_table.insert("", "end", values=(
                    sol["WOB"], sol["RPM"], f"{sol['ROP']:.3f}", f"{sol['BitLife_hours']:.2f}", f"{sol['MudEff']:.3f}", f"{sol['CostPerFt']:.4f}"
                ))

            self.summary_var.set(f"Found {len(self.pareto)} Pareto-optimal solutions (from {len(self.candidates)} candidates). Best ROP = {best_by_rop['ROP']:.3f} ft/hr")

            # plot embedded Pareto (2D)
            self._plot_embedded_pareto()

            # switch to results tab
            self.notebook.select(self.results_tab)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _plot_embedded_pareto(self):
        # default embedded plot: ROP vs BitLife with Pareto highlighted (color = cost efficiency)
        if not self.candidates:
            return
        self.ax.clear()

        # Plot candidates in background: ROP vs BitLife
        x_all = [c["ROP"] for c in self.candidates]
        y_all = [c["BitLife_hours"] for c in self.candidates]
        self.ax.scatter(x_all, y_all, color="lightgray", alpha=0.4, s=20, label="All Candidates")

        if self.pareto:
            # Pareto points colored by cost efficiency (CostEff)
            cost_vals = [p["CostEff"] for p in self.pareto]
            norm = plt.Normalize(min(cost_vals), max(cost_vals))
            cmap = plt.get_cmap("viridis")

            sc = self.ax.scatter(
                [p["ROP"] for p in self.pareto],
                [p["BitLife_hours"] for p in self.pareto],
                c=cost_vals,
                cmap=cmap,
                s=80,
                edgecolor="k",
                label="Pareto Front"
            )

            # add colorbar for cost efficiency
            try:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array(cost_vals)
                cbar = self.fig.colorbar(sm, ax=self.ax)
                cbar.set_label("Cost Efficiency (1 / USD per ft)")
            except Exception:
                pass

        self.ax.set_xlabel("ROP (ft/hr)")
        self.ax.set_ylabel("Estimated Bit Life (hrs)")
        self.ax.set_title("Pareto: ROP vs Bit Life (color = Cost Efficiency)")
        self.ax.grid(True)

        # Legend cleanup
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
        self.canvas.draw()

    def plot_pareto_3d(self):
        if not self.pareto:
            messagebox.showwarning("Warning", "Run the optimizer first to generate Pareto data.")
            return

        rop_vals = [p["ROP"] for p in self.pareto]
        life_vals = [p["BitLife_hours"] for p in self.pareto]
        costeff_vals = [p["CostEff"] for p in self.pareto]

        fig = plt.figure(figsize=(8,6))
        ax3 = fig.add_subplot(111, projection="3d")
        sc = ax3.scatter(rop_vals, life_vals, costeff_vals, c=rop_vals, cmap="viridis", s=80)
        ax3.set_xlabel("ROP (ft/hr)")
        ax3.set_ylabel("Bit Life (hrs)")
        ax3.set_zlabel("Cost Efficiency (1 / USD/ft)")
        plt.colorbar(sc, label="ROP")
        plt.title("Pareto Frontier: ROP vs Bit Life vs CostEff (3D)")
        plt.show()

    def plot_pareto_2d(self):
        if not self.pareto:
            messagebox.showwarning("Warning", "Run the optimizer first to generate Pareto data.")
            return

        # 2D scatter: ROP vs BitLife, highlight Pareto colored by cost efficiency
        x_all = [c["ROP"] for c in self.candidates]
        y_all = [c["BitLife_hours"] for c in self.candidates]

        plt.figure(figsize=(8,6))
        plt.scatter(x_all, y_all, color="lightgray", alpha=0.6, s=20, label="Candidates")

        cost_vals = [p["CostEff"] for p in self.pareto]
        norm = plt.Normalize(min(cost_vals), max(cost_vals))
        cmap = plt.get_cmap("plasma")
        for p in self.pareto:
            plt.scatter(p["ROP"], p["BitLife_hours"], color=cmap(norm(p["CostEff"])), s=100, edgecolor="k", label=f"W{p['WOB']} R{p['RPM']}")

        plt.xlabel("ROP (ft/hr)")
        plt.ylabel("Estimated Bit Life (hrs)")
        plt.title("Pareto Front (2D): ROP vs BitLife (colored by CostEff)")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(cost_vals)
        plt.colorbar(sm, label="Cost Efficiency (1 / USD/ft)")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
        plt.grid(True)
        plt.show()

    # Export CSV (full grid and pareto tag + cost fields)
    def export_all_csv(self):
        if not self.candidates:
            messagebox.showwarning("No Data", "Run optimization first to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path:
            return
        try:
            pareto_set = set((p["WOB"], p["RPM"]) for p in self.pareto)
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["WOB_tons","RPM","ROP_ft_per_hr","WearRate_per_hr","WearAfter","BitLife_hrs","MudEff","CostPerHr","CostPerFt","IsPareto"])
                for c in self.candidates:
                    is_p = (c["WOB"], c["RPM"]) in pareto_set
                    writer.writerow([
                        c["WOB"], c["RPM"], f"{c['ROP']:.6f}", f"{c['WearRate_per_hr']:.8f}",
                        f"{c['WearAfter']:.6f}", f"{c['BitLife_hours']:.3f}", f"{c['MudEff']:.4f}",
                        f"{c['CostPerHr']:.2f}", f"{c['CostPerFt']:.6f}", int(is_p)
                    ])
            messagebox.showinfo("Saved", f"Exported to {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


# ---------------- Run App ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = DrillingApp(root)
    root.mainloop()
