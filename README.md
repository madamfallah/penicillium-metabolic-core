# Penicillium chrysogenum Metabolic Core Analyzer 🧬

A production-grade Python tool built to parse, filter, and analyze Genome-Scale Metabolic Models (GEMs) of *Penicillium chrysogenum*. This project bridges the gap between systems biology (FBA constraint-based modeling) and data science, preparing metabolic flux data for downstream Machine Learning applications.

## 🚀 Key Features
- **Object-Oriented Design (OOP):** Built using modern Python practices with full type-hinting.
- **Automated Exchange Filtering:** Programmatically isolates core exchange reactions (the boundary constraints of the cell) from 1,600+ metabolic pathways.
- **Biomass & Objective Tracking:** Automatically identifies the model's objective function (e.g., `EX_bm` for growth optimization).

## 📊 Dataset Structure
The analyzer operates on validated genome-scale metabolic reconstructions containing:
- **1,632 Metabolic Reactions**
- **15 Features** including GPR (Gene-Protein-Reaction rules), Reversibility, Upper/Lower Bounds, and Subsystems.

## 💻 Tech Stack
- **Language:** Python 3.10+
- **Libraries:** Pandas, NumPy
- **Environment:** Spyder IDE / Terminal Integration

## 🛠️ How to Run
```bash
# Clone the repository
git clone [https://github.com/madamfallah/penicillium-metabolic-core.git](https://github.com/madamfallah/penicillium-metabolic-core.git)

# Run the analyzer module
python analyzer.py
