
# Vegetable_Cd_risk_System_Dynamics-SD-model


# System Dynamics Model of Vegetable Cadmium Pollution: Policy-Environment-Health Integrated Framework

## Overview

This repository contains a comprehensive **System Dynamics (SD) model** that integrates policy interventions, environmental factors, soil properties, bioaccumulation processes, dietary exposure, and population-specific health risks to simulate vegetable cadmium (Cd) pollution dynamics and project policy outcomes from 2025-2035.

The model framework visualizes the complex causal pathways from soil contamination through bioaccumulation to human health risks, incorporating:

- **Policy Layer**: Soil remediation, pH amendment, organic matter enhancement, planting structure adjustment, dietary guidance, and market regulation
- **Environmental System**: Climate zones, geographic regions, provinces, and seasonal variations
- **Soil Contamination & Properties**: Soil Cd content, pH, organic matter, and cation exchange capacity (CEC)
- **Bioaccumulation System**: Bioconcentration factors (BCF) for leafy, root, and fruit vegetables
- **Exposure System**: Urban/rural consumption patterns and body weight demographics
- **Health Risk Assessment**: Target Hazard Quotient (THQ) with integrated machine learning predictions
- **Socioeconomic Feedback**: Gender disparities and regional inequality analysis

---

## Key Features

### 1. Architecture & Visualization Scripts

- **Editable System Dynamics Diagram** (`Vegetable_Cd_SD_Model_Editable.py`)
  - High-resolution (16:9 aspect ratio, 36×20.25 inches) system dynamics framework
  - Professional color-coded layers (Policy → Environmental → Soil → Bioaccumulation → Exposure → Health)
  - Smooth Bézier curves with embedded pathway coefficients
  - PDF/SVG output with **editable text** (fonttype=42 for TrueType vectors)
  - Feedback loops visualization (Reinforcing Loop R1, Social Feedback)
  - Information boxes with model structure and pathway summaries

### 2. Policy Scenario Projection (2025-2035)

- **Realistic System Dynamics Model** (`p24_baseON_SD_predict_2025-2035_05.py`)
  - **Two scenarios**:
    - **BAU (Business-As-Usual)**: No policy intervention, natural decay only
    - **RP (Recommended Policy)**: Evidence-based policy package with realistic constraints
  - **Realistic Constraints Implemented**:
    - Residual baselines for irreducible contamination
    - Policy efficiency decay with a finite half-life
    - Diminishing returns on repeated interventions
    - Biological/physical lower bounds
    - Background exposure from non-vegetable sources
  - **Outputs**:
    - 9-panel comparison visualization (subplots b-j)
    - Cumulative improvement metrics
    - Population-specific risk stratification
    - Complete time-series data export

---

## Model Architecture

### System Dynamics Equations

#### Soil Contamination

```
dSoil_Cd/dt = (Natural decay) + (Atmospheric deposition) - (Policy remediation)
            = -α·removable_Cd + β·remediation_rate·efficiency(t) + 0.015
```

#### Bioconcentration Factor (BCF)

```
BCF(t) = BCF₀ · (1 + β_soil_cd·Δsoil_cd_norm)
              · (1 + pH_effect·ΔpH)
              · (1 + CEC_effect·ΔCEC/5)
              · (1 + SOM_effect·ΔSOM)
              · (1 - policy_reduction·efficiency(t))
```

- **Soil Cd pool → BCF**: β = +0.497 (sum of Soil Cd and SCC; they are identical in 85% of records with VIF > 300, so they cannot be interpreted separately)
- **pH marginal effect**: +1 unit pH → BCF decreases by 9.06% (β = -0.071, p = 0.512)
- **CEC marginal effect**: +5 cmol/kg CEC → BCF increases by 16.91% (β = +0.136, p = 0.160)
- **SOM marginal effect**: +1 g/kg SOM → BCF decreases by 2.04% (β = -0.382, p = 0.003; ≈ +1% SOM (i.e., 10 g/kg) decreases BCF by 20.38%)
- **Residual constraint**: BCF ≥ residual_BCF

#### Vegetable Cadmium Content

```
Veg_Cd(t) = Soil_Cd · BCF(t) · (1 - market_control·efficiency(t))
          + residual_veg_Cd
```

#### Health Risk Assessment (THQ)

```
EDI_veg = (Veg_Cd · Consumption) / (Body_Weight · 365)
THQ_veg = EDI_veg / RfD
THQ_total = THQ_veg + background_THQ
```

- The classic WHO THQ formula is adopted.
- Marginal effects are already reflected in upstream steps such as BCF calculation and are not reapplied in THQ calculation.
- Background THQ remains stable at 0.3014 (from non-vegetable sources such as rice, drinking water, and atmospheric deposition).
- RfD (Cd) = 0.001 mg/kg/day.

### Policy Efficiency Decay Function

```
efficiency(t) = min_efficiency + (1 - min_efficiency) · exp(-λ·t)
λ = ln(2) / half_life
half_life = 8 years
min_efficiency = 0.40
```

---

## Data & Methodology

### Data Source

- **CVCCD Database** (China Vegetable & Cadmium Contamination Dataset)
  - **Data file**: `Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值.xlsx`
  - **Geographic coverage**: Multiple provinces, climate zones, and regions across China
  - **Vegetable diversity**: Leafy, root, and fruit vegetables
  - **Soil types**: Multiple soil categories relevant to vegetable production
  - **Time period**: 2004-2021

### Model Integration

- Machine learning models integrated for prediction and feature analysis.
- Structural equation modeling used to validate major causal pathways.
- Detailed performance metrics and statistical coefficients are not included in this README.

---

## File Descriptions

### Visualization & Framework Scripts

| File | Description | Output Format |
|------|-------------|----------------|
| `Vegetable_Cd_SD_Model_Editable.py` | System dynamics architecture diagram with multiple layers, nodes, and feedback pathways | PNG (400 DPI), PDF, SVG (all with editable text) |
| `p24_baseON_SD_predict_2025-2035_05.py` | 2025-2035 policy scenario projections with realistic constraints | 9-panel figure + CSV summaries + parameter Excel |
| `Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值.xlsx` | CVCCD data file used for model input and validation | XLSX |

### Output Files Generated

```
├── Vegetable_Cd_SD_Model_Editable.png
├── Vegetable_Cd_SD_Model_Editable.pdf
├── Vegetable_Cd_SD_Model_Editable.svg
├── SD_Projection_2025_2035_CBSEM_v05_NewDB_NewSEM.png
├── SD_Projection_2025_2035_CBSEM_v05_NewDB_NewSEM.pdf
├── SD_Projection_2025_2035_CBSEM_Summary_v05_NewDB_NewSEM.csv
├── SD_Projection_2025_2035_CBSEM_Full_Data_v05_NewDB_NewSEM.csv
└── SD_Parameters_v05_NewDB_NewSEM.xlsx
```

---

## Installation & Usage

### Requirements

```bash
pip install numpy pandas matplotlib scipy seaborn scikit-learn openpyxl
```

### Quick Start

1. **Generate System Dynamics Diagram**:

```python
python Vegetable_Cd_SD_Model_Editable.py
```

Output: Editable PDF/SVG with system architecture and coefficient annotations.

2. **Run Policy Scenario Projections**:

```python
python p24_baseON_SD_predict_2025-2035_05.py
```

Output: 9-panel visualization, CSV data tables, and parameter Excel.

3. **Custom Analysis** (modify parameters):

```python
import importlib

module = importlib.import_module("p24_baseON_SD_predict_2025-2035_05")
model = module.VegetableCdSystemDynamicsProjection()

# Customize policy parameters
custom_policy = {
    'soil_remediation_rate': 0.05,
    'pH_amendment_rate': 0.15,
    'SOM_increase_rate': 2.0,
    'BCF_reduction_factor': 0.30,
    'consumption_reduction': 0.15,
    'veg_cd_market_control': 0.20
}

results_custom = model.run_projection(custom_policy, 'Custom Policy')
model.visualize_projection(results_BAU, results_custom)
```

---

## Interpretation Guide

### Subplot Descriptions (a-j)

- **(b) Vegetable Cd Projection**: Shows convergence toward residual baseline with RP policy
- **(c) Average THQ Trend**: Primary health outcome; safety threshold marked
- **(d) Soil Cd Content**: Foundation of food chain exposure
- **(e) Population-Specific THQ**: Gender × Urban-Rural stratification
- **(f) pH Trajectory**: Key soil property affecting BCF
- **(g) BCF Evolution**: Bioaccumulation trend across vegetable types
- **(h) Vegetable Cd Reduction**: Cumulative benefit from policies
- **(i) THQ Reduction**: Main health outcome improvement metric
- **(j) Population THQ Evolution**: Long-term risk distribution across demographic groups

### Risk Assessment Categories

```
No Risk:        THQ ≤ 0.5          → Safe for general population
Low Risk:       0.5 < THQ ≤ 1.0    → Acceptable; routine monitoring
Medium Risk:    1.0 < THQ ≤ 2.0    → Intervention needed; dietary counseling
High Risk:      THQ > 2.0          → Urgent action; medical evaluation recommended
```

---

## Key Parameters & Assumptions

| Parameter | Value | Source/Notes |
|-----------|-------|-------------|
| Soil Cd natural decay | 2%/year | Literature; weathering + leaching |
| Policy efficiency half-life | 8 years | Evidence-based diminishing returns |
| Minimum policy efficiency | 0.40 | Residual policy effectiveness floor |
| Atmospheric deposition | 0.015 mg/kg/year | Model constant |
| Residual soil Cd | 0.00685 mg/kg | Minimum in database |
| Residual veg Cd | 8.1e-05 mg/kg | Minimum in database |
| Background THQ | 0.3014 | Rice, water, air |
| RfD (Cd) | 0.001 mg/kg/day | US EPA reference dose |
| Target pH | 7.0 | Optimal for Cd immobilization |
| SOM effect | +1 g/kg → BCF decreases by 2.04% | β = -0.382, p = 0.003 |
| pH effect | +1 unit → BCF decreases by 9.06% | β = -0.071, p = 0.512 |
| CEC effect | +5 cmol/kg → BCF increases by 16.91% | β = +0.136, p = 0.160 |
| Soil Cd pool → BCF | β = +0.497 | Sum of Soil Cd and SCC; collinear |
| BCF → Veg Cd | β = 0.379 | p < 0.001 |
| Veg Cd → THQ | β = 1.001 | p < 0.001 |
| Consumption → THQ | β = 0.0065 | p = 0.044 |

### Policy Scenario Parameters

| Parameter | BAU | RP |
|-----------|-----|-----|
| soil_remediation_rate | 0.00 | 0.03 |
| pH_amendment_rate | 0.00 | 0.10 |
| SOM_increase_rate | 0.00 | 1.2 |
| BCF_reduction_factor | 0.00 | 0.18 |
| consumption_reduction | 0.00 | 0.10 |
| veg_cd_market_control | 0.00 | 0.12 |

---

## Limitations & Future Work

### Current Limitations

1. **Aggregated scale**: Provincial-level analysis; sub-regional variation not captured
2. **Linear policy assumptions**: Actual implementation heterogeneity not modeled
3. **Climate change**: Fixed climate zone assumptions; future climate shift not projected
4. **Dietary shift lag**: Consumer behavior change modeled as instantaneous
5. **Economic constraints**: Policy cost-effectiveness not integrated

### Future Extensions

- [ ] Sub-grid spatial heterogeneity (county-level)
- [ ] Stochastic uncertainty analysis (Monte Carlo simulation)
- [ ] Climate change scenarios (RCP 4.5/8.5)
- [ ] Agent-based consumer behavior model
- [ ] Cost-benefit analysis framework
- [ ] Sensitivity analysis dashboard (interactive Streamlit app)

---

## References & Citations

### Core Publications

1. **System Framework & Empirical Data**
   - CVCCD Database: observations from China's vegetable production regions (2004-2021)
   - CB-SEM validation: pathway coefficients from structural equation modeling

2. **Health Risk Assessment**
   - US EPA Target Hazard Quotient (THQ) methodology
   - RfD for Cd: 0.001 mg/kg/day
   - Integrated machine learning models for prediction and feature analysis

3. **Policy Scenarios**
   - BAU: Natural decay assumptions based on Chinese agricultural soil studies
   - RP: Recommended policies align with China's Cadmium Pollution Control Action Plan

---

## Developed at

**Hunan Provincial University Key Laboratory for Environmental and Ecological Health**

**Hunan Provincial University Key Laboratory for Environmental Behavior and Control Principle of New Pollutants**

**College of Environment and Resources, Xiangtan University**  
**Xiangtan 411105, China**

---

## License

This project is released under the **MIT License** for academic and research use.

---

