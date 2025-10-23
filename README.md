# Vegetable_Cd_risk_System_Dynamics-SD-model
Vegetable Cd risk System Dynamics (SD) model


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

### 1. **Architecture & Visualization Scripts**
- **Editable System Dynamics Diagram** (`Vegetable_Cd_SD_Model_Editable.py`)
  - High-resolution (16:9 aspect ratio, 36×20.25 inches) system dynamics framework
  - Professional color-coded layers (Policy → Environmental → Soil → Bioaccumulation → Exposure → Health)
  - Smooth Bézier curves with embedded pathway coefficients
  - PDF/SVG output with **editable text** (fonttype=42 for TrueType vectors)
  - Feedback loops visualization (Reinforcing Loop R1, Social Feedback)
  - Information boxes with statistical summaries (CB-SEM paths, SHAP importance, marginal effects, flow contributions)

### 2. **Policy Scenario Projection (2025-2035)**
- **Realistic System Dynamics Model** (`SD_Projection_2025_2035_Realistic.py`)
  - **Two scenarios**:
    - **BAU (Business-As-Usual)**: No policy intervention, natural decay only
    - **RP (Recommended Policy)**: Evidence-based policy package with realistic constraints
  - **Realistic Constraints Implemented**:
    - Residual baselines (irreducible contamination: soil Cd 0.80 mg/kg, veg Cd 0.030 mg/kg)
    - Policy efficiency decay with 8-year half-life
    - Diminishing returns on repeated interventions
    - Biological/physical lower bounds
    - Background exposure from non-vegetable sources (THQ 0.25)
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
            = -α·removable_Cd + β·remediation_rate·efficiency(t) + 0.02
```

#### Bioconcentration Factor (BCF)
```
BCF(t) = BCF₀ · exp(β_pH·ΔpH_norm) · exp(β_SOM·ΔSOM_norm) · (1 - policy_reduction·efficiency(t))
```
- **pH effect**: ΔpH +1 → BCF -21% to -52% (β = -0.346***)
- **SOM effect**: Δ+1% SOM → BCF -0.78% (r = -0.21*)
- **Residual constraint**: BCF ≥ residual_BCF

#### Vegetable Cadmium Content
```
Veg_Cd(t) = Soil_Cd · BCF(t) · (1 - market_control·efficiency(t))
          + residual_veg_Cd
```

#### Health Risk Assessment (THQ)
```
THQ(t) = (Veg_Cd · Consumption) / (Body_Weight · 365 · RfD)
       + Δ(Veg_Cd)·marginal_effect
       + background_THQ·exp(-decay_rate·t)
```
- **Marginal effect**: Δ0.1 mg/kg Veg Cd → THQ ↑ 0.91-0.93
- **Consumption elasticity**: +10 kg/year → THQ ↑ 0.27-0.64
- **Background THQ**: 0.25 from rice, water, air

### Policy Efficiency Decay Function
```
efficiency(t) = min_efficiency + (1 - min_efficiency) · exp(-λ·t)
λ = ln(2) / half_life = ln(2) / 8 years
```

---

## Data & Methodology

### Data Source
- **CVCCD Database** (China Vegetable & Cadmium Contamination Dataset)
  - **Sample size**: n = 2,674 observations
  - **Geographic coverage**: 30 provinces, 8 climate zones, 5 regions
  - **Vegetable diversity**: 115 vegetable types (51.87% leafy, 24.61% root, 23.52% fruit)
  - **Soil types**: 14 categories (Red Soil: 21.2%, Loess soil: BCF up to 1.8×)
  - **Time period**: 2004-2021

### Model Performance
- **Machine Learning Integration**: CNN + XGBoost + LightGBM + Random Forest + SVM + GBDT
- **AUC**: 0.99 | **Accuracy**: 0.97
- **SHAP Feature Importance**: Veg Cd (1.31-4.40), Consumption (0.09-1.38), Soil Cd (0.50-1.20), BCF (0.30-0.90)

### Statistical Validation (CB-SEM)
| Pathway | Coefficient | Significance |
|---------|------------|--------------|
| Climate → Soil Cd | 0.714 | *** |
| pH → BCF | -0.346 | *** |
| Soil Cd → BCF | 0.595 | *** |
| BCF → Veg Cd | 0.247 | *** |
| Veg Cd → THQ | 0.999 | *** |
| Consumption → THQ | 0.016 | *** |

*** p < 0.001, ** p < 0.01

---

## Key Findings

### 2025-2035 Projection Results

#### Scenario Comparison (10-Year Cumulative Effect)

| Indicator | BAU 2025 | BAU 2035 | RP 2025 | RP 2035 | Policy Benefit |
|-----------|----------|----------|---------|---------|----------------|
| Soil Cd (mg/kg) | 2.04 | 2.02 | 2.04 | 1.38 | **-32.4%** |
| Veg Cd (mg/kg) | 0.160 | 0.158 | 0.160 | 0.085 | **-46.2%** |
| Average THQ | 1.713 | 1.701 | 1.713 | 1.182 | **-30.5%** |
| Urban Male THQ | 1.587 | 1.577 | 1.587 | 1.051 | **-33.4%** |
| Urban Female THQ | 1.854 | 1.843 | 1.854 | 1.287 | **-30.2%** |
| Rural Male THQ | 1.615 | 1.604 | 1.615 | 1.112 | **-31.2%** |
| Rural Female THQ | 1.835 | 1.824 | 1.835 | 1.268 | **-30.8%** |

### Health Risk Stratification (RP Scenario by 2035)

| Risk Category | THQ Range | 2035 Population% | Risk Level |
|---------------|-----------|-----------------|-----------|
| No Risk | 0 < THQ ≤ 0.5 | 5-8% | Safe |
| Low Risk | 0.5 < THQ ≤ 1.0 | 35-42% | Acceptable |
| Medium Risk | 1 < THQ ≤ 2.0 | 45-55% | Intervention needed |
| High Risk | THQ > 2.0 | 2-5% | Urgent action required |

### Policy Efficiency & Residual Constraints

1. **Soil Cd reduction** approaches residual baseline (0.80 mg/kg) after 10 years of intensive remediation
2. **Policy effectiveness decay**: 8-year half-life; by 2035, effectiveness reduced to 40-50% of initial impact
3. **Gender disparity**: Female THQ 1.8% higher than males due to lower body weight
4. **Regional inequality**: Central China THQ up 41.3% vs Eastern regions

---

## File Descriptions

### Visualization & Framework Scripts

| File | Description | Output Format |
|------|-------------|----------------|
| `Vegetable_Cd_SD_Model_Editable.py` | System dynamics architecture diagram with 8 layers, 30+ nodes, and 15+ feedback pathways | PNG (400 DPI), PDF, SVG (all with editable text) |
| `SD_Projection_2025_2035_Realistic.py` | 2025-2035 policy scenario projections with realistic constraints | 9-panel figure + CSV summaries |

### Output Files Generated

```
├── Vegetable_Cd_SD_Model_Editable.png          # High-resolution diagram (400 DPI)
├── Vegetable_Cd_SD_Model_Editable.pdf          # Editable vector format
├── Vegetable_Cd_SD_Model_Editable.svg          # Scalable vector graphics
├── SD_Projection_2025_2035_Realistic.png       # 9-panel comparison (400 DPI)
├── SD_Projection_2025_2035_Realistic.pdf       # Publication-ready PDF
├── SD_Projection_2025_2035_Realistic_Summary.csv        # Aggregated metrics
└── SD_Projection_2025_2035_Realistic_Full_Data.csv     # Complete time-series (2025-2035, 0.1-year intervals)
```

---

## Installation & Usage

### Requirements
```bash
pip install numpy pandas matplotlib scipy seaborn scikit-learn
```

### Quick Start

1. **Generate System Dynamics Diagram**:
```python
python Vegetable_Cd_SD_Model_Editable.py
```
Output: Editable PDF/SVG with system architecture and coefficient annotations

2. **Run Policy Scenario Projections**:
```python
python SD_Projection_2025_2035_Realistic.py
```
Output: 9-panel visualization + CSV data tables

3. **Custom Analysis** (modify parameters):
```python
from SD_Projection_2025_2035_Realistic import VegetableCdSystemDynamicsProjection

model = VegetableCdSystemDynamicsProjection()

# Customize policy parameters
custom_policy = {
    'soil_remediation_rate': 0.05,      # Increase soil remediation to 5%/year
    'pH_amendment_rate': 0.15,          # Stronger pH intervention
    'SOM_increase_rate': 2.0,           # More organic matter addition
    'BCF_reduction_factor': 0.30,       # Crop selection focus
    'consumption_reduction': 0.15,      # Dietary shift campaign
    'veg_cd_market_control': 0.20       # Stricter market standards
}

results_custom = model.run_projection(custom_policy, 'Custom Policy')
model.visualize_projection(results_BAU, results_custom)
```

---

## Interpretation Guide

### Subplot Descriptions (a-j)

- **(b) Vegetable Cd Projection**: Shows convergence toward residual baseline with RP policy
- **(c) Average THQ Trend**: Primary health outcome; safety threshold (THQ=1) marked
- **(d) Soil Cd Content**: Foundation of food chain exposure
- **(e) Population-Specific THQ**: Gender × Urban-Rural stratification
- **(f) pH Trajectory**: Key soil property affecting BCF (inverse relationship)
- **(g) BCF Evolution**: Bioaccumulation trend across vegetable types
- **(h) Vegetable Cd Reduction**: Cumulative benefit from policies (%)
- **(i) THQ Reduction**: Main health outcome improvement metric
- **(j) Population THQ Evolution**: Long-term risk distribution across demographic groups

### Risk Assessment Categories

```
No Risk:        THQ ≤ 0.5    → Safe for general population
Low Risk:       0.5 < THQ ≤ 1.0   → Acceptable; routine monitoring
Medium Risk:    1.0 < THQ ≤ 2.0   → Intervention needed; dietary counseling
High Risk:      THQ > 2.0    → Urgent action; medical evaluation recommended
```

---

## Key Parameters & Assumptions

| Parameter | Value | Source/Notes |
|-----------|-------|-------------|
| Soil Cd natural decay | 2%/year | Literature; weathering + leaching |
| Policy efficiency half-life | 8 years | Evidence-based diminishing returns |
| Residual soil Cd | 0.80 mg/kg | Atmospheric deposition + parent material |
| Residual veg Cd | 0.030 mg/kg | Detection limit + genetic background |
| Background THQ | 0.25 | Rice (main staple), water, air |
| RfD (Cd) | 0.001 mg/kg/day | US EPA reference dose |
| Target pH | 7.0 | Optimal for Cd immobilization |

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
   - CVCCD Database: 2,674 observations from China's vegetable production regions (2004-2021)
   - CB-SEM validation: Pathway coefficients from structural equation modeling (p < 0.001 for major paths)

2. **Health Risk Assessment**
   - US EPA Target Hazard Quotient (THQ) methodology
   - RfD for Cd: 0.001 mg/kg/day (reference dose)
   - Integrated ML model: AUC 0.99, Accuracy 97%

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

## Contact & Support

For questions, collaborations, or data access requests:
- **Research Team**: College of Environment and Resources, Xiangtan University
- **Project Lead**: [Your Name/Contact]
- **GitHub Issues**: Submit technical questions and feature requests

---

## Acknowledgments

This research was supported by funding from the [Relevant Research Grant/Program]. We thank the CVCCD database custodians and all field survey participants for data collection efforts.

---

## Citation

If you use this model or code in your research, please cite:

```bibtex
@software{vegetable_cd_sd_2025,
  title={System Dynamics Model of Vegetable Cadmium Pollution: 
         Policy-Environment-Health Integrated Framework},
  author={[Research Team]},
  year={2025},
  institution={Xiangtan University},
  url={https://github.com/[your-repo]}
}
```

---

**Last Updated**: 2025-10-23  
**Model Version**: 2.0 (Realistic Constraints Implementation)  
**Status**: ✓ Production Ready
```
