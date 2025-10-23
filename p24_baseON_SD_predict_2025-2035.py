import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False

class VegetableCdSystemDynamicsProjection:
    """
    蔬菜镉污染系统动力学模型 - 2025-2035年政策情景预测（现实约束版本）
    Realistic System Dynamics Model with Physical/Biological Constraints
    """
    
    def __init__(self):
        """
        初始化模型参数（基于CVCCD数据库2004-2021年统计）
        """
        # ========== 时间参数 ==========
        self.start_year = 2025
        self.end_year = 2035
        self.time_horizon = self.end_year - self.start_year
        self.dt = 0.1
        self.time_steps = np.arange(0, self.time_horizon + self.dt, self.dt)
        
        # ========== 基准参数（2025年初始状态）==========
        self.initial_soil_cd = 2.0427
        self.initial_pH = 6.5955
        self.initial_SOM = 27.7281
        self.initial_CEC = 15.3218
        
        self.initial_veg_cd = 0.1604
        self.initial_BCF_leafy = 0.123
        self.initial_BCF_root = 0.096
        self.initial_BCF_fruit = 0.054
        
        self.urban_male_weight = 66.3962
        self.urban_female_weight = 57.1412
        self.rural_male_weight = 63.0256
        self.rural_female_weight = 55.3530
        
        self.urban_consumption = 106.6222
        self.rural_consumption = 98.2451
        
        self.initial_urban_male_THQ = 1.5868
        self.initial_urban_female_THQ = 1.8542
        self.initial_rural_male_THQ = 1.6148
        self.initial_rural_female_THQ = 1.8345
        
        # ========== CB-SEM路径系数 ==========
        self.beta_soil_cd_to_BCF = 0.595
        self.beta_pH_to_BCF = -0.346
        self.beta_SOM_to_BCF = -0.21
        self.beta_CEC_to_BCF = 0.222
        self.beta_BCF_to_veg_cd = 0.247
        self.beta_veg_cd_to_THQ = 0.999
        self.beta_consumption_to_THQ = 0.016
        
        # ========== 边际效应参数 ==========
        self.marginal_veg_cd_per_01mg = 0.92
        self.marginal_consumption_per_10kg = 0.455
        
        # ========== 自然衰减与演化速率 ==========
        self.soil_cd_natural_decay = 0.02
        self.pH_natural_buffering = 0.05
        self.SOM_decomposition = 0.03
        self.target_pH = 7.0
        
        # ========== 蔬菜类型分布 ==========
        self.leafy_proportion = 0.5187
        self.root_proportion = 0.2461
        self.fruit_proportion = 0.2352
        
        # ========== 残留污染基线 ==========
        self.residual_soil_cd = 0.80
        self.residual_BCF_leafy = 0.035
        self.residual_BCF_root = 0.028
        self.residual_BCF_fruit = 0.018
        self.residual_veg_cd = 0.030
        
        # ========== 背景暴露叠加 ==========
        self.background_THQ = 0.25
        
        # ========== 政策效率衰减函数参数 ==========
        self.policy_efficiency_halflife = 8
        self.min_policy_efficiency = 0.40
        
        # ========== 政策情景参数 ==========
        self.policy_BAU = {
            'soil_remediation_rate': 0.00,
            'pH_amendment_rate': 0.00,
            'SOM_increase_rate': 0.00,
            'BCF_reduction_factor': 0.00,
            'consumption_reduction': 0.00,
            'veg_cd_market_control': 0.00
        }
        
        self.policy_RP = {
            'soil_remediation_rate': 0.035,
            'pH_amendment_rate': 0.12,
            'SOM_increase_rate': 1.5,
            'BCF_reduction_factor': 0.20,
            'consumption_reduction': 0.10,
            'veg_cd_market_control': 0.15
        }
        
        print("="*80)
        print("蔬菜镉污染系统动力学模型 - 2025-2035年政策情景预测（现实约束版本）")
        print("Realistic SD Model with Residual Baselines & Diminishing Returns")
        print("="*80)
        print(f"\n时间范围: {self.start_year}-{self.end_year} ({self.time_horizon}年)")
        print(f"初始状态 (2025基线):")
        print(f"  - 土壤Cd: {self.initial_soil_cd:.4f} mg/kg (残留下限: {self.residual_soil_cd:.2f})")
        print(f"  - 蔬菜Cd: {self.initial_veg_cd:.4f} mg/kg (残留下限: {self.residual_veg_cd:.3f})")
        print(f"  - 平均THQ: {np.mean([self.initial_urban_male_THQ, self.initial_urban_female_THQ, self.initial_rural_male_THQ, self.initial_rural_female_THQ]):.4f}")
        print(f"  - 背景THQ: {self.background_THQ:.2f} (来自非蔬菜源)")
        print("\n政策情景设置:")
        print("  [1] BAU情景: 无政策干预（维持现状）")
        print("  [2] RP情景: 现实约束下的建议政策（考虑边际递减）")
        print("="*80 + "\n")
    
    
    def policy_efficiency_decay(self, t):
        """
        政策效率随时间衰减（边际递减规律）
        """
        decay_rate = np.log(2) / self.policy_efficiency_halflife
        efficiency = self.min_policy_efficiency + \
                    (1.0 - self.min_policy_efficiency) * np.exp(-decay_rate * t)
        return efficiency
    
    
    def calculate_BCF_weighted(self, soil_cd, pH, SOM, CEC, policy_BCF_reduction=0, t=0):
        """
        计算加权平均BCF（考虑3类蔬菜比例 + 生物学下限约束）
        """
        pH_norm = (pH - self.initial_pH) / 1.22
        SOM_norm = (SOM - self.initial_SOM) / self.initial_SOM
        CEC_norm = (CEC - self.initial_CEC) / self.initial_CEC
        soil_cd_norm = (soil_cd - self.initial_soil_cd) / 6.23
        
        policy_efficiency = self.policy_efficiency_decay(t)
        effective_BCF_reduction = policy_BCF_reduction * policy_efficiency
        
        def calc_single_BCF(base_BCF, residual_BCF):
            BCF = base_BCF * np.exp(self.beta_pH_to_BCF * pH_norm)
            BCF *= np.exp(self.beta_SOM_to_BCF * SOM_norm)
            BCF *= np.exp(self.beta_CEC_to_BCF * CEC_norm * 0.5)
            BCF *= (1 + self.beta_soil_cd_to_BCF * soil_cd_norm * 0.1)
            BCF *= (1 - effective_BCF_reduction)
            BCF = max(residual_BCF, BCF)
            return BCF
        
        BCF_leafy = calc_single_BCF(self.initial_BCF_leafy, self.residual_BCF_leafy)
        BCF_root = calc_single_BCF(self.initial_BCF_root, self.residual_BCF_root)
        BCF_fruit = calc_single_BCF(self.initial_BCF_fruit, self.residual_BCF_fruit)
        
        BCF_avg = (BCF_leafy * self.leafy_proportion + 
                   BCF_root * self.root_proportion + 
                   BCF_fruit * self.fruit_proportion)
        
        return max(0.001, BCF_avg)
    
    
    def calculate_veg_cd(self, soil_cd, BCF, policy_market_control=0, t=0):
        """
        计算蔬菜Cd含量（含残留下限约束）
        """
        policy_efficiency = self.policy_efficiency_decay(t)
        effective_market_control = policy_market_control * policy_efficiency
        
        veg_cd = soil_cd * BCF * (1 + self.beta_BCF_to_veg_cd * 0.2)
        veg_cd *= (1 - effective_market_control)
        veg_cd = max(self.residual_veg_cd, veg_cd)
        
        return veg_cd
    
    
    def calculate_THQ(self, veg_cd, consumption, body_weight, t=0):
        """
        计算THQ（含背景暴露叠加）
        """
        EDI_veg = (veg_cd * consumption) / (body_weight * 365)
        RfD = 0.001
        THQ_veg = EDI_veg / RfD
        
        veg_cd_deviation = (veg_cd - self.initial_veg_cd) / 0.1
        THQ_veg += veg_cd_deviation * self.marginal_veg_cd_per_01mg * 0.1
        
        background_decay = 0.01
        background_THQ_current = self.background_THQ * np.exp(-background_decay * t)
        
        THQ_total = max(0, THQ_veg) + background_THQ_current
        
        return THQ_total
    
    
    def system_dynamics_equations(self, state, t, policy_params):
        """
        系统动力学微分方程组（含现实约束）
        """
        soil_cd, pH, SOM, CEC, BCF_avg, veg_cd, \
        UM_THQ, UF_THQ, RM_THQ, RF_THQ = state
        
        soil_remed = policy_params['soil_remediation_rate']
        pH_amend = policy_params['pH_amendment_rate']
        SOM_incr = policy_params['SOM_increase_rate']
        BCF_reduc = policy_params['BCF_reduction_factor']
        consump_reduc = policy_params['consumption_reduction']
        market_ctrl = policy_params['veg_cd_market_control']
        
        efficiency = self.policy_efficiency_decay(t)
        
        # (1) 土壤Cd
        removable_soil_cd = max(0, soil_cd - self.residual_soil_cd)
        natural_decay = -removable_soil_cd * self.soil_cd_natural_decay
        policy_remediation = -removable_soil_cd * soil_remed * efficiency
        atmospheric_deposition = 0.02
        dSoil_Cd_dt = natural_decay + policy_remediation + atmospheric_deposition
        
        # (2) pH
        pH_natural_change = (self.target_pH - pH) * self.pH_natural_buffering
        pH_policy_change = pH_amend * efficiency
        
        dpH_dt = pH_natural_change + pH_policy_change
        
        if pH > 8.5:
            dpH_dt = min(0, dpH_dt)
        elif pH > 8.0:
            dpH_dt *= 0.3
        
        # (3) SOM
        dSOM_dt = -SOM * self.SOM_decomposition + SOM_incr * efficiency
        
        # (4) CEC
        CEC_target = 15 + (SOM - self.initial_SOM) * 0.2
        dCEC_dt = (CEC_target - CEC) * 0.1
        
        # (5) BCF
        new_BCF = self.calculate_BCF_weighted(soil_cd, pH, SOM, CEC, BCF_reduc, t)
        dBCF_dt = (new_BCF - BCF_avg) * 0.5
        
        # (6) 蔬菜Cd
        new_veg_cd = self.calculate_veg_cd(soil_cd, BCF_avg, market_ctrl, t)
        dVeg_Cd_dt = (new_veg_cd - veg_cd) * 0.8
        
        # (7-10) 各人群THQ
        UM_consump = self.urban_consumption * (1 - consump_reduc * efficiency)
        UF_consump = self.urban_consumption * (1 - consump_reduc * efficiency)
        RM_consump = self.rural_consumption * (1 - consump_reduc * efficiency)
        RF_consump = self.rural_consumption * (1 - consump_reduc * efficiency)
        
        new_UM_THQ = self.calculate_THQ(veg_cd, UM_consump, self.urban_male_weight, t)
        new_UF_THQ = self.calculate_THQ(veg_cd, UF_consump, self.urban_female_weight, t)
        new_RM_THQ = self.calculate_THQ(veg_cd, RM_consump, self.rural_male_weight, t)
        new_RF_THQ = self.calculate_THQ(veg_cd, RF_consump, self.rural_female_weight, t)
        
        dUM_THQ_dt = (new_UM_THQ - UM_THQ) * 0.8
        dUF_THQ_dt = (new_UF_THQ - UF_THQ) * 0.8
        dRM_THQ_dt = (new_RM_THQ - RM_THQ) * 0.8
        dRF_THQ_dt = (new_RF_THQ - RF_THQ) * 0.8
        
        return [dSoil_Cd_dt, dpH_dt, dSOM_dt, dCEC_dt, dBCF_dt, dVeg_Cd_dt,
                dUM_THQ_dt, dUF_THQ_dt, dRM_THQ_dt, dRF_THQ_dt]
    
    
    def run_projection(self, policy_params, scenario_name):
        """
        运行单个情景预测
        """
        initial_BCF = (self.initial_BCF_leafy * self.leafy_proportion +
                       self.initial_BCF_root * self.root_proportion +
                       self.initial_BCF_fruit * self.fruit_proportion)
        
        initial_state = [
            self.initial_soil_cd,
            self.initial_pH,
            self.initial_SOM,
            self.initial_CEC,
            initial_BCF,
            self.initial_veg_cd,
            self.initial_urban_male_THQ,
            self.initial_urban_female_THQ,
            self.initial_rural_male_THQ,
            self.initial_rural_female_THQ
        ]
        
        solution = odeint(
            self.system_dynamics_equations,
            initial_state,
            self.time_steps,
            args=(policy_params,)
        )
        
        years = self.start_year + self.time_steps
        
        results = pd.DataFrame({
            'Year': years,
            'Soil_Cd': solution[:, 0],
            'pH': solution[:, 1],
            'SOM': solution[:, 2],
            'CEC': solution[:, 3],
            'BCF': solution[:, 4],
            'Veg_Cd': solution[:, 5],
            'Urban_Male_THQ': solution[:, 6],
            'Urban_Female_THQ': solution[:, 7],
            'Rural_Male_THQ': solution[:, 8],
            'Rural_Female_THQ': solution[:, 9]
        })
        
        results['Average_THQ'] = results[[
            'Urban_Male_THQ', 'Urban_Female_THQ',
            'Rural_Male_THQ', 'Rural_Female_THQ'
        ]].mean(axis=1)
        
        def classify_risk(thq):
            if thq <= 0.5:
                return 'No Risk'
            elif thq <= 1.0:
                return 'Low Risk'
            elif thq <= 2.0:
                return 'Medium Risk'
            else:
                return 'High Risk'
        
        results['Risk_Level'] = results['Average_THQ'].apply(classify_risk)
        results['Scenario'] = scenario_name
        
        print(f"\n✓ {scenario_name} 情景模拟完成")
        print(f"  2025年: 蔬菜Cd={results['Veg_Cd'].iloc[0]:.4f} mg/kg, 平均THQ={results['Average_THQ'].iloc[0]:.4f}")
        print(f"  2035年: 蔬菜Cd={results['Veg_Cd'].iloc[-1]:.4f} mg/kg, 平均THQ={results['Average_THQ'].iloc[-1]:.4f}")
        print(f"  变化率: 蔬菜Cd {(results['Veg_Cd'].iloc[-1]/results['Veg_Cd'].iloc[0]-1)*100:.2f}%, THQ {(results['Average_THQ'].iloc[-1]/results['Average_THQ'].iloc[0]-1)*100:.2f}%")
        
        if 'RP' in scenario_name:
            soil_approach = (results['Soil_Cd'].iloc[-1] - self.residual_soil_cd) / \
                           (self.initial_soil_cd - self.residual_soil_cd) * 100
            veg_approach = (results['Veg_Cd'].iloc[-1] - self.residual_veg_cd) / \
                          (self.initial_veg_cd - self.residual_veg_cd) * 100
            
            print(f"  残留基线接近度:")
            print(f"    - 土壤Cd: 距基线{self.residual_soil_cd:.2f} mg/kg还有{soil_approach:.1f}%")
            print(f"    - 蔬菜Cd: 距基线{self.residual_veg_cd:.3f} mg/kg还有{veg_approach:.1f}%")
        
        return results
    
    
    def run_all_scenarios(self):
        """
        运行所有情景并对比
        """
        print("\n" + "="*80)
        print("开始运行2025-2035年政策情景预测（现实约束版本）...")
        print("="*80)
        
        results_BAU = self.run_projection(self.policy_BAU, 'BAU (No Policy)')
        results_RP = self.run_projection(self.policy_RP, 'RP (Recommended Policy)')
        
        results_combined = pd.concat([results_BAU, results_RP], ignore_index=True)
        
        return results_BAU, results_RP, results_combined
    
    
    def visualize_projection(self, results_BAU, results_RP, save_path=None):
        """
        可视化预测结果对比（完整9子图版本，编号b-j）
        """
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        color_BAU = '#E74C3C'
        color_RP = '#27AE60'
        
        # ============================================================================
        # (b) 蔬菜Cd含量趋势
        # ============================================================================
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results_BAU['Year'], results_BAU['Veg_Cd'], 
                color=color_BAU, linewidth=3, label='BAU (No Policy)', 
                marker='o', markersize=3, markevery=20)
        ax1.plot(results_RP['Year'], results_RP['Veg_Cd'], 
                color=color_RP, linewidth=3, label='RP (Recommended)', 
                marker='s', markersize=3, markevery=20)
        ax1.axhline(y=0.2, color='orange', linestyle='--', linewidth=2, 
                alpha=0.7, label='National Limit (0.2 mg/kg)')
        ax1.axhline(y=self.residual_veg_cd, color='purple', linestyle=':', 
                linewidth=2.5, alpha=0.6, 
                label=f'Residual Baseline ({self.residual_veg_cd:.3f} mg/kg)')
        ax1.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax1.set_ylabel('Vegetable Cd Content (mg/kg)', fontsize=13, weight='bold', family='serif')
        ax1.set_title('(b) Vegetable Cadmium Content Projection', fontsize=14, weight='bold', family='serif')
        ax1.legend(fontsize=8.5, loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax1.grid(alpha=0.3)
        ax1.set_xlim(2025, 2035)
        
        # ============================================================================
        # (c) 平均THQ趋势
        # ============================================================================
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results_BAU['Year'], results_BAU['Average_THQ'], 
                color=color_BAU, linewidth=3.5, label='BAU', 
                marker='o', markersize=3, markevery=20)
        ax2.plot(results_RP['Year'], results_RP['Average_THQ'], 
                color=color_RP, linewidth=3.5, label='RP', 
                marker='s', markersize=3, markevery=20)
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Safety Threshold (THQ=1)')
        ax2.axhline(y=self.background_THQ, color='purple', linestyle=':', 
                linewidth=2.5, alpha=0.6, 
                label=f'Background THQ ({self.background_THQ:.2f})')
        ax2.fill_between(results_BAU['Year'], 0, 1, color='green', alpha=0.1, 
                        label='No Risk Zone')
        ax2.fill_between(results_BAU['Year'], 1, 2, color='orange', alpha=0.1, 
                        label='Low-Medium Risk')
        ax2.fill_between(results_BAU['Year'], 2, ax2.get_ylim()[1], 
                        color='red', alpha=0.1, label='High Risk')
        ax2.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax2.set_ylabel('Average THQ', fontsize=13, weight='bold', family='serif')
        ax2.set_title('(c) Average Health Risk (THQ) Projection', fontsize=14, weight='bold', family='serif')
        ax2.legend(fontsize=8, loc='best', ncol=2, framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax2.grid(alpha=0.3)
        ax2.set_xlim(2025, 2035)
        
        # ============================================================================
        # (d) 土壤Cd含量趋势
        # ============================================================================
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(results_BAU['Year'], results_BAU['Soil_Cd'], 
                color=color_BAU, linewidth=3, label='BAU', 
                marker='o', markersize=3, markevery=20)
        ax3.plot(results_RP['Year'], results_RP['Soil_Cd'], 
                color=color_RP, linewidth=3, label='RP', 
                marker='s', markersize=3, markevery=20)
        ax3.axhline(y=self.residual_soil_cd, color='purple', linestyle=':', 
                linewidth=2.5, alpha=0.6, 
                label=f'Residual Baseline ({self.residual_soil_cd:.2f} mg/kg)')
        ax3.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax3.set_ylabel('Soil Cd Content (mg/kg)', fontsize=13, weight='bold', family='serif')
        ax3.set_title('(d) Soil Cadmium Content Projection', fontsize=14, weight='bold', family='serif')
        ax3.legend(fontsize=9, loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax3.grid(alpha=0.3)
        ax3.set_xlim(2025, 2035)
        
        # ============================================================================
        # (e) 各人群THQ对比 - 2025 vs 2035
        # ============================================================================
        ax4 = fig.add_subplot(gs[1, 0])
        populations = ['Urban\nMale', 'Urban\nFemale', 'Rural\nMale', 'Rural\nFemale']
        pop_cols = ['Urban_Male_THQ', 'Urban_Female_THQ', 'Rural_Male_THQ', 'Rural_Female_THQ']
        
        x = np.arange(len(populations))
        width = 0.2
        
        BAU_2025 = [results_BAU[col].iloc[0] for col in pop_cols]
        RP_2025 = [results_RP[col].iloc[0] for col in pop_cols]
        BAU_2035 = [results_BAU[col].iloc[-1] for col in pop_cols]
        RP_2035 = [results_RP[col].iloc[-1] for col in pop_cols]
        
        ax4.bar(x - 1.5*width, BAU_2025, width, label='BAU 2025', 
            color=color_BAU, alpha=0.5, edgecolor='black')
        ax4.bar(x - 0.5*width, BAU_2035, width, label='BAU 2035', 
            color=color_BAU, alpha=1.0, edgecolor='black')
        ax4.bar(x + 0.5*width, RP_2025, width, label='RP 2025', 
            color=color_RP, alpha=0.5, edgecolor='black')
        ax4.bar(x + 1.5*width, RP_2035, width, label='RP 2035', 
            color=color_RP, alpha=1.0, edgecolor='black')
        
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.axhline(y=self.background_THQ, color='purple', linestyle=':', 
                linewidth=2, alpha=0.5)
        ax4.set_xlabel('Population Group', fontsize=13, weight='bold', family='serif')
        ax4.set_ylabel('THQ', fontsize=13, weight='bold', family='serif')
        ax4.set_title('(e) THQ by Population: 2025 vs 2035', fontsize=14, weight='bold', family='serif')
        ax4.set_xticks(x)
        ax4.set_xticklabels(populations, fontsize=11, family='serif', weight='bold')
        ax4.legend(fontsize=8.5, ncol=2, loc='upper left', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax4.grid(alpha=0.3, axis='y')
        
        # ============================================================================
        # (f) pH趋势 - 上限改为8
        # ============================================================================
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results_BAU['Year'], results_BAU['pH'], 
                color=color_BAU, linewidth=3, label='BAU', 
                marker='o', markersize=3, markevery=20)
        ax5.plot(results_RP['Year'], results_RP['pH'], 
                color=color_RP, linewidth=3, label='RP', 
                marker='s', markersize=3, markevery=20)
        ax5.axhline(y=7.0, color='blue', linestyle='--', linewidth=2, 
                alpha=0.5, label='Target pH (7.0)')
        ax5.axhline(y=9.0, color='orange', linestyle=':', linewidth=2, 
                alpha=0.5, label='Upper Limit (9.0)')
        ax5.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax5.set_ylabel('Soil pH', fontsize=13, weight='bold', family='serif')
        ax5.set_title('(f) Soil pH Projection', fontsize=14, weight='bold', family='serif')
        ax5.legend(fontsize=9, loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax5.grid(alpha=0.3)
        ax5.set_xlim(2025, 2035)
        ax5.set_ylim(6.5, 8.0)
        
        # ============================================================================
        # (g) BCF趋势
        # ============================================================================
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(results_BAU['Year'], results_BAU['BCF'], 
                color=color_BAU, linewidth=3, label='BAU', 
                marker='o', markersize=3, markevery=20)
        ax6.plot(results_RP['Year'], results_RP['BCF'], 
                color=color_RP, linewidth=3, label='RP', 
                marker='s', markersize=3, markevery=20)
        
        residual_BCF = (self.residual_BCF_leafy * self.leafy_proportion +
                        self.residual_BCF_root * self.root_proportion +
                        self.residual_BCF_fruit * self.fruit_proportion)
        ax6.axhline(y=residual_BCF, color='purple', linestyle=':', 
                linewidth=2.5, alpha=0.6, 
                label=f'Residual BCF ({residual_BCF:.3f})')
        
        ax6.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax6.set_ylabel('Bioconcentration Factor (BCF)', fontsize=13, weight='bold', family='serif')
        ax6.set_title('(g) BCF Projection', fontsize=14, weight='bold', family='serif')
        ax6.legend(fontsize=9, loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax6.grid(alpha=0.3)
        ax6.set_xlim(2025, 2035)
        
        # ============================================================================
        # (h) 累积改善效应 - 蔬菜Cd降幅
        # ============================================================================
        ax7 = fig.add_subplot(gs[2, 0])
        veg_cd_reduction_BAU = (results_BAU['Veg_Cd'].iloc[0] - results_BAU['Veg_Cd']) / \
                            results_BAU['Veg_Cd'].iloc[0] * 100
        veg_cd_reduction_RP = (results_RP['Veg_Cd'].iloc[0] - results_RP['Veg_Cd']) / \
                            results_RP['Veg_Cd'].iloc[0] * 100
        
        ax7.fill_between(results_BAU['Year'], 0, veg_cd_reduction_BAU, 
                        color=color_BAU, alpha=0.3, label='BAU Reduction')
        ax7.fill_between(results_RP['Year'], 0, veg_cd_reduction_RP, 
                        color=color_RP, alpha=0.3, label='RP Reduction')
        ax7.plot(results_BAU['Year'], veg_cd_reduction_BAU, 
                color=color_BAU, linewidth=2.5)
        ax7.plot(results_RP['Year'], veg_cd_reduction_RP, 
                color=color_RP, linewidth=2.5)
        
        max_reduction = veg_cd_reduction_RP.max()
        ax7.text(2030, max_reduction*0.5, f'Max: {max_reduction:.1f}%', 
                fontsize=12, color=color_RP, weight='bold', family='serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        ax7.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax7.set_ylabel('Veg Cd Reduction (%)', fontsize=13, weight='bold', family='serif')
        ax7.set_title('(h) Cumulative Vegetable Cd Reduction', fontsize=14, weight='bold', family='serif')
        ax7.legend(fontsize=9, loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax7.grid(alpha=0.3)
        ax7.set_xlim(2025, 2035)
        
        # ============================================================================
        # (i) 累积改善效应 - THQ降幅
        # ============================================================================
        ax8 = fig.add_subplot(gs[2, 1])
        THQ_reduction_BAU = (results_BAU['Average_THQ'].iloc[0] - results_BAU['Average_THQ']) / \
                        results_BAU['Average_THQ'].iloc[0] * 100
        THQ_reduction_RP = (results_RP['Average_THQ'].iloc[0] - results_RP['Average_THQ']) / \
                        results_RP['Average_THQ'].iloc[0] * 100
        
        ax8.fill_between(results_BAU['Year'], 0, THQ_reduction_BAU, 
                        color=color_BAU, alpha=0.3, label='BAU Reduction')
        ax8.fill_between(results_RP['Year'], 0, THQ_reduction_RP, 
                        color=color_RP, alpha=0.3, label='RP Reduction')
        ax8.plot(results_BAU['Year'], THQ_reduction_BAU, 
                color=color_BAU, linewidth=2.5)
        ax8.plot(results_RP['Year'], THQ_reduction_RP, 
                color=color_RP, linewidth=2.5)
        
        policy_benefit = THQ_reduction_RP.iloc[-1] - THQ_reduction_BAU.iloc[-1]
        ax8.text(2030, 40, f'Policy Benefit:\n+{policy_benefit:.1f}%', 
                fontsize=12, color='darkgreen', weight='bold', family='serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        ax8.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax8.set_ylabel('Average THQ Reduction (%)', fontsize=13, weight='bold', family='serif')
        ax8.set_title('(i) Cumulative Health Risk (THQ) Reduction', fontsize=14, weight='bold', family='serif')
        ax8.legend(fontsize=9, loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold'})
        ax8.grid(alpha=0.3)
        ax8.set_xlim(2025, 2035)
        
        # ============================================================================
        # (j) 人群THQ达标率演变
        # ============================================================================
        ax9 = fig.add_subplot(gs[2, 2])
        
        pop_names_short = ['UM', 'UF', 'RM', 'RF']
        pop_labels = ['Urban Male', 'Urban Female', 'Rural Male', 'Rural Female']
        colors_pop = ['#3498DB', '#E74C3C', '#9B59B6', '#F39C12']
        
        # BAU情景（虚线）
        for i, col in enumerate(pop_cols):
            ax9.plot(results_BAU['Year'], results_BAU[col], 
                    color=colors_pop[i], linewidth=2.5, linestyle='--', 
                    alpha=0.6, label=f'{pop_names_short[i]} (BAU)')
        
        # RP情景（实线）
        for i, col in enumerate(pop_cols):
            ax9.plot(results_RP['Year'], results_RP[col], 
                    color=colors_pop[i], linewidth=3, linestyle='-', 
                    label=f'{pop_names_short[i]} (RP)')
        
        # 添加参考线
        ax9.axhline(y=1.0, color='red', linestyle='--', linewidth=2.5, 
                alpha=0.7, label='Safety Threshold (THQ=1)', zorder=0)
        ax9.axhline(y=0.5, color='green', linestyle=':', linewidth=2, 
                alpha=0.5, label='No Risk (THQ=0.5)', zorder=0)
        
        ax9.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax9.set_ylabel('THQ', fontsize=13, weight='bold', family='serif')
        ax9.set_title('(j) Population-Specific THQ Evolution', fontsize=14, weight='bold', family='serif')
        ax9.legend(fontsize=7.5, loc='upper right', ncol=2, framealpha=0.9, prop={'family': 'serif', 'size': 7.5})
        ax9.grid(alpha=0.3)
        ax9.set_xlim(2025, 2035)
        ax9.set_ylim(0, 2.0)
        
        # 添加注释
        ax9.text(2032, 0.3, 'All groups below\nTHQ=1 by 2035 (RP)', 
                fontsize=11, color='darkgreen', weight='bold', family='serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # 总标题
        plt.suptitle('System Dynamics Projection: Vegetable Cd Pollution & Health Risk (2025-2035)\n'
                    'Realistic Policy Scenario with Residual Baselines',
                    fontsize=18, weight='bold', family='serif', y=0.98)
        
        # 保存
        if save_path:
            plt.savefig(save_path, dpi=400, bbox_inches='tight', facecolor='white')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            print(f"\n✓ 图表已保存至: {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    
    def generate_summary_table(self, results_BAU, results_RP, save_path=None):
        """
        生成汇总对比表
        """
        summary_data = []
        
        indicators = [
            ('Soil_Cd', 'Soil Cd (mg/kg)'),
            ('pH', 'Soil pH'),
            ('SOM', 'SOM (g/kg)'),
            ('BCF', 'BCF'),
            ('Veg_Cd', 'Veg Cd (mg/kg)'),
            ('Average_THQ', 'Average THQ'),
            ('Urban_Male_THQ', 'Urban Male THQ'),
            ('Urban_Female_THQ', 'Urban Female THQ'),
            ('Rural_Male_THQ', 'Rural Male THQ'),
            ('Rural_Female_THQ', 'Rural Female THQ')
        ]
        
        for col, label in indicators:
            BAU_2025 = results_BAU[col].iloc[0]
            BAU_2035 = results_BAU[col].iloc[-1]
            BAU_change = (BAU_2035 - BAU_2025) / BAU_2025 * 100
            
            RP_2025 = results_RP[col].iloc[0]
            RP_2035 = results_RP[col].iloc[-1]
            RP_change = (RP_2035 - RP_2025) / RP_2025 * 100
            
            policy_benefit = ((BAU_2035 - RP_2035) / BAU_2035 * 100) if 'THQ' in col or 'Cd' in col or 'BCF' in col else ((RP_2035 - BAU_2035) / BAU_2035 * 100)
            
            summary_data.append({
                'Indicator': label,
                'BAU 2025': f'{BAU_2025:.4f}',
                'BAU 2035': f'{BAU_2035:.4f}',
                'BAU Change (%)': f'{BAU_change:+.2f}',
                'RP 2025': f'{RP_2025:.4f}',
                'RP 2035': f'{RP_2035:.4f}',
                'RP Change (%)': f'{RP_change:+.2f}',
                'Policy Benefit (%)': f'{policy_benefit:+.2f}'
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        print("\n" + "="*120)
        print("2025-2035年政策情景预测汇总对比表（现实约束版本）")
        print("="*120)
        print(summary_df.to_string(index=False))
        print("="*120)
        
        print("\n残留污染基线（不可消除部分）:")
        print(f"  - 土壤Cd: {self.residual_soil_cd:.2f} mg/kg (大气沉降+母质)")
        print(f"  - 蔬菜Cd: {self.residual_veg_cd:.3f} mg/kg (品种遗传+检测限)")
        print(f"  - 背景THQ: {self.background_THQ:.2f} (米+水+空气)")
        print("="*120)
        
        if save_path:
            summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 汇总表已保存至: {save_path}")
        
        return summary_df


# ============================================================================
# 主程序执行
# ============================================================================

if __name__ == "__main__":
    
    model = VegetableCdSystemDynamicsProjection()
    
    results_BAU, results_RP, results_combined = model.run_all_scenarios()
    
    model.visualize_projection(
        results_BAU, 
        results_RP, 
        save_path='SD_Projection_2025_2035_Realistic.png'
    )
    
    summary_df = model.generate_summary_table(
        results_BAU, 
        results_RP, 
        save_path='SD_Projection_2025_2035_Realistic_Summary.csv'
    )
    
    results_combined.to_csv('SD_Projection_2025_2035_Realistic_Full_Data.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("✓ 2025-2035年政策情景预测完成（现实约束版本）！")
    print("="*80)
    print("\n生成文件:")
    print("  1. SD_Projection_2025_2035_Realistic.png (可视化对比图)")
    print("  2. SD_Projection_2025_2035_Realistic.pdf (矢量图)")
    print("  3. SD_Projection_2025_2035_Realistic_Summary.csv (汇总对比表)")
    print("  4. SD_Projection_2025_2035_Realistic_Full_Data.csv (完整时间序列数据)")
    print("="*80 + "\n")