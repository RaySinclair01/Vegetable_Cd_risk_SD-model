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

# ============================================================================
# 版本说明（v05, 2025-09-24）
# ----------------------------------------------------------------------------
# 相对 v04 的修改：全部由数据库统计得到的参数、CB-SEM路径系数、边际效应参数
# 均已更新为"补充2022-2025数据 + 剔除土壤Cd>10"后的最新结果。
#
#   [1] 基准参数
#       来源：Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值.xlsx
#             （sheet: 01_Field_Dryland_Database, n=2446, 2004-2025）
#
#   [2] CB-SEM标准化路径系数
#       来源：结构方程图 code and data/Average_Population_Results_NewData/路径系数完整表.xlsx
#             （sheet: 图中路径系数；SEM拟合样本 N=111 平均人群数据）
#
#   [3] 边际效应参数（实际单位）
#       来源：图S11和图S12/p20_marginal_plot_new.py 的方法（标准化系数→实际单位换算：
#             β_unstd = β_std × SD_y/SD_x，再按论文报告的单位缩放），
#             输入为图S11和图S12/Average_Population_Results/original_data.xlsx（N=111）
#             与新版SEM系数。
#
#   所有参数已导出：SD_Parameters_v05_NewDB_NewSEM.xlsx（由本脚本自动生成）
# ============================================================================


class VegetableCdSystemDynamicsProjection:
    """
    蔬菜镉污染系统动力学模型 - 2025-2035年政策情景预测（CB-SEM因果机制版本）

    理论基础：
    ─────────────────────────────────────────────────────────────
    整合CB-SEM（Covariance-Based Structural Equation Modeling）
    路径系数与边际效应参数，建立多层次因果影响路径：

    Climate/Region → Soil Type → Soil Properties (pH/SOM/CEC)
                                      ↓
                                Bioavailability (BCF)
                                      ↓
                                Veg Cd Content
                                      ↓
                    Human Health Risk (THQ) ← Consumption Rate

    参数体系：
    ─────────────────────────────────────────────────────────────
    1. CB-SEM标准化路径系数（β）：相对影响强度
    2. 边际效应参数：实际单位变化的绝对影响
    3. 混合策略：CB-SEM用于机制识别，边际效应用于定量预测

    数据来源：
    ─────────────────────────────────────────────────────────────
    Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值（2004-2025, n=2446）
    CB-SEM分析：Average population model（N=111），补充2022-2025数据
    边际效应提取：标准化系数经样本标准差换算为实际单位（同图S11/S12方法）
    """

    def __init__(self):
        """
        初始化模型参数（基于新数据库2004-2025年统计 + 新版CB-SEM因果机制）
        """
        # ========== 时间参数 ==========
        self.start_year = 2025
        self.end_year = 2035
        self.time_horizon = self.end_year - self.start_year
        self.dt = 0.1
        self.time_steps = np.arange(0, self.time_horizon + self.dt, self.dt)

        # ========== 基准参数（2025年初始状态）==========
        # 来源：Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值.xlsx 全库均值
        #       （n=2446, 2004-2025；土壤Cd>10 mg/kg 记录已剔除）
        self.initial_soil_cd = 1.255809904758758
        self.initial_pH = 6.8103533333333335
        self.initial_SOM = 26.28703054187192
        self.initial_CEC = 15.251774999999999

        self.initial_veg_cd = 0.11979130945246438
        self.initial_BCF_all = 0.16177742473980247
        self.initial_BCF_leafy = 0.20612066361843318
        self.initial_BCF_root = 0.15954841846593698
        self.initial_BCF_fruit = 0.07778034985919477

        self.urban_male_weight = 66.14975470155356
        self.urban_female_weight = 56.933646770237125
        self.rural_male_weight = 62.89787408013083
        self.rural_female_weight = 55.19570727718724

        self.urban_consumption = 107.20956663941128
        self.rural_consumption = 99.56541291905151

        self.initial_urban_male_THQ = 1.160353390045581
        self.initial_urban_female_THQ = 1.3714144639301937
        self.initial_rural_male_THQ = 1.1780060774000294
        self.initial_rural_female_THQ = 1.3356772979473153

        # ========== CB-SEM标准化路径系数（β）==========
        # 来源：路径系数完整表.xlsx（图中路径系数 sheet；N=111）
        # 标准化路径系数：用于评估相对影响强度
        # 注意：土壤Cd 与 SCC 在85%的记录上取值相同（VIF>300），二者系数不可单独解读，
        #       故取两者之和作为"土壤Cd库 → BCF"的标准化效应（-0.726+1.223=+0.497）
        self.beta_soil_cd_to_BCF = 0.49681809760697937   # = -0.725834 (Soil Cd) + 1.222652 (SCC)；p>0.05（共线）
        self.beta_pH_to_BCF = -0.07130603517108515       # p = 0.512
        self.beta_SOM_to_BCF = -0.3824043492983312       # p = 0.003 **
        self.beta_CEC_to_BCF = 0.1356720850037091        # p = 0.160
        self.beta_BCF_to_veg_cd = 0.3792650572939975     # p < 0.001 ***
        self.beta_veg_cd_to_THQ = 1.001075066045007      # p < 0.001 ***（定义式关系，仅作展示）
        self.beta_consumption_to_THQ = 0.006528397455796445  # p = 0.044 *

        # ========== 边际效应参数（实际单位）==========
        # 来源：图S11/S12 的 p20 方法（β_unstd = β_std × SD_y/SD_x）
        #       输入：Average_Population_Results/original_data.xlsx（N=111）+ 新版SEM系数
        # 复现：p24_v05_params_derivation.py
        # pH边际效应：pH每上升1个单位 → BCF减少9.06%
        # （SEM换算值；p=0.512，95% CI：每下降1单位 BCF变化 -17.99% ~ +36.10%）
        self.pH_effect_per_unit = -0.09058246534486866

        # CEC边际效应：CEC每增加5 cmol/kg → BCF增加16.91%
        # （p=0.160，95% CI：-6.65% ~ +40.47%）
        self.CEC_effect_per_5cmol = 0.1691028309824471

        # SOM边际效应：SOM每增加1 g/kg → BCF减少2.04%
        # 即每增加1%（≈10 g/kg）→ BCF减少20.38%（p=0.003，95% CI：-6.94% ~ -33.82%）
        # 注：v04 中该参数名为 SOM_effect_per_1pct（-0.0078），但当时是"每1 g/kg"应用，
        #     与命名/论文"每1%"口径不一致；本版统一为"每1 g/kg"口径并更名。
        self.SOM_effect_per_gkg = -0.02037829959736018

        # 蔬菜Cd边际效应：蔬菜Cd每增加0.1 mg/kg → THQ增加0.918
        # （p<0.001，95% CI：0.913 ~ 0.923）
        self.marginal_veg_cd_per_01mg = 0.9175348697800987

        # 消费量边际效应：消费量每增加10 kg/年 → THQ增加0.231
        # （p=0.044，95% CI：0.006 ~ 0.456）
        self.marginal_consumption_per_10kg = 0.23106142707715738

        # ========== 自然衰减与演化速率 ==========
        self.soil_cd_natural_decay = 0.02
        self.pH_natural_buffering = 0.005793169830430412
        self.SOM_decomposition = 0.0
        self.BCF_natural_decay = 0.0
        self.target_pH = 7.0

        # ========== 蔬菜类型分布 ==========
        # 来源：新数据库 Major Veg Category 占比
        self.leafy_proportion = 0.509403107113655
        self.root_proportion = 0.2277187244480785
        self.fruit_proportion = 0.26287816843826656

        # ========== 残留污染基线 ==========
        # 来源：新数据库各变量的最小值（不可消除的背景水平）
        self.residual_soil_cd = 0.00685
        self.residual_veg_cd = 8.1e-05
        self.residual_BCF_all = 0.0002659574468085106
        self.residual_BCF_leafy = 0.0008166666666666674
        self.residual_BCF_root = 0.0004166666666666667
        self.residual_BCF_fruit = 0.0002659574468085106

        # ========== 背景暴露叠加 ==========
        self.background_THQ = 0.30144883719685983

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
            'soil_remediation_rate': 0.03,
            'pH_amendment_rate': 0.10,
            'SOM_increase_rate': 1.2,
            'BCF_reduction_factor': 0.18,
            'consumption_reduction': 0.10,
            'veg_cd_market_control': 0.12
        }

        print("="*120)
        print("蔬菜镉污染系统动力学模型 - 2025-2035年政策情景预测（CB-SEM因果机制版本 v05）")
        print("Realistic SD Model with CB-SEM Causal Pathways & Marginal Effects")
        print("="*120)

        print(f"\n【理论基础】")
        print(f"  ✓ CB-SEM多层次因果影响路径分析")
        print(f"  ✓ Climate/Region → Soil Type → Soil Properties → BCF → Veg Cd → THQ")
        print(f"  ✓ 整合标准化路径系数（相对影响）+ 边际效应参数（绝对影响）")

        print(f"\n【时间范围】")
        print(f"  {self.start_year}-{self.end_year} ({self.time_horizon}年)")

        print(f"\n【初始状态 (2025基线)】")
        print(f"  - 土壤Cd: {self.initial_soil_cd:.4f} mg/kg (残留下限: {self.residual_soil_cd:.5f})")
        print(f"  - 土壤pH: {self.initial_pH:.2f} (目标: {self.target_pH:.1f})")
        print(f"  - SOM: {self.initial_SOM:.2f} g/kg")
        print(f"  - CEC: {self.initial_CEC:.2f} cmol/kg")
        print(f"  - 蔬菜Cd: {self.initial_veg_cd:.4f} mg/kg (残留下限: {self.residual_veg_cd:.7f})")
        print(f"  - 平均BCF: {self.initial_BCF_all:.4f} (残留下限: {self.residual_BCF_all:.7f})")
        print(f"  - 平均THQ: {np.mean([self.initial_urban_male_THQ, self.initial_urban_female_THQ, self.initial_rural_male_THQ, self.initial_rural_female_THQ]):.4f}")
        print(f"  - 背景THQ: {self.background_THQ:.4f} (来自非蔬菜源)")

        print(f"\n【CB-SEM标准化路径系数（β）】（新版SEM, N=111）")
        print(f"  土壤Cd → BCF:      {self.beta_soil_cd_to_BCF:+.3f}  (Soil Cd 与 SCC 之和；两者共线不可单独解读)")
        print(f"  pH → BCF:          {self.beta_pH_to_BCF:+.3f}  (p = 0.512)")
        print(f"  SOM → BCF:         {self.beta_SOM_to_BCF:+.3f}  (p = 0.003 **)")
        print(f"  CEC → BCF:         {self.beta_CEC_to_BCF:+.3f}  (p = 0.160)")
        print(f"  BCF → 蔬菜Cd:      {self.beta_BCF_to_veg_cd:+.3f}  (p < 0.001 ***)")
        print(f"  蔬菜Cd → THQ:      {self.beta_veg_cd_to_THQ:+.3f}  (p < 0.001 ***；定义式关系)")
        print(f"  消费量 → THQ:      {self.beta_consumption_to_THQ:+.3f}  (p = 0.044 *)")

        print(f"\n【边际效应参数（实际单位）】（图S11/S12方法换算，N=111）")
        print(f"  pH每↑1 单位:       BCF减少 {abs(self.pH_effect_per_unit)*100:.2f}% (95% CI: -17.99% ~ +36.10%)")
        print(f"  CEC每↑5 cmol/kg:   BCF增加 {self.CEC_effect_per_5cmol*100:.2f}% (95% CI: -6.65% ~ +40.47%)")
        print(f"  SOM每↑1 g/kg:      BCF减少 {abs(self.SOM_effect_per_gkg)*100:.2f}% (≈每↑1%即10 g/kg 减少20.38%; 95% CI: -6.94% ~ -33.82%)")
        print(f"  蔬菜Cd每↑0.1mg/kg: THQ增加 {self.marginal_veg_cd_per_01mg:.3f} (95% CI: 0.913 ~ 0.923)")
        print(f"  消费量每↑10kg/年:  THQ增加 {self.marginal_consumption_per_10kg:.3f} (95% CI: 0.006 ~ 0.456)")

        print("\n【政策情景设置】")
        print("  [1] BAU情景: 无政策干预（维持现状）")
        print("  [2] RP情景: 现实约束下的建议政策（考虑边际递减）")

        print(f"\n【数据来源】")
        print(f"  ✓ Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值（2004-2025, n=2446）")
        print(f"  ✓ CB-SEM平均人群模型（N=111，补充2022-2025数据）")
        print(f"  ✓ 边际效应：标准化系数经样本SD换算为实际单位（图S11/S12方法）")

        print("="*120 + "\n")


    def policy_efficiency_decay(self, t):
        """
        政策效率随时间衰减（边际递减规律）
        """
        decay_rate = np.log(2) / self.policy_efficiency_halflife
        efficiency = self.min_policy_efficiency + \
                    (1.0 - self.min_policy_efficiency) * np.exp(-decay_rate * t)
        return efficiency


    def calculate_BCF_CBSEM(self, soil_cd, pH, SOM, CEC, policy_BCF_reduction=0, t=0):
        """
        计算加权平均BCF（基于CB-SEM因果机制 + 边际效应参数）

        方法论：
        ─────────────────────────────────────────────────
        1. CB-SEM路径系数：用于相对影响强度判断
        2. 边际效应参数：用于精确定量预测
        3. 混合策略：
           - 对于pH/CEC/SOM：主要使用边际效应（已换算为实际单位）
           - 对于土壤Cd：使用CB-SEM路径系数（无独立边际效应）
        ─────────────────────────────────────────────────

        新版参数依据（N=111, 补充2022-2025数据）：
        ─────────────────────────────────────────────────
        - 土壤Cd库 → BCF：β=+0.497（Soil Cd -0.726 与 SCC +1.223 之和；
          两者在85%记录上相同、VIF>300，不可单独解读）
        - pH每下降1 → BCF增加9.06%（β=-0.071, p=0.512）
        - CEC每增加5 cmol/kg → BCF增加16.91%（β=+0.136, p=0.160）
        - SOM每增加1 g/kg → BCF减少2.04%（β=-0.382, p=0.003）
        ─────────────────────────────────────────────────
        """
        # 实际变化量（用于边际效应）
        pH_change = pH - self.initial_pH
        SOM_change = SOM - self.initial_SOM
        CEC_change = CEC - self.initial_CEC

        # 标准化（用于CB-SEM路径系数）
        soil_cd_norm = (soil_cd - self.initial_soil_cd) / self.initial_soil_cd

        # 政策效率
        policy_efficiency = self.policy_efficiency_decay(t)
        effective_BCF_reduction = policy_BCF_reduction * policy_efficiency

        def calc_single_BCF(base_BCF, residual_BCF):
            """计算单一蔬菜类型的BCF"""

            # ========== 起始值 ==========
            BCF = base_BCF

            # ========== 土壤Cd效应（使用CB-SEM路径系数）==========
            # 土壤Cd库是BCF的污染源（标准化总效应 β=+0.497）
            BCF *= (1 + self.beta_soil_cd_to_BCF * soil_cd_norm)

            # ========== pH边际效应（使用实际单位参数）==========
            # 每下降1个单位 → BCF增加9.06%（换算自新版SEM, N=111）
            # pH上升 → BCF减少9.06%
            BCF *= (1 + self.pH_effect_per_unit * pH_change)

            # ========== CEC边际效应（使用实际单位参数）==========
            # 每增加5 cmol/kg → BCF增加16.91%
            BCF *= (1 + self.CEC_effect_per_5cmol * (CEC_change / 5.0))

            # ========== SOM边际效应（使用实际单位参数）==========
            # 每增加1 g/kg → BCF减少2.04%（即每增加1%≈10 g/kg → 减少20.38%）
            BCF *= (1 + self.SOM_effect_per_gkg * SOM_change)

            # ========== 政策影响 ==========
            BCF *= (1 - effective_BCF_reduction)

            # ========== 自然演化 ==========
            BCF *= np.exp(-self.BCF_natural_decay * t)

            # ========== 不低于残留基线 ==========
            BCF = max(residual_BCF, BCF)

            return BCF

        # 分别计算三类蔬菜的BCF
        BCF_leafy = calc_single_BCF(self.initial_BCF_leafy, self.residual_BCF_leafy)
        BCF_root = calc_single_BCF(self.initial_BCF_root, self.residual_BCF_root)
        BCF_fruit = calc_single_BCF(self.initial_BCF_fruit, self.residual_BCF_fruit)

        # 加权平均
        BCF_avg = (BCF_leafy * self.leafy_proportion +
                   BCF_root * self.root_proportion +
                   BCF_fruit * self.fruit_proportion)

        return max(0.001, BCF_avg)


    def calculate_veg_cd(self, soil_cd, BCF, policy_market_control=0, t=0):
        """
        计算蔬菜Cd浓度（基于CB-SEM路径 + 残留下限约束）

        新版参数依据（N=111）：
        ─────────────────────────────────────────────────
        "BCF (β=0.379, p<0.001) 与 土壤Cd (β=0.574, p<0.001) 共同作用于
         蔬菜Cd累积过程"
        ─────────────────────────────────────────────────
        """
        # 政策效率
        policy_efficiency = self.policy_efficiency_decay(t)
        effective_market_control = policy_market_control * policy_efficiency

        # 基础蔬菜Cd = 土壤Cd × BCF
        veg_cd = soil_cd * BCF

        # CB-SEM路径系数效应（BCF → 蔬菜Cd）
        BCF_norm = (BCF - self.initial_BCF_all) / self.initial_BCF_all
        veg_cd *= (1 + self.beta_BCF_to_veg_cd * BCF_norm)

        # 市场控制效应
        veg_cd *= (1 - effective_market_control)

        # 不低于残留基线
        veg_cd = max(self.residual_veg_cd, veg_cd)

        return veg_cd


    def calculate_THQ_CBSEM(self, veg_cd, consumption, body_weight, t=0, is_urban=True):
        """
        计算THQ（基于CB-SEM因果机制 + 背景暴露）

        ✅ 修正说明（2025-01-21）：
        ─────────────────────────────────────────────────
        问题：原方法中边际效应的叠加导致RP 2035年THQ被错误
              计算为接近0，所有人群THQ趋同于背景值0.30

        根源：
        1. CB-SEM路径系数已在BCF等上游环节体现
        2. 边际效应参数不应在THQ计算中重复应用
        3. 负向deviation导致THQ被过度削减

        修正：回归WHO推荐的经典THQ线性公式
        ─────────────────────────────────────────────────

        新版参数依据（N=111）：
        ─────────────────────────────────────────────────
        "the Cd content in vegetables becomes the dominant factor
        determining human health risks (β=1.001, p<0.001)"

        THQ公式：
        EDI = (Veg_Cd × Consumption) / (Body_Weight × 365)
        THQ = EDI / RfD
        Total_THQ = THQ_veg + Background_THQ
        ─────────────────────────────────────────────────
        """
        # ========== 日均蔬菜Cd摄入量 (mg/kg-bw/day) ==========
        EDI_veg = (veg_cd * consumption) / (body_weight * 365)

        # ========== 参考剂量 (mg/kg-bw/day) ==========
        RfD = 0.001

        # ========== 蔬菜THQ（WHO经典公式）==========
        THQ_veg = EDI_veg / RfD

        # ========== 背景THQ（来自大米、饮用水、大气沉降，保持稳定）==========
        background_THQ_current = self.background_THQ

        # ========== 总THQ = 蔬菜暴露 + 背景暴露 ==========
        THQ_total = THQ_veg + background_THQ_current

        return max(0, THQ_total)


    def system_dynamics_equations(self, state, t, policy_params):
        """
        系统动力学微分方程组（CB-SEM因果机制版 + 现实约束）

        理论框架：
        ─────────────────────────────────────────────────
        基于CB-SEM识别的多层次因果影响路径（新版参数, N=111）：

        (1) Soil Cd → BCF (β=+0.497, 土壤Cd库总效应)
        (2) pH → BCF (β=-0.071)
                  每↑1 → BCF减少9.06%
        (3) CEC → BCF (β=+0.136)
                   每↑5 cmol/kg → BCF增加16.91%
        (4) SOM → BCF (β=-0.382, p=0.003)
                      每↑1 g/kg → BCF减少2.04%
        (5) BCF → Veg Cd (β=0.379, p<0.001)
        (6) Veg Cd → THQ (β=1.001, p<0.001)
                      使用经典THQ公式
        (7) Consumption → THQ (线性关系)
        ─────────────────────────────────────────────────
        """
        # 解包状态变量
        soil_cd, pH, SOM, CEC, BCF_avg, veg_cd, \
        UM_THQ, UF_THQ, RM_THQ, RF_THQ = state

        # 提取政策参数
        soil_remed = policy_params['soil_remediation_rate']
        pH_amend = policy_params['pH_amendment_rate']
        SOM_incr = policy_params['SOM_increase_rate']
        BCF_reduc = policy_params['BCF_reduction_factor']
        consump_reduc = policy_params['consumption_reduction']
        market_ctrl = policy_params['veg_cd_market_control']

        # 政策效率
        efficiency = self.policy_efficiency_decay(t)

        # ========== (1) 土壤Cd动态 ==========
        removable_soil_cd = max(0, soil_cd - self.residual_soil_cd)
        natural_decay = -removable_soil_cd * self.soil_cd_natural_decay
        policy_remediation = -removable_soil_cd * soil_remed * efficiency
        atmospheric_deposition = 0.015

        dSoil_Cd_dt = natural_decay + policy_remediation + atmospheric_deposition

        # ========== (2) pH动态 ==========
        pH_natural_change = (self.target_pH - pH) * self.pH_natural_buffering
        pH_policy_change = pH_amend * efficiency
        dpH_dt = pH_natural_change + pH_policy_change

        if pH > 8.5:
            dpH_dt = min(0, dpH_dt)
        elif pH > 8.0:
            dpH_dt *= 0.2

        # ========== (3) SOM动态 ==========
        natural_decomp = -SOM * self.SOM_decomposition
        policy_increase = SOM_incr * efficiency
        dSOM_dt = natural_decomp + policy_increase

        # ========== (4) CEC动态 ==========
        # CEC与SOM协同变化
        CEC_target = self.initial_CEC + (SOM - self.initial_SOM) * 0.18
        dCEC_dt = (CEC_target - CEC) * 0.12

        # ========== (5) BCF动态（使用CB-SEM因果机制）==========
        new_BCF = self.calculate_BCF_CBSEM(soil_cd, pH, SOM, CEC, BCF_reduc, t)
        dBCF_dt = (new_BCF - BCF_avg) * 0.6

        # ========== (6) 蔬菜Cd动态（CB-SEM路径）==========
        new_veg_cd = self.calculate_veg_cd(soil_cd, BCF_avg, market_ctrl, t)
        dVeg_Cd_dt = (new_veg_cd - veg_cd) * 0.7

        # ========== (7-10) THQ动态（经典公式 + 背景暴露）==========
        UM_consump = self.urban_consumption * (1 - consump_reduc * efficiency)
        UF_consump = self.urban_consumption * (1 - consump_reduc * efficiency)
        RM_consump = self.rural_consumption * (1 - consump_reduc * efficiency)
        RF_consump = self.rural_consumption * (1 - consump_reduc * efficiency)

        new_UM_THQ = self.calculate_THQ_CBSEM(veg_cd, UM_consump, self.urban_male_weight, t, is_urban=True)
        new_UF_THQ = self.calculate_THQ_CBSEM(veg_cd, UF_consump, self.urban_female_weight, t, is_urban=True)
        new_RM_THQ = self.calculate_THQ_CBSEM(veg_cd, RM_consump, self.rural_male_weight, t, is_urban=False)
        new_RF_THQ = self.calculate_THQ_CBSEM(veg_cd, RF_consump, self.rural_female_weight, t, is_urban=False)

        dUM_THQ_dt = (new_UM_THQ - UM_THQ) * 0.7
        dUF_THQ_dt = (new_UF_THQ - UF_THQ) * 0.7
        dRM_THQ_dt = (new_RM_THQ - RM_THQ) * 0.7
        dRF_THQ_dt = (new_RF_THQ - RF_THQ) * 0.7

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
            args=(policy_params,),
            rtol=1e-6,
            atol=1e-8
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

        print(f"\n{'='*100}")
        print(f"✓ {scenario_name} 情景模拟完成（基于CB-SEM因果机制）")
        print(f"{'='*100}")
        print(f"  2025年基线:")
        print(f"    - 土壤Cd:   {results['Soil_Cd'].iloc[0]:.4f} mg/kg")
        print(f"    - pH:       {results['pH'].iloc[0]:.2f}")
        print(f"    - SOM:      {results['SOM'].iloc[0]:.2f} g/kg")
        print(f"    - CEC:      {results['CEC'].iloc[0]:.2f} cmol/kg")
        print(f"    - BCF:      {results['BCF'].iloc[0]:.4f}")
        print(f"    - 蔬菜Cd:   {results['Veg_Cd'].iloc[0]:.4f} mg/kg")
        print(f"    - 平均THQ:  {results['Average_THQ'].iloc[0]:.3f}")

        print(f"\n  2035年预测:")
        print(f"    - 土壤Cd:   {results['Soil_Cd'].iloc[-1]:.4f} mg/kg")
        print(f"    - pH:       {results['pH'].iloc[-1]:.2f}")
        print(f"    - SOM:      {results['SOM'].iloc[-1]:.2f} g/kg")
        print(f"    - CEC:      {results['CEC'].iloc[-1]:.2f} cmol/kg")
        print(f"    - BCF:      {results['BCF'].iloc[-1]:.4f}")
        print(f"    - 蔬菜Cd:   {results['Veg_Cd'].iloc[-1]:.4f} mg/kg")
        print(f"    - 平均THQ:  {results['Average_THQ'].iloc[-1]:.3f}")

        # 打印各人群2035年THQ（验证差异性）
        print(f"\n  2035年各人群THQ:")
        print(f"    - 城市男性: {results['Urban_Male_THQ'].iloc[-1]:.3f}")
        print(f"    - 城市女性: {results['Urban_Female_THQ'].iloc[-1]:.3f}")
        print(f"    - 农村男性: {results['Rural_Male_THQ'].iloc[-1]:.3f}")
        print(f"    - 农村女性: {results['Rural_Female_THQ'].iloc[-1]:.3f}")

        print(f"\n  10年变化:")
        print(f"    - 蔬菜Cd: {(results['Veg_Cd'].iloc[-1]/results['Veg_Cd'].iloc[0]-1)*100:+.2f}%")
        print(f"    - 平均THQ: {(results['Average_THQ'].iloc[-1]/results['Average_THQ'].iloc[0]-1)*100:+.2f}%")

        if results['Average_THQ'].iloc[-1] < 1.0:
            print(f"\n  ✓ 2035年达标（THQ < 1.0）")
        else:
            print(f"\n  ✗ 2035年未达标（THQ ≥ 1.0）")

        if 'RP' in scenario_name:
            soil_approach = (results['Soil_Cd'].iloc[-1] - self.residual_soil_cd) / \
                           (self.initial_soil_cd - self.residual_soil_cd) * 100
            veg_approach = (results['Veg_Cd'].iloc[-1] - self.residual_veg_cd) / \
                          (self.initial_veg_cd - self.residual_veg_cd) * 100

            print(f"\n  残留基线接近度:")
            print(f"    - 土壤Cd: 距基线{self.residual_soil_cd:.5f} mg/kg还有{soil_approach:.1f}%")
            print(f"    - 蔬菜Cd: 距基线{self.residual_veg_cd:.7f} mg/kg还有{veg_approach:.1f}%")

        print(f"{'='*100}\n")

        return results


    def run_all_scenarios(self):
        """
        运行所有情景并对比
        """
        print("\n" + "="*100)
        print("开始运行2025-2035年政策情景预测（CB-SEM因果机制版本 v05）...")
        print("="*100)

        results_BAU = self.run_projection(self.policy_BAU, 'BAU (No Policy)')
        results_RP = self.run_projection(self.policy_RP, 'RP (Recommended Policy)')

        results_combined = pd.concat([results_BAU, results_RP], ignore_index=True)

        return results_BAU, results_RP, results_combined


    def visualize_projection(self, results_BAU, results_RP, save_path=None):
        """
        可视化预测结果对比（完整9子图版本，编号b-j）
        所有图例缩小到原来的75%，标注框去除外框线和背景色
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
        ax1.legend(loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.075})
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
        ax2.legend(loc='best', ncol=2, framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 7.6})
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
        ax3.legend(loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.55})
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
        ax4.legend(ncol=2, loc='upper left', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.075})
        ax4.grid(alpha=0.3, axis='y')

        # ============================================================================
        # (f) pH趋势
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
        ax5.legend(loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.55})
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
        ax6.legend(loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.55})
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
                bbox=dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='none'))

        ax7.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax7.set_ylabel('Veg Cd Reduction (%)', fontsize=13, weight='bold', family='serif')
        ax7.set_title('(h) Cumulative Vegetable Cd Reduction', fontsize=14, weight='bold', family='serif')
        ax7.legend(loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.55})
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
        annotation_y = max(THQ_reduction_RP.max(), THQ_reduction_BAU.max()) * 0.55
        ax8.text(2030, annotation_y, f'Policy Benefit:\n+{policy_benefit:.1f}%',
                fontsize=12, color='darkgreen', weight='bold', family='serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='none'))

        ax8.set_xlabel('Year', fontsize=13, weight='bold', family='serif')
        ax8.set_ylabel('Average THQ Reduction (%)', fontsize=13, weight='bold', family='serif')
        ax8.set_title('(i) Cumulative Health Risk (THQ) Reduction', fontsize=14, weight='bold', family='serif')
        ax8.legend(loc='best', framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 8.55})
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
        ax9.legend(loc='upper right', ncol=2, framealpha=0.9, prop={'family': 'serif', 'weight': 'bold', 'size': 7.125})
        ax9.grid(alpha=0.3)
        ax9.set_xlim(2025, 2035)
        ax9.set_ylim(0, 2.0)

        # 添加注释（根据RP情景2035年是否全部达标自动生成）
        all_below = all(results_RP[col].iloc[-1] < 1.0 for col in pop_cols)
        note = ('All groups below\nTHQ=1 by 2035 (RP)' if all_below
                else 'Some groups still\nabove THQ=1 in 2035 (RP)')
        ax9.text(2031.5, 0.3, note,
                fontsize=11, color='darkgreen' if all_below else 'darkred',
                weight='bold', family='serif',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='none', edgecolor='none'))

        # 总标题
        plt.suptitle('System Dynamics Projection: Vegetable Cd Pollution & Health Risk (2025-2035)\n'
                    'CB-SEM Causal Pathways with Marginal Effects & Residual Baselines',
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
        print("2025-2035年政策情景预测汇总对比表（CB-SEM因果机制版本 v05 - 新参数）")
        print("="*120)
        print(summary_df.to_string(index=False))
        print("="*120)

        print("\n【残留污染基线（不可消除部分）】")
        print(f"  - 土壤Cd: {self.residual_soil_cd:.5f} mg/kg (大气沉降+母质)")
        print(f"  - 蔬菜Cd: {self.residual_veg_cd:.7f} mg/kg (品种遗传+检测限)")
        print(f"  - 背景THQ: {self.background_THQ:.4f} (米+水+空气)")

        print("\n【CB-SEM边际效应参数应用】（新版SEM换算值, N=111）")
        print(f"  ✓ pH每上升1 → BCF减少9.06% (95% CI: -17.99% ~ +36.10%)")
        print(f"  ✓ CEC每增加5 cmol/kg → BCF增加16.91% (95% CI: -6.65% ~ +40.47%)")
        print(f"  ✓ SOM每增加1 g/kg → BCF减少2.04% (≈每增加1% 减少20.38%)")
        print(f"  ✓ THQ采用WHO经典公式（不重复应用边际效应）")

        print("\n【参数更新说明（v05, 2025-09-24）】")
        print(f"  基准参数：Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值（n=2446）")
        print(f"  路径系数：路径系数完整表.xlsx（新版SEM, N=111）")
        print(f"  边际效应：图S11/S12 的 p20 换算方法 + 新版SEM系数")
        print(f"  参数明细已导出：SD_Parameters_v05_NewDB_NewSEM.xlsx")

        print("="*120)

        if save_path:
            summary_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"\n✓ 汇总表已保存至: {save_path}")

        return summary_df


    @staticmethod
    def _round_floats(df, nd=6):
        """数值列保留6位小数（脚本内仍使用更高精度的原始值）"""
        out = df.copy()
        for c in out.columns:
            if pd.api.types.is_float_dtype(out[c]):
                out[c] = out[c].round(nd)
        return out

    @staticmethod
    def _format_sheet(ws, wrap_cols=(), max_width=58):
        """
        统一Excel表格外观：表头加粗居中、冻结首行、按内容自适应列宽、长文本自动换行
        """
        from openpyxl.styles import Font, Alignment
        from openpyxl.utils import get_column_letter

        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        ws.freeze_panes = 'A2'

        def disp_width(value):
            text = '' if value is None else str(value)
            return sum(2 if ord(ch) > 127 else 1 for ch in text)

        for idx, col in enumerate(ws.columns, start=1):
            header = ws.cell(row=1, column=idx).value
            longest = max([disp_width(header)] + [disp_width(c.value) for c in col])
            ws.column_dimensions[get_column_letter(idx)].width = min(max(10, longest + 2), max_width)
            if header in wrap_cols:
                for c in col:
                    c.alignment = Alignment(wrap_text=True, vertical='top')

    def save_parameters_to_excel(self, save_path):
        """
        将所有模型参数（基准参数、CB-SEM路径系数、边际效应、模型常量、政策情景）
        导出为Excel，并附来源说明与v04→v05变更对照。
        """
        # ---------- 参数来源与说明 ----------
        src_db = 'Comprehensive_Database_Field_Dryland_ONLY_Cd去除异常值.xlsx (n=2446, 2004-2025)'
        src_sem = '路径系数完整表.xlsx (新版SEM, N=111, 补充2022-2025数据)'
        src_me = '图S11/S12 p20方法换算 (original_data.xlsx N=111 + 新版SEM系数)'

        baseline_def = [
            ('initial_soil_cd', '初始土壤Cd', 'mg/kg', '全库均值', 1.2662134336025357),
            ('initial_pH', '初始土壤pH', '-', '全库均值', 7.057777777777778),
            ('initial_SOM', '初始SOM', 'g/kg', '全库均值', 32.51247407407408),
            ('initial_CEC', '初始CEC', 'cmol/kg', '全库均值', 18.11740740740741),
            ('initial_veg_cd', '初始蔬菜Cd', 'mg/kg', '全库均值', 0.13136505895874265),
            ('initial_BCF_all', '初始BCF（总体）', '-', '全库均值', 0.14289611724818072),
            ('initial_BCF_leafy', '初始BCF（叶菜）', '-', '叶菜类均值', 0.18067332144132378),
            ('initial_BCF_root', '初始BCF（根菜）', '-', '根菜类均值', 0.12276309699928978),
            ('initial_BCF_fruit', '初始BCF（果菜）', '-', '果菜类均值', 0.06436252585809678),
            ('urban_male_weight', '城市男性体重', 'kg', '全库均值', 66.25795677799607),
            ('urban_female_weight', '城市女性体重', 'kg', '全库均值', 57.2361493123772),
            ('rural_male_weight', '农村男性体重', 'kg', '全库均值', 62.34302554027504),
            ('rural_female_weight', '农村女性体重', 'kg', '全库均值', 55.207858546168964),
            ('urban_consumption', '城市蔬菜消费量', 'kg/year/capita', '全库均值', 107.5770137524558),
            ('rural_consumption', '农村蔬菜消费量', 'kg/year/capita', '全库均值', 98.38546168958742),
            ('initial_urban_male_THQ', '初始城市男性THQ', '-', '全库均值', 1.3172232807393542),
            ('initial_urban_female_THQ', '初始城市女性THQ', '-', '全库均值', 1.532759892495895),
            ('initial_rural_male_THQ', '初始农村男性THQ', '-', '全库均值', 1.3384464267888618),
            ('initial_rural_female_THQ', '初始农村女性THQ', '-', '全库均值', 1.5170779462889459),
            ('leafy_proportion', '叶菜占比', '-', '分类占比', 0.518698578908003),
            ('root_proportion', '根菜占比', '-', '分类占比', 0.24607329842931938),
            ('fruit_proportion', '果菜占比', '-', '分类占比', 0.23522812266267765),
            ('residual_soil_cd', '土壤Cd残留下限', 'mg/kg', '全库最小值', 0.00685),
            ('residual_veg_cd', '蔬菜Cd残留下限', 'mg/kg', '全库最小值', 0.0002310000000000002),
            ('residual_BCF_all', 'BCF残留下限（总体）', '-', '全库最小值', 0.0002659574468085106),
            ('residual_BCF_leafy', 'BCF残留下限（叶菜）', '-', '叶菜最小值', 0.0008166666666666674),
            ('residual_BCF_root', 'BCF残留下限（根菜）', '-', '根菜最小值', 0.0004166666666666667),
            ('residual_BCF_fruit', 'BCF残留下限（果菜）', '-', '果菜最小值', 0.0002659574468085106),
        ]

        beta_def = [
            ('beta_soil_cd_to_BCF', '土壤Cd → BCF', 0.595,
             '新建模：Soil Cd (-0.7258, p=0.624) 与 SCC (+1.2227, p=0.407) 之和；两者在85%记录上相同(VIF>300)，'
             '不可单独解读，故采用“土壤Cd库”总效应'),
            ('beta_pH_to_BCF', 'pH → BCF', -0.346, '新版SEM标准化系数 (p=0.512)'),
            ('beta_SOM_to_BCF', 'SOM → BCF', -0.21, '新版SEM标准化系数 (p=0.003 **)'),
            ('beta_CEC_to_BCF', 'CEC → BCF', 0.222, '新版SEM标准化系数 (p=0.160)'),
            ('beta_BCF_to_veg_cd', 'BCF → 蔬菜Cd', 0.247, '新版SEM标准化系数 (p<0.001 ***)'),
            ('beta_veg_cd_to_THQ', '蔬菜Cd → THQ', 0.999, '新版SEM标准化系数 (p<0.001 ***；定义式关系，仅作展示)'),
            ('beta_consumption_to_THQ', '消费量 → THQ', 0.016, '新版SEM标准化系数 (p=0.044 *)'),
        ]
        beta_new = {
            'beta_soil_cd_to_BCF': self.beta_soil_cd_to_BCF,
            'beta_pH_to_BCF': self.beta_pH_to_BCF,
            'beta_SOM_to_BCF': self.beta_SOM_to_BCF,
            'beta_CEC_to_BCF': self.beta_CEC_to_BCF,
            'beta_BCF_to_veg_cd': self.beta_BCF_to_veg_cd,
            'beta_veg_cd_to_THQ': self.beta_veg_cd_to_THQ,
            'beta_consumption_to_THQ': self.beta_consumption_to_THQ,
        }

        # 边际效应：新版换算值 / 已发表图S11-S12值(旧SEM系数) / 清洗数据版备选
        # report列按论文/图S11-S12的报告口径（BCF类为%，THQ类为绝对值）；CI基于同一报告口径
        me_rows = [
            dict(key='pH_effect_per_unit', name='pH边际效应', basis='每上升1个pH单位 → BCF比例变化',
                 old=-0.365, new=self.pH_effect_per_unit, report='9.06%',
                 ci_low=-17.9856, ci_high=36.1021, alt_clean=9.1676,
                 note='论文口径：pH每下降1单位 → BCF增加9.06%（95% CI: -17.99% ~ +36.10%）'),
            dict(key='CEC_effect_per_5cmol', name='CEC边际效应', basis='每增加5 cmol/kg → BCF比例变化',
                 old=0.245, new=self.CEC_effect_per_5cmol, report='16.91%',
                 ci_low=-6.6515, ci_high=40.4720, alt_clean=17.4117,
                 note='95% CI基于原报告口径（未含截距不确定性，等同图S11/S12的CI算法）'),
            dict(key='SOM_effect_per_gkg', name='SOM边际效应', basis='每增加1 g/kg → BCF比例变化',
                 old=-0.0078, new=self.SOM_effect_per_gkg, report='-20.38% (每10 g/kg ≈ 1%)',
                 ci_low=-33.8176, ci_high=-6.9390, alt_clean=-43.1418,
                 note='v04参数名为SOM_effect_per_1pct但按“每1 g/kg”应用；本版统一为每g/kg口径并更名'),
            dict(key='marginal_veg_cd_per_01mg', name='蔬菜Cd边际效应',
                 basis='每增加0.1 mg/kg → THQ变化（绝对值）',
                 old=0.92, new=self.marginal_veg_cd_per_01mg, report='+0.918',
                 ci_low=0.9125, ci_high=0.9226, alt_clean=1.1321,
                 note='标准化系数换算值；THQ计算中不重复应用（仅作机制说明）'),
            dict(key='marginal_consumption_per_10kg', name='消费量边际效应',
                 basis='每增加10 kg/年 → THQ变化（绝对值）',
                 old=0.455, new=self.marginal_consumption_per_10kg, report='+0.231',
                 ci_low=0.0064, ci_high=0.4558, alt_clean=0.0388,
                 note='标准化系数换算值；THQ计算中不重复应用（仅作机制说明）'),
        ]

        policy_rows = []
        for name, d in [('BAU', self.policy_BAU), ('RP', self.policy_RP)]:
            for k, v in d.items():
                policy_rows.append((name, k, v))

        const_rows = [
            ('soil_cd_natural_decay', '土壤Cd自然衰减率', '1/year', self.soil_cd_natural_decay),
            ('pH_natural_buffering', 'pH自然缓冲速率', '1/year', self.pH_natural_buffering),
            ('SOM_decomposition', 'SOM分解速率', '1/year', self.SOM_decomposition),
            ('BCF_natural_decay', 'BCF自然衰减速率', '1/year', self.BCF_natural_decay),
            ('target_pH', '目标pH', '-', self.target_pH),
            ('background_THQ', '背景THQ（非蔬菜源）', '-', self.background_THQ),
            ('atmospheric_deposition', '大气沉降速率', 'mg/kg/year', 0.015),
            ('policy_efficiency_halflife', '政策效率半衰期', 'year', self.policy_efficiency_halflife),
            ('min_policy_efficiency', '最低政策效率', '-', self.min_policy_efficiency),
        ]

        readme = pd.DataFrame({
            '项目': [
                '脚本版本', '生成日期', '基准参数来源', 'CB-SEM路径系数来源', '边际效应来源',
                'SEM拟合样本', '新数据库样本量', '单位说明1', '单位说明2', '共线性说明', '备选参数',
            ],
            '说明': [
                'p24_baseON_SD_predict_2025-2035_05.py（v04的参数更新版）',
                '2025-09-24',
                src_db,
                src_sem,
                src_me,
                'N=111（平均人群数据，2004-2025，含2022-2025补充数据）',
                'n=2446（Field Dryland 数据库，已剔除土壤Cd>10 mg/kg）',
                'SOM单位为 g/kg；SOM边际效应按“每1 g/kg”口径应用，1% SOM ≈ 10 g/kg',
                'pH_effect_per_unit为“每上升1个pH单位BCF的比例变化”；'
                'CEC_effect_per_5cmol为“每增加5 cmol/kg BCF的比例变化”',
                '土壤Cd与SCC在85%记录上取值相同（VIF>300），二者SEM系数不可单独解读；'
                'SD模型中采用两者之和作为土壤Cd库总效应',
                '"边际效应_清洗数据备选"列为使用Cd>10已去除数据的平均人群子集(n=99)换算的结果，'
                '若图S11/S12改用清洗后数据重绘，应采用该列数值',
            ]
        })

        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            readme.to_excel(writer, sheet_name='00_说明', index=False)
            self._format_sheet(writer.sheets['00_说明'], wrap_cols=('项目', '说明'))

            baseline_df = pd.DataFrame([
                {'参数名': k, '含义': name, '单位': unit, '统计口径': how,
                 'v05取值': getattr(self, k), 'v04取值': old}
                for k, name, unit, how, old in baseline_def
            ])
            self._round_floats(baseline_df).to_excel(writer, sheet_name='01_基准参数', index=False)
            self._format_sheet(writer.sheets['01_基准参数'])

            sem_df = pd.DataFrame([
                {'参数名': k, '路径': path, 'v05标准化系数': beta_new[k], 'v04取值': old, '说明': note}
                for k, path, old, note in beta_def
            ])
            self._round_floats(sem_df).to_excel(writer, sheet_name='02_CBSEM路径系数', index=False)
            self._format_sheet(writer.sheets['02_CBSEM路径系数'], wrap_cols=('说明',))

            me_df = pd.DataFrame([
                {'参数名': r['key'], '含义': r['name'], '应用口径': r['basis'],
                 'v04取值': r['old'], 'v05取值': r['new'], '图上报告值': r['report'],
                 '95%CI_下限(报告口径)': r['ci_low'], '95%CI_上限(报告口径)': r['ci_high'],
                 'v05取值_清洗数据备选': r['alt_clean'], '备注': r['note']}
                for r in me_rows
            ])
            self._round_floats(me_df).to_excel(writer, sheet_name='03_边际效应', index=False)
            self._format_sheet(writer.sheets['03_边际效应'], wrap_cols=('应用口径', '图上报告值', '备注'))

            self._round_floats(pd.DataFrame(const_rows, columns=['参数名', '含义', '单位', '取值'])
                               ).to_excel(writer, sheet_name='04_模型常量', index=False)
            self._format_sheet(writer.sheets['04_模型常量'])

            self._round_floats(pd.DataFrame(policy_rows, columns=['情景', '政策参数', '取值'])
                               ).to_excel(writer, sheet_name='05_政策情景', index=False)
            self._format_sheet(writer.sheets['05_政策情景'])

            # v04 → v05 全参数对照
            compare_rows = []
            for k, name, unit, how, old in baseline_def:
                compare_rows.append(('基准参数', k, name, old, getattr(self, k)))
            for k, path, old, note in beta_def:
                compare_rows.append(('CB-SEM路径系数', k, path, old, beta_new[k]))
            for r in me_rows:
                compare_rows.append(('边际效应', r['key'], r['name'], r['old'], r['new']))
            self._round_floats(pd.DataFrame(compare_rows, columns=['参数类别', '参数名', '含义', 'v04取值', 'v05取值'])
                               ).to_excel(writer, sheet_name='06_v04_v05对照', index=False)
            self._format_sheet(writer.sheets['06_v04_v05对照'])

        print(f"\n✓ 参数表已保存至: {save_path}")


# ============================================================================
# 主程序执行
# ============================================================================

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('Agg')

    model = VegetableCdSystemDynamicsProjection()

    results_BAU, results_RP, results_combined = model.run_all_scenarios()

    model.visualize_projection(
        results_BAU,
        results_RP,
        save_path='SD_Projection_2025_2035_CBSEM_v05_NewDB_NewSEM.png'
    )

    summary_df = model.generate_summary_table(
        results_BAU,
        results_RP,
        save_path='SD_Projection_2025_2035_CBSEM_Summary_v05_NewDB_NewSEM.csv'
    )

    results_combined.to_csv('SD_Projection_2025_2035_CBSEM_Full_Data_v05_NewDB_NewSEM.csv',
                            index=False, encoding='utf-8-sig')

    model.save_parameters_to_excel('SD_Parameters_v05_NewDB_NewSEM.xlsx')

    print("\n" + "="*120)
    print("✓ 2025-2035年政策情景预测完成（CB-SEM因果机制版本 v05 - 新参数）！")
    print("="*120)
    print("\n生成文件:")
    print("  1. SD_Projection_2025_2035_CBSEM_v05_NewDB_NewSEM.png (可视化对比图)")
    print("  2. SD_Projection_2025_2035_CBSEM_v05_NewDB_NewSEM.pdf (矢量图)")
    print("  3. SD_Projection_2025_2035_CBSEM_Summary_v05_NewDB_NewSEM.csv (汇总对比表)")
    print("  4. SD_Projection_2025_2035_CBSEM_Full_Data_v05_NewDB_NewSEM.csv (完整时间序列数据)")
    print("  5. SD_Parameters_v05_NewDB_NewSEM.xlsx (模型参数表，含来源与v04/v05对照)")

    print("\n【模型科学依据总结】")
    print("="*120)
    print("1. CB-SEM标准化路径系数（β）（新版SEM, N=111）：评估相对影响强度")
    print("   - 蔬菜Cd是THQ的主导因素（β=1.001, p<0.001）")
    print("   - 土壤Cd库是BCF的源头（β=+0.497；Soil Cd -0.726 与 SCC +1.223 之和）")
    print("   - SOM对BCF有显著负向影响（β=-0.382, p=0.003）")

    print("\n2. 边际效应参数：精确定量预测（应用于上游环节）")
    print("   - pH每上升1 → BCF减少9.06% (95% CI: -17.99% ~ +36.10%)")
    print("   - CEC每增加5 cmol/kg → BCF增加16.91% (95% CI: -6.65% ~ +40.47%)")
    print("   - SOM每增加1 g/kg → BCF减少2.04% (≈每增加1% 减少20.38%)")

    print("\n3. THQ计算修正：")
    print("   ✓ 采用WHO经典公式：THQ = (Veg_Cd × Consumption) / (Body_Weight × 365 × RfD)")
    print("   ✓ 边际效应已在BCF计算中体现，不在THQ计算中重复应用")
    print("   ✓ 背景THQ保持稳定（来自非蔬菜源）")
    print("   ✓ 不同人群THQ差异符合生理学逻辑")

    print("="*120 + "\n")
