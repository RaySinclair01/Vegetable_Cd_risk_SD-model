import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import numpy as np

# 设置全局字体和PDF文字可编辑
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
# 关键设置：使PDF文字可编辑
plt.rcParams['pdf.fonttype'] = 42  # TrueType字体
plt.rcParams['ps.fonttype'] = 42   # PostScript字体
plt.rcParams['svg.fonttype'] = 'none'  # SVG中保留文本

# 创建画布 - 使用16:9比例
fig, ax = plt.subplots(1, 1, figsize=(36, 20.25))
ax.set_xlim(0, 32)
ax.set_ylim(0, 18)
ax.axis('off')

# 定义专业配色方案
COLOR_POLICY = '#2C5F9E'        
COLOR_ENV = '#8B4789'            
COLOR_SOIL = '#C65D3B'           
COLOR_BIO = '#D4A017'            
COLOR_EXPOSURE = '#E67E22'       
COLOR_HEALTH = '#C0392B'         
COLOR_FEEDBACK = '#27AE60'       
COLOR_SOCIAL = '#8E44AD'         

# ============================================================================
# 优化后的辅助绘图函数
# ============================================================================

def draw_rounded_box(ax, x, y, width, height, text, color, alpha=0.85, fontsize=20, textcolor='white', zorder=10):
    """绘制圆角矩形 - 自动调整大小"""
    # 增大尺寸以适应更大字体
    width = width * 1.15  # 宽度增加15%
    height = height * 1.2  # 高度增加20%
    
    box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                         boxstyle="round,pad=0.1", 
                         facecolor=color, edgecolor='white',
                         alpha=alpha, linewidth=3, zorder=zorder)
    ax.add_patch(box)
    
    txt = ax.text(x, y, text, ha='center', va='center', 
                  fontsize=fontsize, color=textcolor, weight='bold', 
                  family='serif', wrap=True, zorder=zorder+1)
    txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white', alpha=0.3)])
    
    return width, height  # 返回实际尺寸

def draw_ellipse_node(ax, x, y, width, height, text, color, alpha=0.85, fontsize=20, zorder=10):
    """绘制椭圆节点 - 自动调整大小"""
    # 增大尺寸以适应更大字体
    width = width * 1.2  # 宽度增加20%
    height = height * 1.25  # 高度增加25%
    
    ellipse = Ellipse((x, y), width, height,
                      facecolor=color, edgecolor='white',
                      alpha=alpha, linewidth=3, zorder=zorder)
    ax.add_patch(ellipse)
    
    txt = ax.text(x, y, text, ha='center', va='center',
                  fontsize=fontsize, color='white', weight='bold',
                  family='serif', zorder=zorder+1)
    txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white', alpha=0.3)])
    
    return width, height  # 返回实际尺寸

def draw_stock_box(ax, x, y, width, height, text, color, fontsize=22, zorder=10):
    """绘制存量变量框 - 自动调整大小"""
    # 增大尺寸以适应更大字体
    width = width * 1.15
    height = height * 1.2
    
    rect = Rectangle((x-width/2, y-height/2), width, height,
                     facecolor=color, edgecolor='white',
                     alpha=0.9, linewidth=3.5, zorder=zorder)
    ax.add_patch(rect)
    
    txt = ax.text(x, y, text, ha='center', va='center',
                  fontsize=fontsize, color='white', weight='bold',
                  family='serif', zorder=zorder+1)
    txt.set_path_effects([path_effects.withStroke(linewidth=1, foreground='white', alpha=0.3)])
    
    return width, height  # 返回实际尺寸

def calculate_edge_point(x_center, y_center, width, height, target_x, target_y, shape='rectangle'):
    """计算从中心到目标方向的边缘交点"""
    dx = target_x - x_center
    dy = target_y - y_center
    
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return x_center, y_center
    
    if shape == 'ellipse':
        # 椭圆边缘计算
        angle = np.arctan2(dy, dx)
        edge_x = x_center + (width / 2) * np.cos(angle)
        edge_y = y_center + (height / 2) * np.sin(angle)
        return edge_x, edge_y
    
    else:  # rectangle or rounded_box
        # 矩形边缘计算
        half_w = width / 2
        half_h = height / 2
        
        # 计算射线与四条边的交点
        if abs(dx) > 0.01:
            # 与左右边的交点
            t_x = half_w / abs(dx)
            y_at_edge = dy * t_x
            if abs(y_at_edge) <= half_h:
                edge_x = x_center + half_w if dx > 0 else x_center - half_w
                edge_y = y_center + y_at_edge
                return edge_x, edge_y
        
        # 与上下边的交点
        if abs(dy) > 0.01:
            t_y = half_h / abs(dy)
            x_at_edge = dx * t_y
            edge_x = x_center + x_at_edge
            edge_y = y_center + half_h if dy > 0 else y_center - half_h
            return edge_x, edge_y
        
        return x_center, y_center

def create_smooth_curve_path(x1, y1, x2, y2, curvature=0.3, direction='auto'):
    """创建平滑贝塞尔曲线路径"""
    dx = x2 - x1
    dy = y2 - y1
    distance = np.sqrt(dx**2 + dy**2)
    
    if direction == 'auto':
        if abs(dx) > abs(dy):
            direction = 'horizontal'
        else:
            direction = 'vertical'
    
    if direction == 'horizontal':
        ctrl1_x = x1 + dx * 0.5
        ctrl1_y = y1
        ctrl2_x = x1 + dx * 0.5
        ctrl2_y = y2
    elif direction == 'vertical':
        ctrl1_x = x1
        ctrl1_y = y1 + dy * 0.5
        ctrl2_x = x2
        ctrl2_y = y1 + dy * 0.5
    elif direction == 'arc_left':
        ctrl1_x = x1 - distance * curvature * 0.3
        ctrl1_y = y1 + dy * 0.3
        ctrl2_x = x2 - distance * curvature * 0.3
        ctrl2_y = y2 - dy * 0.3
    elif direction == 'arc_right':
        ctrl1_x = x1 + distance * curvature * 0.3
        ctrl1_y = y1 + dy * 0.3
        ctrl2_x = x2 + distance * curvature * 0.3
        ctrl2_y = y2 - dy * 0.3
    else:
        ctrl1_x = x1 + dx * curvature
        ctrl1_y = y1 + dy * (1 - curvature)
        ctrl2_x = x2 - dx * curvature
        ctrl2_y = y2 - dy * (1 - curvature)
    
    return (ctrl1_x, ctrl1_y), (ctrl2_x, ctrl2_y)

def draw_bezier_arrow(ax, x1, y1, x2, y2, color, label='', linewidth=3,
                     curvature=0.3, direction='auto', style='->', 
                     labelsize=16, labelcolor='black', label_offset=(0, 0), zorder=5,
                     start_shape='rectangle', start_size=(3, 1), 
                     end_shape='rectangle', end_size=(3, 1)):
    """
    使用贝塞尔曲线绘制平滑箭头 - 箭头连接到边缘
    
    参数:
    - start_shape: 起点形状 ('rectangle', 'ellipse', 'rounded_box')
    - start_size: 起点尺寸 (width, height)
    - end_shape: 终点形状
    - end_size: 终点尺寸
    """
    # 计算起点边缘
    edge_x1, edge_y1 = calculate_edge_point(x1, y1, start_size[0], start_size[1], x2, y2, start_shape)
    
    # 计算终点边缘
    edge_x2, edge_y2 = calculate_edge_point(x2, y2, end_size[0], end_size[1], x1, y1, end_shape)
    
    distance = np.sqrt((edge_x2-edge_x1)**2 + (edge_y2-edge_y1)**2)
    
    if distance < 2.5:
        connectionstyle = "arc3,rad=0.15"
        arrow = FancyArrowPatch((edge_x1, edge_y1), (edge_x2, edge_y2),
                               arrowstyle=style, color=color,
                               linewidth=linewidth, alpha=0.8,
                               connectionstyle=connectionstyle,
                               mutation_scale=25, zorder=zorder)
        ax.add_patch(arrow)
    else:
        ctrl1, ctrl2 = create_smooth_curve_path(edge_x1, edge_y1, edge_x2, edge_y2, curvature, direction)
        
        verts = [(edge_x1, edge_y1), ctrl1, ctrl2, (edge_x2, edge_y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color,
                                   linewidth=linewidth, alpha=0.8, zorder=zorder)
        ax.add_patch(patch)
        
        if '->' in style:
            # 箭头方向从倒数第二个控制点指向终点
            arrow = FancyArrowPatch((edge_x2-0.15*(edge_x2-ctrl2[0]), edge_y2-0.15*(edge_y2-ctrl2[1])), 
                                   (edge_x2, edge_y2),
                                   arrowstyle='->', color=color, linewidth=linewidth,
                                   alpha=0.8, zorder=zorder, mutation_scale=25)
            ax.add_patch(arrow)
    
    if label:
        # 标签位置在曲线中点
        if distance >= 2.5:
            t = 0.5
            mid_x = (1-t)**3*edge_x1 + 3*(1-t)**2*t*ctrl1[0] + 3*(1-t)*t**2*ctrl2[0] + t**3*edge_x2
            mid_y = (1-t)**3*edge_y1 + 3*(1-t)**2*t*ctrl1[1] + 3*(1-t)*t**2*ctrl2[1] + t**3*edge_y2
        else:
            mid_x = (edge_x1 + edge_x2) / 2
            mid_y = (edge_y1 + edge_y2) / 2
        
        mid_x += label_offset[0]
        mid_y += label_offset[1]
        
        ax.text(mid_x, mid_y, label, fontsize=labelsize,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=color, alpha=0.92, linewidth=2),
               ha='center', va='center', color=labelcolor, weight='bold',
               family='serif', zorder=zorder+3)

def draw_feedback_loop(ax, points, color, label='', linewidth=2.5, labelsize=18, zorder=3):
    """绘制平滑的反馈环路"""
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        
        ctrl1, ctrl2 = create_smooth_curve_path(x1, y1, x2, y2, curvature=0.2, direction='auto')
        
        verts = [(x1, y1), ctrl1, ctrl2, (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        path = Path(verts, codes)
        
        patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color,
                                   linewidth=linewidth, alpha=0.45, 
                                   linestyle='--', zorder=zorder)
        ax.add_patch(patch)
        
        if i == len(points) - 2:
            arrow = FancyArrowPatch((x2-0.2*(x2-ctrl2[0]), y2-0.2*(y2-ctrl2[1])), (x2, y2),
                                   arrowstyle='->', color=color, linewidth=linewidth,
                                   alpha=0.45, zorder=zorder, mutation_scale=20)
            ax.add_patch(arrow)
    
    if label:
        center_x = np.mean([p[0] for p in points])
        center_y = np.mean([p[1] for p in points])
        ax.text(center_x, center_y, label, fontsize=labelsize,
                bbox=dict(boxstyle='round,pad=0.6', facecolor=color,
                         edgecolor='white', alpha=0.2),
                ha='center', color=color, weight='bold',
                style='italic', family='serif', zorder=zorder+1)

# ============================================================================
# 存储所有节点的位置和尺寸信息（用于箭头连接）
# ============================================================================
nodes = {}

# ============================================================================
# 绘制所有层级
# ============================================================================

y_policy = 16.8
ax.text(16, 17.6, 'System Dynamics Model of Vegetable Cadmium Pollution:\nPolicy-Environment-Health Integrated Framework', 
        ha='center', fontsize=28, weight='bold', family='serif',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                 edgecolor='black', alpha=0.9, linewidth=3.5))

# 政策层
w, h = draw_rounded_box(ax, 3.5, y_policy, 3.0, 1.1, 
                'Soil Remediation\nPolicy\n(0-100%)', 
                COLOR_POLICY, fontsize=19, zorder=10)
nodes['policy1'] = {'pos': (3.5, y_policy), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 8, y_policy, 3.0, 1.1, 
                'pH Amendment\nProgram\n(Target: 6.5-7.5)', 
                COLOR_POLICY, fontsize=19, zorder=10)
nodes['policy2'] = {'pos': (8, y_policy), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 12.5, y_policy, 3.0, 1.1, 
                'Organic Matter\nEnhancement\n(+2 g/kg/year)', 
                COLOR_POLICY, fontsize=19, zorder=10)
nodes['policy3'] = {'pos': (12.5, y_policy), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 17, y_policy, 3.0, 1.1, 
                'Planting Structure\nAdjustment\n(Low-BCF crops)', 
                COLOR_POLICY, fontsize=19, zorder=10)
nodes['policy4'] = {'pos': (17, y_policy), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 21.5, y_policy, 3.0, 1.1, 
                'Dietary Guidance\n& Education\n(-15% consumption)', 
                COLOR_POLICY, fontsize=19, zorder=10)
nodes['policy5'] = {'pos': (21.5, y_policy), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 26, y_policy, 3.0, 1.1, 
                'Market Access\nRegulation\n(Cd limit: 0.2 mg/kg)', 
                COLOR_POLICY, fontsize=19, zorder=10)
nodes['policy6'] = {'pos': (26, y_policy), 'size': (w, h), 'shape': 'rounded_box'}

# 环境层
y_env = 14.2
ax.text(5, 15.0, 'Environmental System', ha='center', 
        fontsize=21, weight='bold', color=COLOR_ENV, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_ENV, alpha=0.2))

w, h = draw_ellipse_node(ax, 2.5, y_env, 2.5, 1.0, 
                 'Climate Zone\n(8 types)\nβ=0.714***', 
                 COLOR_ENV, fontsize=17, zorder=10)
nodes['env1'] = {'pos': (2.5, y_env), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 6, y_env, 2.5, 1.0, 
                 'Geographic\nRegion\n(5 regions)\nβ=0.140***', 
                 COLOR_ENV, fontsize=17, zorder=10)
nodes['env2'] = {'pos': (6, y_env), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 9.5, y_env, 2.5, 1.0, 
                 'Province\n(30 provinces)\nβ=0.122**', 
                 COLOR_ENV, fontsize=17, zorder=10)
nodes['env3'] = {'pos': (9.5, y_env), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_rounded_box(ax, 13, y_env, 2.3, 0.9, 
                'Season\nSpring BCF: 0.109\nSummer: 0.215', 
                COLOR_ENV, alpha=0.8, fontsize=17, zorder=10)
nodes['env4'] = {'pos': (13, y_env), 'size': (w, h), 'shape': 'rounded_box'}

# 土壤层
y_soil = 11.5
ax.text(16, 12.5, 'Soil Contamination & Properties System', ha='center', 
        fontsize=21, weight='bold', color=COLOR_SOIL, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_SOIL, alpha=0.2))

w, h = draw_stock_box(ax, 4.5, y_soil, 3.2, 1.3, 
              'Soil Cd Content\n0.007-193.5 mg/kg\nMean: 2.04±6.23\nDecay: -2%/year', 
              COLOR_SOIL, fontsize=19, zorder=10)
nodes['soil_cd'] = {'pos': (4.5, y_soil), 'size': (w, h), 'shape': 'rectangle'}

w, h = draw_stock_box(ax, 11, y_soil, 2.8, 1.1, 
              'Soil pH\n4.14-8.91\nMean: 6.60±1.22\nβ to BCF: -0.346***', 
              COLOR_SOIL, fontsize=17, zorder=10)
nodes['soil_ph'] = {'pos': (11, y_soil), 'size': (w, h), 'shape': 'rectangle'}

w, h = draw_stock_box(ax, 16, y_soil, 2.8, 1.1, 
              'Soil Organic Matter\n0.14-210.7 g/kg\nMean: 27.7±22.0\nr to BCF: -0.21*', 
              COLOR_SOIL, fontsize=17, zorder=10)
nodes['soil_som'] = {'pos': (16, y_soil), 'size': (w, h), 'shape': 'rectangle'}

w, h = draw_stock_box(ax, 21, y_soil, 2.8, 1.1, 
              'CEC\n3.6-37.0 cmol/kg\nMean: 15.3±5.8\nβ to BCF: +0.222**', 
              COLOR_SOIL, fontsize=17, zorder=10)
nodes['soil_cec'] = {'pos': (21, y_soil), 'size': (w, h), 'shape': 'rectangle'}

w, h = draw_ellipse_node(ax, 25.5, y_soil, 2.8, 1.0, 
                 'Soil Type\n(14 types)\nRed Soil: 21.2%\nLoess: BCF up 1.8x', 
                 COLOR_SOIL, alpha=0.8, fontsize=17, zorder=10)
nodes['soil_type'] = {'pos': (25.5, y_soil), 'size': (w, h), 'shape': 'ellipse'}

# 生物富集层
y_bio = 8.5
ax.text(16, 9.5, 'Bioaccumulation System', ha='center', 
        fontsize=21, weight='bold', color=COLOR_BIO, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_BIO, alpha=0.2))

w, h = draw_stock_box(ax, 8, y_bio, 3.8, 1.3, 
              'Bioconcentration\nFactor (BCF)\nβ(Soil Cd): 0.595***\nFlow contribution: 31%', 
              COLOR_BIO, fontsize=19, zorder=10)
nodes['bcf'] = {'pos': (8, y_bio), 'size': (w, h), 'shape': 'rectangle'}

w, h = draw_ellipse_node(ax, 15, y_bio, 2.8, 1.1, 
                 'Leafy Vegetables\n51.87%\nBCF: 0.123±0.011\nAmaranth, Cabbage', 
                 COLOR_BIO, fontsize=17, zorder=10)
nodes['veg_leafy'] = {'pos': (15, y_bio), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 19.5, y_bio, 2.8, 1.1, 
                 'Root Vegetables\n24.61%\nBCF: 0.096±0.009\nLotus, Potato', 
                 COLOR_BIO, fontsize=17, zorder=10)
nodes['veg_root'] = {'pos': (19.5, y_bio), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 24, y_bio, 2.8, 1.1, 
                 'Fruit Vegetables\n23.52%\nBCF: 0.054±0.007\nTomato, Pepper', 
                 COLOR_BIO, fontsize=17, zorder=10)
nodes['veg_fruit'] = {'pos': (24, y_bio), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_stock_box(ax, 15, 6.2, 4.2, 1.4, 
              'Vegetable Cd Content\n0.0002-57.79 mg/kg\nMean: 0.16±1.20\nβ to THQ: 0.999***\nSHAP: 1.31-4.40', 
              COLOR_BIO, fontsize=19, zorder=10)
nodes['veg_cd'] = {'pos': (15, 6.2), 'size': (w, h), 'shape': 'rectangle'}

# 暴露层
y_exp = 4.2
ax.text(5, 5.2, 'Exposure System', ha='center', 
        fontsize=21, weight='bold', color=COLOR_EXPOSURE, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_EXPOSURE, alpha=0.2))

w, h = draw_rounded_box(ax, 3, y_exp, 2.5, 1.0, 
                'Urban Consumption\n106.6±12.9 kg/year\nβ to THQ: 0.016***', 
                COLOR_EXPOSURE, fontsize=17, zorder=10)
nodes['consump_urban'] = {'pos': (3, y_exp), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 6.5, y_exp, 2.5, 1.0, 
                'Rural Consumption\n98.2±15.7 kg/year\nElasticity: 0.27-0.64', 
                COLOR_EXPOSURE, fontsize=17, zorder=10)
nodes['consump_rural'] = {'pos': (6.5, y_exp), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_ellipse_node(ax, 10, y_exp, 2.2, 0.9, 
                 'Self-Production\nUrban: 90.9%\nRural: 93.3%', 
                 COLOR_EXPOSURE, alpha=0.8, fontsize=16, zorder=10)
nodes['self_prod'] = {'pos': (10, y_exp), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_rounded_box(ax, 13.5, y_exp, 2.2, 1.0, 
                'Body Weight\nMale: 63-66 kg\nFemale: 55-57 kg\nNegative effect', 
                COLOR_EXPOSURE, fontsize=16, zorder=10)
nodes['body_weight'] = {'pos': (13.5, y_exp), 'size': (w, h), 'shape': 'rounded_box'}

# 人群层
y_pop = 2.5
ax.text(23, 3.5, 'Population Groups', ha='center', 
        fontsize=21, weight='bold', color=COLOR_HEALTH, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_HEALTH, alpha=0.2))

w, h = draw_ellipse_node(ax, 18.5, y_pop, 2.2, 0.9, 
                 'Urban Male\nWeight: 66.4 kg\nBaseline THQ: 1.59', 
                 COLOR_HEALTH, fontsize=17, zorder=10)
nodes['pop_um'] = {'pos': (18.5, y_pop), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 21.5, y_pop, 2.2, 0.9, 
                 'Urban Female\nWeight: 57.1 kg\nBaseline THQ: 1.85', 
                 COLOR_HEALTH, fontsize=17, zorder=10)
nodes['pop_uf'] = {'pos': (21.5, y_pop), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 24.5, y_pop, 2.2, 0.9, 
                 'Rural Male\nWeight: 63.0 kg\nBaseline THQ: 1.61', 
                 COLOR_HEALTH, fontsize=17, zorder=10)
nodes['pop_rm'] = {'pos': (24.5, y_pop), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 27.5, y_pop, 2.2, 0.9, 
                 'Rural Female\nWeight: 55.4 kg\nBaseline THQ: 1.83', 
                 COLOR_HEALTH, fontsize=17, zorder=10)
nodes['pop_rf'] = {'pos': (27.5, y_pop), 'size': (w, h), 'shape': 'ellipse'}

# 健康风险层
y_health = 0.8
ax.text(16, 1.8, 'Health Risk Assessment System', ha='center', 
        fontsize=21, weight='bold', color=COLOR_HEALTH, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_HEALTH, alpha=0.2))

w, h = draw_stock_box(ax, 11.5, y_health, 5.5, 1.3, 
              'Target Hazard Quotient (THQ)\nIntegrated ML Model (AUC=0.99, ACC=0.97)\nMarginal Effect: +0.1 mg/kg Veg Cd → +0.91-0.93 THQ', 
              COLOR_HEALTH, fontsize=19, zorder=10)
nodes['thq'] = {'pos': (11.5, y_health), 'size': (w, h), 'shape': 'rectangle'}

w, h = draw_ellipse_node(ax, 19.5, y_health, 2.2, 0.9, 
                 'No Risk\n0<THQ≤0.5\nSafe', 
                 COLOR_FEEDBACK, fontsize=17, zorder=10)
nodes['risk_no'] = {'pos': (19.5, y_health), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 22.5, y_health, 2.2, 0.9, 
                 'Low Risk\n0.5<THQ≤1.0\nAcceptable', 
                 '#F39C12', fontsize=17, zorder=10)
nodes['risk_low'] = {'pos': (22.5, y_health), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 25.5, y_health, 2.2, 0.9, 
                 'Medium Risk\n1<THQ≤2.0\nIntervention', 
                 '#E67E22', fontsize=17, zorder=10)
nodes['risk_med'] = {'pos': (25.5, y_health), 'size': (w, h), 'shape': 'ellipse'}

w, h = draw_ellipse_node(ax, 28.5, y_health, 2.2, 0.9, 
                 'High Risk\nTHQ>2.0\nUrgent Action', 
                 '#C0392B', fontsize=17, zorder=10)
nodes['risk_high'] = {'pos': (28.5, y_health), 'size': (w, h), 'shape': 'ellipse'}

# 社会经济反馈
y_social = 8.5
ax.text(29.5, y_social+1.2, 'Socioeconomic\nFeedback', ha='center', 
        fontsize=19, weight='bold', color=COLOR_SOCIAL, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_SOCIAL, alpha=0.2))

w, h = draw_rounded_box(ax, 29.5, y_social-0.3, 2.5, 0.9, 
                'Theil Index\nWithin-group: 98%\nBetween: 2%', 
                COLOR_SOCIAL, fontsize=17, zorder=10)
nodes['social1'] = {'pos': (29.5, y_social-0.3), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 29.5, y_social-1.6, 2.5, 0.9, 
                'Gender Disparity\nFemale THQ up 1.8%\nRural Female: High', 
                COLOR_SOCIAL, fontsize=17, zorder=10)
nodes['social2'] = {'pos': (29.5, y_social-1.6), 'size': (w, h), 'shape': 'rounded_box'}

w, h = draw_rounded_box(ax, 29.5, y_social-2.9, 2.5, 0.9, 
                'Regional Inequality\nCentral up 41.3%\nvs East China', 
                COLOR_SOCIAL, fontsize=17, zorder=10)
nodes['social3'] = {'pos': (29.5, y_social-2.9), 'size': (w, h), 'shape': 'rounded_box'}

# ============================================================================
# 主要箭头连接 - 使用边缘连接
# ============================================================================

def get_node_info(node_key):
    """获取节点信息"""
    if node_key in nodes:
        return nodes[node_key]['pos'][0], nodes[node_key]['pos'][1], \
               nodes[node_key]['size'][0], nodes[node_key]['size'][1], \
               nodes[node_key]['shape']
    return None, None, 3, 1, 'rectangle'

# 政策→土壤
x1, y1, w1, h1, s1 = get_node_info('policy1')
x2, y2, w2, h2, s2 = get_node_info('soil_cd')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_POLICY, 
                 'Remediation\n-5%/yr', linewidth=4, 
                 curvature=0.35, direction='vertical', zorder=4, label_offset=(-0.8, 0), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('policy2')
x2, y2, w2, h2, s2 = get_node_info('soil_ph')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_POLICY, 
                 'pH+0.3/yr', linewidth=4, 
                 curvature=0.3, direction='auto', zorder=4, label_offset=(0, 0.4), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('policy3')
x2, y2, w2, h2, s2 = get_node_info('soil_som')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_POLICY, 
                 'SOM+2g/kg', linewidth=4, 
                 curvature=0.25, direction='horizontal', zorder=4, label_offset=(0, 0.4), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('policy4')
x2, y2, w2, h2, s2 = get_node_info('veg_leafy')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_POLICY, 
                 'BCF-30%', linewidth=3.5, 
                 curvature=0.2, direction='vertical', zorder=4, label_offset=(0.8, 0), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('policy5')
x2, y2, w2, h2, s2 = get_node_info('consump_rural')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_POLICY, 
                 'Consump-15%', linewidth=3.5, 
                 curvature=0.25, direction='arc_left', zorder=4, label_offset=(-1.5, 0), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('policy6')
x2, y2, w2, h2, s2 = get_node_info('veg_cd')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_POLICY, 
                 'Veg Cd-20%', linewidth=3.5, 
                 curvature=0.2, direction='vertical', zorder=4, label_offset=(1.5, 0), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 环境→土壤
x1, y1, w1, h1, s1 = get_node_info('env1')
x2, y2, w2, h2, s2 = get_node_info('soil_cd')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_ENV, 
                 'β=0.714***', linewidth=3.5, 
                 curvature=0.25, direction='vertical', zorder=4, label_offset=(-1, 0), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('env2')
x2, y2, w2, h2, s2 = get_node_info('soil_cd')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_ENV, 
                 'β=0.140***', linewidth=3, 
                 curvature=0.2, direction='vertical', zorder=4, label_offset=(-0.8, 0.3), labelsize=16,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 土壤Cd→BCF
x1, y1, w1, h1, s1 = get_node_info('soil_cd')
x2, y2, w2, h2, s2 = get_node_info('bcf')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_SOIL, 
                 'β=0.595***\nFlow:31%', linewidth=5, 
                 curvature=0.2, direction='vertical', zorder=5, labelsize=17, label_offset=(-1, 0),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 土壤性质→BCF
x1, y1, w1, h1, s1 = get_node_info('soil_ph')
x2, y2, w2, h2, s2 = get_node_info('bcf')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_SOIL, 
                 'pH: β=-0.346***\nΔpH+1→BCF-21%', linewidth=4, 
                 curvature=0.25, direction='arc_left', zorder=4, labelsize=15, label_offset=(-1.5, -0.4),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('soil_som')
x2, y2, w2, h2, s2 = get_node_info('bcf')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_SOIL, 
                 'SOM: r=-0.21\nΔ1%→BCF-0.78%', linewidth=3, 
                 curvature=0.3, direction='arc_left', zorder=4, labelsize=15, label_offset=(-1.8, 0),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# BCF→蔬菜Cd
x1, y1, w1, h1, s1 = get_node_info('bcf')
x2, y2, w2, h2, s2 = get_node_info('veg_cd')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_BIO, 
                 'β=0.247***\nFlow:10%', linewidth=5, 
                 curvature=0.15, direction='horizontal', zorder=5, labelsize=17, label_offset=(0, -0.6),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 蔬菜类型→蔬菜Cd
x1, y1, w1, h1, s1 = get_node_info('veg_leafy')
x2, y2, w2, h2, s2 = get_node_info('veg_cd')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_BIO, 
                 'Leafy\n0.123', linewidth=3.5, 
                 curvature=0.1, direction='vertical', zorder=4, labelsize=15, label_offset=(-0.8, 0),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 蔬菜Cd→THQ
x1, y1, w1, h1, s1 = get_node_info('veg_cd')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_BIO, 
                 'β=0.999***\nSHAP:1.31-4.40\nΔ0.1mg→Δ0.91', 
                 linewidth=6, 
                 curvature=0.2, direction='vertical', zorder=6, labelsize=18, label_offset=(-2, 0),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 暴露参数→THQ
x1, y1, w1, h1, s1 = get_node_info('consump_urban')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_EXPOSURE, 
                 'Urban β=0.016***', linewidth=3.5, 
                 curvature=0.25, direction='horizontal', zorder=4, labelsize=16, label_offset=(0, -0.6),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('consump_rural')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_EXPOSURE, 
                 'Rural Flow:12.4%\nElast:0.27-0.64', 
                 linewidth=4, 
                 curvature=0.2, direction='horizontal', zorder=4, labelsize=16, label_offset=(0, 1),
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# 人群→THQ
x1, y1, w1, h1, s1 = get_node_info('pop_um')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3, 
                 curvature=0.2, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('pop_uf')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3, 
                 curvature=0.18, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('pop_rm')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3, 
                 curvature=0.2, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('pop_rf')
x2, y2, w2, h2, s2 = get_node_info('thq')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3, 
                 curvature=0.22, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# THQ→风险等级
x1, y1, w1, h1, s1 = get_node_info('thq')
x2, y2, w2, h2, s2 = get_node_info('risk_no')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3.5, curvature=0.08, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('thq')
x2, y2, w2, h2, s2 = get_node_info('risk_low')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3.5, curvature=0.06, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('thq')
x2, y2, w2, h2, s2 = get_node_info('risk_med')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3.5, curvature=0.08, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

x1, y1, w1, h1, s1 = get_node_info('thq')
x2, y2, w2, h2, s2 = get_node_info('risk_high')
draw_bezier_arrow(ax, x1, y1, x2, y2, COLOR_HEALTH, 
                 '', linewidth=3.5, curvature=0.1, direction='horizontal', zorder=4,
                 start_shape=s1, start_size=(w1, h1), end_shape=s2, end_size=(w2, h2))

# ============================================================================
# 反馈环路
# ============================================================================

feedback_loop1 = [
    (29, 1.2), (30.5, 5), (30.5, 10), (30.5, 15),
    (22, 17), (17, 17), (15, 12), (15, 9),
    (15, 6.2), (12.5, 1.5)
]
draw_feedback_loop(ax, feedback_loop1, COLOR_FEEDBACK, 
                  'Reinforcing Loop R1\nPolicy Response', linewidth=3.5, labelsize=18, zorder=2)

feedback_loop2 = [
    (25, y_health+0.2), (28, 3), (28, 7), (28, 10),
    (22, 17), (17, 13), (15, 7.5), (13.5, 1.8)
]
draw_feedback_loop(ax, feedback_loop2, COLOR_SOCIAL, 
                  'Social Feedback Loop\nInequality Response', linewidth=3, labelsize=18, zorder=2)

# ============================================================================
# 信息框
# ============================================================================

sem_text = """CB-SEM Paths
——————————————
Climate→Soil: 0.714***
pH→BCF: -0.346***
Soil Cd→BCF: 0.595***
BCF→Veg Cd: 0.247***
Veg Cd→THQ: 0.999***
Consump→THQ: 0.016***
——————————————
***p<0.001, **p<0.01"""

ax.text(1.2, 10, sem_text, fontsize=14, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', 
                 edgecolor=COLOR_SOIL, alpha=0.95, linewidth=2.5),
        verticalalignment='top', weight='bold', zorder=15)

shap_text = """SHAP Importance
—————————————
Veg Cd: 1.31-4.40
Consumption: 0.09-1.38
Soil Cd: 0.50-1.20
BCF: 0.30-0.90
pH: 0.15-0.45
Weight: 0.10-0.35
—————————————"""

ax.text(1.2, 6.5, shap_text, fontsize=14, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF9E7',
                 edgecolor=COLOR_BIO, alpha=0.95, linewidth=2.5),
        verticalalignment='top', weight='bold', zorder=15)

marginal_text = """Marginal Effects
—————————————
Veg Cd +0.1 mg/kg
  → THQ +0.91-0.93

Consump +10 kg/year
  → THQ +0.27-0.64

pH +1 unit
  → BCF -21 to -52%
—————————————"""

ax.text(1.2, 3.2, marginal_text, fontsize=14, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5',
                 edgecolor=COLOR_EXPOSURE, alpha=0.95, linewidth=2.5),
        verticalalignment='top', weight='bold', zorder=15)

flow_text = """Flow Contribution
————————————————
Soil Cd→BCF→Veg Cd: 31%
  (Main pathway)

Soil Properties→BCF: 10.1%
  (pH, SOM, CEC)

Rural Consumption: 12.4%
Urban Consumption: 6.7%
————————————————"""

ax.text(31, 14.2, flow_text, fontsize=14, family='serif',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDF2E9',
                 edgecolor=COLOR_BIO, alpha=0.95, linewidth=2.5),
        verticalalignment='top', weight='bold', zorder=15)

# ============================================================================
# 图例和数据库标注
# ============================================================================

legend_elements = [
    mpatches.Patch(facecolor=COLOR_POLICY, label='Policy Layer', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_ENV, label='Environmental Layer', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_SOIL, label='Soil System', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_BIO, label='Bioaccumulation', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_EXPOSURE, label='Exposure System', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_HEALTH, label='Health Risk', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_FEEDBACK, label='Feedback Loop', alpha=0.8),
    mpatches.Patch(facecolor=COLOR_SOCIAL, label='Socioeconomic', alpha=0.8),
]

legend = ax.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, -0.02), ncol=8, fontsize=17,
                  framealpha=0.95, edgecolor='black', fancybox=True,
                  prop={'family': 'serif', 'weight': 'bold'})

ax.text(16, 0.08,
        'Data Source: CVCCD Database | n=2,674 | 30 Provinces | 115 Vegetable Types | 14 Soil Types | 2004-2021\n'
        'Model Performance: AUC=0.99 | ACC=0.97 | Integrated ML: CNN+XGBoost+LightGBM+RF+SVM+GBDT',
        ha='center', fontsize=16, style='italic', color='gray', family='serif',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='white', alpha=0.92),
        weight='bold', zorder=15)

# 保存图片
plt.tight_layout()
plt.savefig('Vegetable_Cd_SD_Model_Editable.png', dpi=400, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('Vegetable_Cd_SD_Model_Editable.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('Vegetable_Cd_SD_Model_Editable.svg', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("\n" + "="*80)
print("✓ Editable-Text System Dynamics Model created!")
print("="*80)
print("\n新增配置 (PDF文字可编辑):")
print("  • plt.rcParams['pdf.fonttype'] = 42")
print("    → 使用TrueType字体（而非Type 3光栅字体）")
print("    → PDF中文字为矢量文本，可在Adobe软件中编辑")
print("\n  • plt.rcParams['ps.fonttype'] = 42")
print("    → PostScript输出同样使用TrueType")
print("    → 确保EPS格式也可编辑")
print("\n  • plt.rcParams['svg.fonttype'] = 'none'")
print("    → SVG格式保留文本对象（而非转换为路径）")
print("    → 可在Inkscape/Illustrator中直接编辑文字")
print("\n优势对比:")
print("  【之前】fonttype = 3 (默认)")
print("    - 文字转换为图形路径")
print("    - PDF中无法选中/编辑文本")
print("    - 文件更大，搜索功能失效")
print("\n  【现在】fonttype = 42")
print("    ✓ 文字保留为可编辑文本")
print("    ✓ 可在PDF编辑器中修改内容")
print("    ✓ 文字可复制、搜索")
print("    ✓ 文件更小，加载更快")
print("    ✓ 字体保持矢量特性")
print("\n文件保存:")
print("  • Vegetable_Cd_SD_Model_Editable.png (400 DPI, 位图)")
print("  • Vegetable_Cd_SD_Model_Editable.pdf (矢量+可编辑文本)")
print("  • Vegetable_Cd_SD_Model_Editable.svg (矢量+可编辑文本)")
print("\n使用建议:")
print("  1. 在Adobe Acrobat/Illustrator中打开PDF")
print("  2. 使用文本工具可直接编辑任何文字")
print("  3. 字体、大小、颜色均可修改")
print("  4. 适合发表前的最后微调")
print("="*80 + "\n")

plt.show()