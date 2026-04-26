"""
Plot Styles - 统一的绘图风格和工具函数
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from config import (
    DEFAULT_STYLE, AVAILABLE_STYLES, FONT_SIZES, FONT_FAMILY, 
    DPI, OUTPUT_FORMATS
)


def setup_paper_style(style_name: str = None):
    """
    设置论文级别的绘图风格
    
    Args:
        style_name: 样式名称，如'large_12px'，默认使用DEFAULT_STYLE
    """
    # 使用styles文件夹中的样式
    if style_name and style_name in AVAILABLE_STYLES:
        style_path = AVAILABLE_STYLES[style_name]
    else:
        style_path = DEFAULT_STYLE
    
    if style_path.exists():
        plt.style.use(str(style_path))
        print(f"✅ Loaded style: {style_path.name}")
    else:
        print(f"⚠️  Style file not found: {style_path}, using default matplotlib style")
        # 使用基本的论文风格设置
        plt.rcParams['font.family'] = FONT_FAMILY
        plt.rcParams['font.size'] = FONT_SIZES['label']
        plt.rcParams['axes.titlesize'] = FONT_SIZES['title']
        plt.rcParams['axes.labelsize'] = FONT_SIZES['label']
        plt.rcParams['xtick.labelsize'] = FONT_SIZES['tick']
        plt.rcParams['ytick.labelsize'] = FONT_SIZES['tick']
        plt.rcParams['legend.fontsize'] = FONT_SIZES['legend']
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.grid'] = True
        plt.rcParams['axes.axisbelow'] = True
        plt.rcParams['axes.unicode_minus'] = False


def save_figure(fig, name: str, output_dir: Path, 
                formats: List[str] = None, dpi: int = None):
    """
    统一的图表保存函数
    
    Args:
        fig: matplotlib figure对象
        name: 文件名（不含扩展名）
        output_dir: 输出目录
        formats: 输出格式列表，默认['png', 'svg']
        dpi: 分辨率，默认300
    """
    if formats is None:
        formats = OUTPUT_FORMATS
    if dpi is None:
        dpi = DPI
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        saved_files.append(filepath)
        print(f"  ✅ Saved: {filepath}")
    
    return saved_files


def add_value_labels(ax, bars, format_str: str = '.2f', 
                     offset: float = 0.01):
    """
    在柱状图上添加数值标签
    
    Args:
        ax: matplotlib axes对象
        bars: bar容器
        format_str: 数值格式
        offset: 标签偏移量（相对于y轴范围）
    """
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    offset_value = y_range * offset
    
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + offset_value,
                   f'{height:{format_str}}',
                   ha='center', va='bottom')


def add_significance_markers(ax, x_positions: List[float], 
                            y_position: float, pairs: List[Tuple],
                            p_values: List[float]):
    """
    添加统计显著性标记
    
    Args:
        ax: matplotlib axes对象
        x_positions: x轴位置列表
        y_position: 标记的y轴位置
        pairs: 比较对列表 [(idx1, idx2), ...]
        p_values: p值列表
    """
    def get_significance_marker(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'ns'
    
    for (i, j), p in zip(pairs, p_values):
        marker = get_significance_marker(p)
        if marker != 'ns':
            x1, x2 = x_positions[i], x_positions[j]
            ax.plot([x1, x1, x2, x2], 
                   [y_position, y_position + 0.02, y_position + 0.02, y_position],
                   'k-', linewidth=1)
            ax.text((x1 + x2) / 2, y_position + 0.03, marker,
                   ha='center', va='bottom')


def create_legend_outside(ax, ncol: int = 1, loc: str = 'upper left',
                         bbox_to_anchor: Tuple = (1.02, 1)):
    """
    在图表外部创建图例
    
    Args:
        ax: matplotlib axes对象
        ncol: 图例列数
        loc: 图例位置
        bbox_to_anchor: 图例锚点
    """
    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, 
             ncol=ncol, frameon=True, framealpha=0.9)


def format_axis_scientific(ax, axis: str = 'y', threshold: float = 1000):
    """
    格式化坐标轴为科学计数法
    
    Args:
        ax: matplotlib axes对象
        axis: 'x' 或 'y'
        threshold: 超过此值使用科学计数法
    """
    from matplotlib.ticker import FuncFormatter
    
    def scientific_formatter(x, pos):
        if abs(x) >= threshold:
            return f'{x:.1e}'
        else:
            return f'{x:.0f}'
    
    if axis == 'y':
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_formatter))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_formatter))


def add_grid(ax, axis: str = 'both', alpha: float = None):
    """
    添加网格
    
    Args:
        ax: matplotlib axes对象
        axis: 'x', 'y', 或 'both'
        alpha: 透明度
    """
    if alpha is None:
        alpha = GRID_ALPHA
    
    ax.grid(True, axis=axis, alpha=alpha, linestyle=GRID_LINESTYLE)


def set_axis_limits_with_margin(ax, data, axis: str = 'y', margin: float = 0.1):
    """
    设置坐标轴范围（带边距）
    
    Args:
        ax: matplotlib axes对象
        data: 数据数组
        axis: 'x' 或 'y'
        margin: 边距比例
    """
    data_min, data_max = np.min(data), np.max(data)
    data_range = data_max - data_min
    
    lower = data_min - data_range * margin
    upper = data_max + data_range * margin
    
    if axis == 'y':
        ax.set_ylim(lower, upper)
    else:
        ax.set_xlim(lower, upper)


def create_color_palette(n_colors: int, palette: str = 'husl') -> List:
    """
    创建颜色调色板
    
    Args:
        n_colors: 颜色数量
        palette: 调色板名称
    
    Returns:
        颜色列表
    """
    return sns.color_palette(palette, n_colors)


def annotate_heatmap(ax, data, format_str: str = '.2f', 
                     threshold: float = None, textcolors: Tuple = ("black", "white")):
    """
    在热力图上添加数值标注
    
    Args:
        ax: matplotlib axes对象
        data: 数据矩阵
        format_str: 数值格式
        threshold: 颜色切换阈值
        textcolors: 文字颜色元组 (低于阈值, 高于阈值)
    """
    if threshold is None:
        threshold = data.max() / 2
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = textcolors[int(value > threshold)]
            ax.text(j, i, f'{value:{format_str}}',
                   ha="center", va="center", color=text_color)


def create_twin_axis(ax, ylabel: str, color: str = 'red'):
    """
    创建双y轴
    
    Args:
        ax: 主axes对象
        ylabel: 次y轴标签
        color: 次y轴颜色
    
    Returns:
        次axes对象
    """
    ax2 = ax.twinx()
    ax2.set_ylabel(ylabel, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    return ax2


def add_watermark(fig, text: str = 'Draft', alpha: float = 0.1):
    """
    添加水印
    
    Args:
        fig: matplotlib figure对象
        text: 水印文字
        alpha: 透明度
    """
    fig.text(0.5, 0.5, text, color='gray',
            ha='center', va='center', alpha=alpha, rotation=30)


# ============================================================================
# 预设样式
# ============================================================================
def apply_ieee_style():
    """应用IEEE论文风格"""
    setup_paper_style()
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['legend.fontsize'] = 8


def apply_nature_style():
    """应用Nature论文风格"""
    setup_paper_style()
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.titlesize'] = 9


def apply_acm_style():
    """应用ACM论文风格"""
    setup_paper_style()
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
