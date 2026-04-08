"""
Matplotlib 中文字体配置
确保图表中的中文标题、标签不会显示为方块乱码。
在任何使用 matplotlib 绘图的模块中，导入此模块即可。
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def _find_cjk_font() -> str:
    """
    查找系统中可用的中文字体。
    优先级：Noto Sans CJK > SimHei > WenQuanYi > 任何含 CJK 的字体 > sans-serif 回退
    """
    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "SimHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "STHeiti",
        "Arial Unicode MS",
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for font_name in preferred:
        if font_name in available:
            return font_name

    # 尝试模糊匹配含 CJK 或中文关键字的字体
    for f in fm.fontManager.ttflist:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in ["cjk", "noto", "hei", "song", "fang", "kai"]):
            return f.name

    return "sans-serif"


def setup_chinese_font():
    """配置 matplotlib 使用中文字体，解决中文乱码问题。"""
    font_name = _find_cjk_font()

    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块

    return font_name


# 模块导入时自动配置
_configured_font = setup_chinese_font()
