"""
全局状态管理模块
统一管理所有回调共享的数据，解决跨模块全局变量不共享的问题
"""

import pandas as pd
import numpy as np

# ============================================
# 原始数据
# ============================================
vessel_df: pd.DataFrame = None
container_df: pd.DataFrame = None
yard_df: pd.DataFrame = None

# ============================================
# 预处理结果
# ============================================
preprocessor = None
daily_throughput: pd.DataFrame = None
yard_util_time: pd.DataFrame = None

# ============================================
# 预测缓存
# ============================================
forecast_results_cache: dict = {}

# ============================================
# 翻箱率模拟缓存
# ============================================
rehandling_results_cache: pd.DataFrame = None

# ============================================
# 超期箱分析缓存
# ============================================
overdue_cache: dict = None

# ============================================
# KPI缓存
# ============================================
kpi_cache: dict = None

# ============================================
# 初始化标志
# ============================================
_initialized = False


def is_initialized():
    """检查数据是否已初始化"""
    global _initialized
    return _initialized and preprocessor is not None


def set_initialized(val: bool = True):
    """设置初始化状态"""
    global _initialized
    _initialized = val


def initialize(data_dir: str = 'data'):
    """
    初始化全局数据 - 加载或生成数据并完成预处理
    返回: (是否成功, 错误信息)
    """
    global vessel_df, container_df, yard_df
    global preprocessor, daily_throughput, yard_util_time
    global forecast_results_cache, rehandling_results_cache
    global overdue_cache, kpi_cache, _initialized

    try:
        from .data_generator import load_port_data, PortDataGenerator

        try:
            vessel_df, container_df, yard_df = load_port_data(data_dir)
        except Exception:
            gen = PortDataGenerator()
            vessel_df, container_df, yard_df = gen.generate_all(data_dir)

        from .data_processor import DataPreprocessor
        preprocessor = DataPreprocessor(vessel_df, container_df, yard_df)
        daily_throughput = preprocessor.get_daily_throughput()
        yard_util_time = preprocessor.calculate_yard_utilization_time()

        forecast_results_cache.clear()
        rehandling_results_cache = None
        overdue_cache = None
        kpi_cache = None

        _initialized = True

        return True, ''
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, str(e)


def get_summary():
    """获取数据摘要（用于调试）"""
    info = {}
    if vessel_df is not None:
        info['船舶记录数'] = len(vessel_df)
    if container_df is not None:
        info['集装箱记录数'] = len(container_df)
    if yard_df is not None:
        info['堆场区域数'] = len(yard_df)
    if daily_throughput is not None:
        info['吞吐量天数'] = len(daily_throughput)
    if yard_util_time is not None:
        info['利用率记录数'] = len(yard_util_time)
    return info
