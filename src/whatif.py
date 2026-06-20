"""
What-if情景分析模块
模拟吞吐量增长、新增堆场区域、缩短堆存天数对运营指标的影响
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict
from .data_processor import DataPreprocessor
from .rehandling import OverdueBoxAnalyzer, YardStackSimulator


class WhatIfScenario:
    """What-if情景分析器"""

    def __init__(self, preprocessor: DataPreprocessor):
        self.preprocessor = preprocessor
        self.daily_util = preprocessor.calculate_yard_utilization_time()
        self.container_df = preprocessor.container_df
        self.yard_df = preprocessor.yard_df

    def _calculate_peak_utilization(self, util_df: pd.DataFrame = None) -> float:
        """计算堆场峰值利用率"""
        if util_df is None:
            util_df = self.daily_util
        overall = util_df.groupby('日期').agg(
            总占用=('占用箱数', 'sum'),
            总容量=('总容量', 'sum')
        ).reset_index()
        overall['利用率'] = overall['总占用'] / overall['总容量']
        return overall['利用率'].max()

    def _calculate_total_capacity(self, yard_df: pd.DataFrame = None) -> int:
        """计算堆场总容量"""
        if yard_df is None:
            yard_df = self.yard_df
        return (yard_df['贝位数量'] * yard_df['列数'] * yard_df['最大堆叠层数']).sum()

    def scenario_throughput_growth(self, growth_rate: float = 0.1) -> Dict:
        """
        情景1: 吞吐量增长对堆场利用率峰值的影响
        growth_rate: 0.1表示增长10%
        """
        original_peak = self._calculate_peak_utilization()
        original_capacity = self._calculate_total_capacity()

        new_util = self.daily_util.copy()
        new_util['占用箱数'] = (new_util['占用箱数'] * (1 + growth_rate)).round().astype(int)

        overall = new_util.groupby(['日期', '区域']).agg(
            占用箱数=('占用箱数', 'sum'),
            总容量=('总容量', 'first')
        ).reset_index()
        overall['利用率'] = overall['占用箱数'] / overall['总容量']
        overall['利用率'] = overall['利用率'].clip(upper=1.0)

        daily_overall = overall.groupby('日期').agg(
            总占用=('占用箱数', 'sum'),
            总容量=('总容量', 'sum')
        ).reset_index()
        daily_overall['利用率'] = (daily_overall['总占用'] / daily_overall['总容量']).clip(upper=1.0)
        new_peak = daily_overall['利用率'].max()

        peak_days = (daily_overall['利用率'] > 0.9).sum()

        return {
            '情景名称': f'吞吐量增长{int(growth_rate*100)}%',
            '原始峰值利用率': original_peak,
            '新峰值利用率': new_peak,
            '利用率提升幅度': new_peak - original_peak,
            '日利用率>90%的天数': int(peak_days),
            '预计容量缺口_TEU': max(0, int(original_capacity * (new_peak - 0.9)))
        }

    def scenario_new_yard_area(self, bays: int = 10, rows: int = 8, tiers: int = 6) -> Dict:
        """
        情景2: 新增一个堆场区域后利用率下降幅度
        """
        new_area_id = f'NEW_{len(self.yard_df)+1:02d}'
        new_area_df = pd.DataFrame([{
            '区域编号': new_area_id,
            '贝位数量': bays,
            '列数': rows,
            '最大堆叠层数': tiers
        }])
        new_yard_df = pd.concat([self.yard_df, new_area_df], ignore_index=True)

        original_capacity = self._calculate_total_capacity()
        new_capacity = self._calculate_total_capacity(new_yard_df)

        original_peak = self._calculate_peak_utilization()

        overall = self.daily_util.groupby('日期').agg(
            总占用=('占用箱数', 'sum')
        ).reset_index()
        overall['总容量_原始'] = original_capacity
        overall['总容量_新增'] = new_capacity
        overall['原始利用率'] = overall['总占用'] / overall['总容量_原始']
        overall['新增后利用率'] = overall['总占用'] / overall['总容量_新增']

        new_peak = overall['新增后利用率'].max()
        avg_original = overall['原始利用率'].mean()
        avg_new = overall['新增后利用率'].mean()

        return {
            '情景名称': f'新增堆场区域({bays}x{rows}x{tiers})',
            '新增区域容量': bays * rows * tiers,
            '总容量增长率': (new_capacity - original_capacity) / original_capacity,
            '原始峰值利用率': original_peak,
            '新增后峰值利用率': new_peak,
            '峰值利用率下降幅度': original_peak - new_peak,
            '原始平均利用率': avg_original,
            '新增后平均利用率': avg_new,
            '平均利用率下降幅度': avg_original - avg_new
        }

    def scenario_reduce_storage_days(self, reduce_days: int = 1) -> Dict:
        """
        情景3: 缩短平均堆存天数对超期箱和翻箱率的影响
        reduce_days: 从当前均值减少N天
        """
        analyzer = OverdueBoxAnalyzer(free_days=5)

        original_df = self.container_df.copy()
        _, orig_by_customer, orig_by_area, orig_by_route = analyzer.analyze(original_df)

        original_overdue_count = (original_df['堆存时长_天'] > 5).sum()
        original_fee_sum = original_df['堆存时长_天'].apply(
            lambda x: analyzer.calculate_storage_fee(x, 5)
        ).sum()
        original_avg_storage = original_df['堆存时长_天'].mean()

        new_df = original_df.copy()
        new_df['堆存时长_天'] = np.maximum(1.0, new_df['堆存时长_天'] - reduce_days)
        new_df['出场时间'] = new_df['进场时间'] + pd.to_timedelta(new_df['堆存时长_天'], unit='D')

        _, new_by_customer, new_by_area, new_by_route = analyzer.analyze(new_df)

        new_overdue_count = (new_df['堆存时长_天'] > 5).sum()
        new_fee_sum = new_df['堆存时长_天'].apply(
            lambda x: analyzer.calculate_storage_fee(x, 5)
        ).sum()
        new_avg_storage = new_df['堆存时长_天'].mean()

        simulator = YardStackSimulator(self.yard_df, num_simulations=200, seed=42)
        results_list = []
        for area_id in self.yard_df['区域编号'].tolist()[:2]:
            fill_ratio_old = min(0.7, 0.7)
            fill_ratio_new = max(0.3, fill_ratio_old * (new_avg_storage / max(0.1, original_avg_storage)))

            orig_results = simulator.run_single_area_simulation(area_id, fill_ratio=fill_ratio_old)
            orig_rate = np.mean([r.avg_relocation_rate for r in orig_results.values()])

            new_results = simulator.run_single_area_simulation(area_id, fill_ratio=fill_ratio_new)
            new_rate = np.mean([r.avg_relocation_rate for r in new_results.values()])
            results_list.append((orig_rate, new_rate))

        avg_orig_rehandle = np.mean([r[0] for r in results_list]) if results_list else 0.0
        avg_new_rehandle = np.mean([r[1] for r in results_list]) if results_list else 0.0

        return {
            '情景名称': f'平均堆存天数减少{reduce_days}天',
            '原始平均堆存天数': original_avg_storage,
            '新平均堆存天数': new_avg_storage,
            '堆存天数减少量': original_avg_storage - new_avg_storage,
            '原始超期箱数量': int(original_overdue_count),
            '新超期箱数量': int(new_overdue_count),
            '超期箱减少数量': int(original_overdue_count - new_overdue_count),
            '超期箱减少比例': (original_overdue_count - new_overdue_count) / max(1, original_overdue_count),
            '原始滞箱费用': original_fee_sum,
            '新滞箱费用': new_fee_sum,
            '滞箱费用减少': original_fee_sum - new_fee_sum,
            '原始平均翻箱率': avg_orig_rehandle,
            '新平均翻箱率': avg_new_rehandle,
            '翻箱率改善幅度': avg_orig_rehandle - avg_new_rehandle
        }

    def run_all_scenarios(self) -> pd.DataFrame:
        """运行所有预设情景"""
        results = []

        for growth in [0.1, 0.2, 0.3]:
            r = self.scenario_throughput_growth(growth)
            results.append(r)

        r = self.scenario_new_yard_area(10, 8, 6)
        results.append(r)

        for days in [1, 2, 3]:
            r = self.scenario_reduce_storage_days(days)
            results.append(r)

        return pd.DataFrame(results)
