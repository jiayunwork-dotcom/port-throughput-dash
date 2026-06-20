"""
翻箱率分析模块 - 蒙特卡洛模拟三种堆存策略
策略A: 随机堆放
策略B: 按预计提箱时间排序（早提的放上层）
策略C: 按航线聚堆（同航线箱放同列）
"""

import numpy as np
import pandas as pd
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


@dataclass
class Container:
    """集装箱数据类"""
    id: str
    bay: int
    row: int
    tier: int
    exit_time: pd.Timestamp
    route: str
    area: str


@dataclass
class SimulationResult:
    """模拟结果"""
    strategy: str
    total_pickups: int = 0
    total_relocations: int = 0
    relocation_rates: List[float] = field(default_factory=list)

    @property
    def avg_relocation_rate(self):
        if len(self.relocation_rates) == 0:
            return 0
        return np.mean(self.relocation_rates)

    @property
    def p95_relocation_rate(self):
        if len(self.relocation_rates) == 0:
            return 0
        return np.percentile(self.relocation_rates, 95)


class YardStackSimulator:
    """堆场翻箱率蒙特卡洛模拟器"""

    STRATEGIES = ['策略A-随机堆放', '策略B-按提箱时间排序', '策略C-按航线聚堆']

    def __init__(self, yard_config: pd.DataFrame, num_simulations: int = 1000, seed: int = 42):
        self.yard_config = yard_config
        self.num_simulations = num_simulations
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _sample_containers(
        self,
        n_containers: int,
        area_config: Dict,
        area_id: str,
        routes: List[str]
    ) -> List[Container]:
        """采样生成一批集装箱数据"""
        containers = []
        base_time = pd.Timestamp('2026-04-01')

        for i in range(n_containers):
            exit_time = base_time + pd.Timedelta(days=random.uniform(1, 15))
            route = random.choice(routes)
            containers.append(Container(
                id=f'BOX_{i:06d}',
                bay=0, row=0, tier=0,
                exit_time=exit_time,
                route=route,
                area=area_id
            ))
        return containers

    def _place_strategy_a(
        self, containers: List[Container],
        bays: int, rows: int, max_tiers: int
    ) -> List[Container]:
        """策略A: 随机堆放"""
        placed = []
        available_slots = []
        for b in range(bays):
            for r in range(rows):
                for t in range(max_tiers):
                    available_slots.append((b, r, t))
        random.shuffle(available_slots)

        for i, cont in enumerate(containers):
            if i < len(available_slots):
                b, r, t = available_slots[i]
                cont.bay = b
                cont.row = r
                cont.tier = t
                placed.append(cont)
        return placed

    def _place_strategy_b(
        self, containers: List[Container],
        bays: int, rows: int, max_tiers: int
    ) -> List[Container]:
        """策略B: 按预计提箱时间排序（早提的放上层）"""
        sorted_conts = sorted(containers, key=lambda c: c.exit_time, reverse=True)
        placed = []

        idx = 0
        for b in range(bays):
            for r in range(rows):
                for t in range(max_tiers):
                    if idx < len(sorted_conts):
                        sorted_conts[idx].bay = b
                        sorted_conts[idx].row = r
                        sorted_conts[idx].tier = t
                        placed.append(sorted_conts[idx])
                        idx += 1
        return placed

    def _place_strategy_c(
        self, containers: List[Container],
        bays: int, rows: int, max_tiers: int,
        routes: List[str]
    ) -> List[Container]:
        """策略C: 按航线聚堆（同航线箱放同列）"""
        route_row_map = {}
        for i, route in enumerate(routes):
            route_row_map[route] = i % rows

        row_stack = {r: 0 for r in range(rows)}
        placed = []

        for cont in containers:
            preferred_row = route_row_map.get(cont.route, 0)
            r = preferred_row
            if row_stack[r] >= max_tiers * bays:
                for alt_r in range(rows):
                    if row_stack[alt_r] < max_tiers * bays:
                        r = alt_r
                        break

            stack_pos = row_stack[r]
            b = (stack_pos // max_tiers) % bays
            t = stack_pos % max_tiers

            cont.bay = b
            cont.row = r
            cont.tier = t
            row_stack[r] += 1
            placed.append(cont)
        return placed

    def _simulate_pickups(self, containers: List[Container]) -> Tuple[int, int]:
        """
        模拟提箱过程，计算翻箱次数
        提箱顺序: 按出场时间从早到晚
        翻箱判定: 目标箱不在最顶层则其上方所有箱计一次翻箱
        """
        if len(containers) == 0:
            return 0, 0

        bay_row_stacks: Dict[Tuple[int, int], List[Container]] = {}
        for cont in containers:
            key = (cont.bay, cont.row)
            if key not in bay_row_stacks:
                bay_row_stacks[key] = []
            bay_row_stacks[key].append(cont)

        for key in bay_row_stacks:
            bay_row_stacks[key].sort(key=lambda c: c.tier)

        pickup_order = sorted(containers, key=lambda c: c.exit_time)

        total_relocations = 0
        total_pickups = 0
        present = {cont.id: cont for cont in containers}

        for target in pickup_order:
            if target.id not in present:
                continue

            key = (target.bay, target.row)
            stack = bay_row_stacks.get(key, [])
            stack_ids = [c.id for c in stack]

            try:
                target_idx = stack_ids.index(target.id)
            except ValueError:
                continue

            above_boxes = stack[target_idx + 1:]
            total_relocations += len(above_boxes)

            for box in above_boxes:
                box_id = box.id
                if box_id in present:
                    placed = False
                    for alt_key, alt_stack in bay_row_stacks.items():
                        if alt_key == key:
                            continue
                        if len(alt_stack) < 10:
                            max_t = max((c.tier for c in alt_stack), default=-1)
                            box.bay, box.row = alt_key
                            box.tier = max_t + 1
                            alt_stack.append(box)
                            bay_row_stacks[alt_key] = sorted(alt_stack, key=lambda c: c.tier)
                            placed = True
                            break
                    if not placed:
                        del present[box_id]

            new_stack = [c for c in stack if c.id != target.id]
            for c in new_stack:
                key_new = (c.bay, c.row)
            bay_row_stacks[key] = new_stack
            del present[target.id]
            total_pickups += 1

        return total_pickups, total_relocations

    def run_single_area_simulation(
        self,
        area_id: str,
        fill_ratio: float = 0.7,
        sample_containers: List[Container] = None
    ) -> Dict[str, SimulationResult]:
        """对单个区域执行三种策略的模拟"""
        area_row = self.yard_config[self.yard_config['区域编号'] == area_id]
        if len(area_row) == 0:
            return {}

        area_info = area_row.iloc[0]
        bays = int(area_info['贝位数量'])
        rows = int(area_info['列数'])
        max_tiers = int(area_info['最大堆叠层数'])
        total_slots = bays * rows * max_tiers
        n_containers = int(total_slots * fill_ratio)

        routes = ['欧洲-远东', '跨太平洋东行', '跨太平洋西行', '亚洲区内',
                  '澳新航线', '中东印巴', '南美航线', '非洲航线']

        results = {}
        for strategy in self.STRATEGIES:
            results[strategy] = SimulationResult(strategy=strategy)

        for sim in range(self.num_simulations):
            sim_seed = self.seed + sim
            random.seed(sim_seed)
            np.random.seed(sim_seed)

            containers = self._sample_containers(
                n_containers, area_info.to_dict(), area_id, routes
            )

            for strategy in self.STRATEGIES:
                sim_conts = [Container(
                    id=c.id, bay=0, row=0, tier=0,
                    exit_time=c.exit_time, route=c.route, area=c.area
                ) for c in containers]

                if strategy == '策略A-随机堆放':
                    placed = self._place_strategy_a(sim_conts, bays, rows, max_tiers)
                elif strategy == '策略B-按提箱时间排序':
                    placed = self._place_strategy_b(sim_conts, bays, rows, max_tiers)
                else:
                    placed = self._place_strategy_c(sim_conts, bays, rows, max_tiers, routes)

                pickups, relocations = self._simulate_pickups(placed)
                rate = relocations / pickups if pickups > 0 else 0

                results[strategy].total_pickups += pickups
                results[strategy].total_relocations += relocations
                results[strategy].relocation_rates.append(rate)

        return results

    def run_full_simulation(self) -> pd.DataFrame:
        """对所有区域运行模拟并汇总结果"""
        all_results = []

        for area_id in self.yard_config['区域编号'].tolist():
            area_results = self.run_single_area_simulation(area_id)
            for strategy, result in area_results.items():
                all_results.append({
                    '区域': area_id,
                    '策略': strategy,
                    '平均翻箱率': result.avg_relocation_rate,
                    'P95翻箱率': result.p95_relocation_rate,
                    '总提箱次数': result.total_pickups,
                    '总翻箱次数': result.total_relocations
                })

        return pd.DataFrame(all_results)


class OverdueBoxAnalyzer:
    """超期箱统计分析器"""

    def __init__(self, free_days: int = 5):
        self.free_days = free_days

    @staticmethod
    def calculate_storage_fee(stay_days: float, free_days: int = 5) -> float:
        """计算阶梯滞箱费
        第6-10天: 50元/天
        第11-20天: 80元/天
        超过20天: 120元/天
        """
        overdue = max(0, stay_days - free_days)
        if overdue <= 0:
            return 0.0
        fee = 0.0
        d1 = min(overdue, 5)
        fee += d1 * 50
        if overdue > 5:
            d2 = min(overdue - 5, 10)
            fee += d2 * 80
        if overdue > 15:
            d3 = overdue - 15
            fee += d3 * 120
        return fee

    def analyze(self, container_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        分析超期箱
        返回: (超期箱明细, 按客户分组, 按区域分组)
        """
        df = container_df.copy()

        if '堆存时长_天' not in df.columns:
            df['进场时间'] = pd.to_datetime(df['进场时间'])
            df['出场时间'] = pd.to_datetime(df['出场时间'])
            df['堆存时长_天'] = (df['出场时间'] - df['进场时间']).dt.total_seconds() / 86400

        df['是否超期'] = df['堆存时长_天'] > self.free_days
        df['超期天数'] = np.maximum(0, df['堆存时长_天'] - self.free_days)
        df['滞箱费用'] = df['堆存时长_天'].apply(
            lambda x: self.calculate_storage_fee(x, self.free_days)
        )

        overdue_df = df[df['是否超期']].copy()

        by_customer = overdue_df.groupby('客户').agg(
            超期箱数量=('箱号', 'count'),
            平均超期天数=('超期天数', 'mean'),
            预估滞箱费用=('滞箱费用', 'sum'),
            平均堆存天数=('堆存时长_天', 'mean')
        ).reset_index().sort_values('超期箱数量', ascending=False)

        by_area = overdue_df.groupby('堆场区域').agg(
            超期箱数量=('箱号', 'count'),
            平均超期天数=('超期天数', 'mean'),
            预估滞箱费用=('滞箱费用', 'sum')
        ).reset_index().sort_values('超期箱数量', ascending=False)

        by_route = overdue_df.groupby('航线').agg(
            超期箱数量=('箱号', 'count'),
            预估滞箱费用=('滞箱费用', 'sum')
        ).reset_index().sort_values('超期箱数量', ascending=False)

        return overdue_df, by_customer, by_area, by_route
