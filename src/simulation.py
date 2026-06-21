"""
船舶排队调度离散事件仿真引擎
模拟船舶从到港锚地等待 → 分配泊位 → 装卸完成离港的全流程

支持三种泊位分配策略:
- FCFS: 先到先服务 (First Come First Served)
- SJF: 最短作业优先 (Shortest Job First)，预估装卸时间最短的船优先
- LWF: 最长作业优先 (Longest Job First)，预估装卸时间最长的船优先
"""

import heapq
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

STRATEGY_FCFS = 'FCFS'
STRATEGY_SJF = 'SJF'
STRATEGY_LWF = 'LWF'
STRATEGY_ALL = 'ALL'

STRATEGY_LABELS = {
    STRATEGY_FCFS: '先到先服务 (FCFS)',
    STRATEGY_SJF: '最短作业优先 (SJF)',
    STRATEGY_LWF: '最长作业优先 (LWF)',
}


@dataclass
class ShipRecord:
    """单艘船舶的记录"""
    ship_id: int
    arrival_time: float = 0.0
    start_time: float = 0.0
    departure_time: float = 0.0
    wait_time: float = 0.0
    berth_id: int = -1
    service_time: float = 0.0
    rejected: bool = False
    estimated_service_time: float = 0.0


@dataclass
class SimulationEvent:
    """仿真事件"""
    time: float
    event_type: str
    ship_id: int
    priority: int = 0

    def __lt__(self, other):
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority


@dataclass
class SimulationResult:
    """仿真结果"""
    ships: List[ShipRecord] = field(default_factory=list)
    timeline_data: pd.DataFrame = None
    berth_occupancy: pd.DataFrame = None
    stats: Dict = field(default_factory=dict)
    params: Dict = field(default_factory=dict)
    strategy: str = STRATEGY_FCFS
    berth_timeline: Dict = field(default_factory=dict)


class ShipQueueSimulator:
    """船舶排队调度仿真器"""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def run(self,
            arrival_mean: float = 8.0,
            service_mean: float = 36.0,
            service_std: float = 6.0,
            num_berths: int = 4,
            max_anchor: int = 20,
            sim_duration: float = 720.0,
            strategy: str = STRATEGY_FCFS) -> SimulationResult:
        """
        运行一次仿真

        Args:
            arrival_mean: 船舶到港间隔均值 (小时), 指数分布
            service_mean: 装卸作业时长均值 (小时), 正态分布
            service_std: 装卸作业时长标准差 (小时)
            num_berths: 泊位数量
            max_anchor: 最大锚地等待容量
            sim_duration: 仿真时长 (小时)
            strategy: 泊位分配策略 (FCFS/SJF/LWF)

        Returns:
            SimulationResult 仿真结果
        """
        rng = np.random.default_rng(self.seed)

        ships: List[ShipRecord] = []
        event_queue: List[SimulationEvent] = []
        anchor_queue: List[int] = []
        berth_available: List[bool] = [True] * num_berths
        berth_end_times: List[float] = [0.0] * num_berths

        timeline_events = []
        berth_timeline = {i: [] for i in range(num_berths)}

        current_time = 0.0
        ship_counter = 0

        first_arrival = rng.exponential(arrival_mean)
        if first_arrival <= sim_duration:
            heapq.heappush(event_queue, SimulationEvent(first_arrival, 'ship_arrive', ship_counter, 0))
            ship_counter += 1

        anchor_count = 0
        berth_count = 0

        while event_queue:
            event = heapq.heappop(event_queue)
            current_time = event.time

            if current_time > sim_duration:
                break

            if event.event_type == 'ship_arrive':
                est_service = max(1.0, rng.normal(service_mean, service_std))
                ship = ShipRecord(
                    ship_id=event.ship_id,
                    arrival_time=current_time,
                    estimated_service_time=est_service
                )
                ships.append(ship)

                assigned = False
                for berth_id in range(num_berths):
                    if berth_available[berth_id]:
                        self._assign_berth(ship, berth_id, current_time, service_mean, service_std, rng)
                        berth_available[berth_id] = False
                        berth_end_times[berth_id] = ship.departure_time
                        heapq.heappush(event_queue, SimulationEvent(
                            ship.departure_time, 'berth_release', ship.ship_id, 2
                        ))
                        berth_timeline[berth_id].append((current_time, ship.departure_time, ship.ship_id))
                        assigned = True
                        break

                if not assigned:
                    if len(anchor_queue) < max_anchor:
                        self._enqueue_ship(anchor_queue, ships, ship, strategy, current_time)
                        ship.wait_time = 0.0
                    else:
                        ship.rejected = True

                next_arrival = current_time + rng.exponential(arrival_mean)
                if next_arrival <= sim_duration:
                    heapq.heappush(event_queue, SimulationEvent(next_arrival, 'ship_arrive', ship_counter, 0))
                    ship_counter += 1

                timeline_events.append((current_time, len(anchor_queue), sum(1 for b in berth_available if not b)))

            elif event.event_type == 'berth_release':
                ship = next((s for s in ships if s.ship_id == event.ship_id), None)
                if ship is None or ship.berth_id < 0:
                    continue

                berth_id = ship.berth_id
                berth_available[berth_id] = True

                if anchor_queue:
                    next_ship_id = self._dequeue_ship(anchor_queue, ships, strategy, current_time)
                    next_ship = next((s for s in ships if s.ship_id == next_ship_id), None)
                    if next_ship is not None:
                        self._assign_berth(next_ship, berth_id, current_time, service_mean, service_std, rng)
                        berth_available[berth_id] = False
                        berth_end_times[berth_id] = next_ship.departure_time
                        heapq.heappush(event_queue, SimulationEvent(
                            next_ship.departure_time, 'berth_release', next_ship.ship_id, 2
                        ))
                        berth_timeline[berth_id].append((current_time, next_ship.departure_time, next_ship.ship_id))

                timeline_events.append((current_time, len(anchor_queue), sum(1 for b in berth_available if not b)))

        accepted_ships = [s for s in ships if not s.rejected]
        completed_ships = [s for s in accepted_ships if s.departure_time > 0 and s.departure_time <= sim_duration]

        timeline_df = self._build_timeline_data(timeline_events, sim_duration)

        berth_records = []
        for berth_id, intervals in berth_timeline.items():
            for start, end, sid in intervals:
                if start <= sim_duration:
                    actual_end = min(end, sim_duration)
                    ship = next((s for s in ships if s.ship_id == sid), None)
                    service_t = actual_end - start
                    berth_records.append({
                        '泊位编号': f'泊位{berth_id + 1}',
                        '船名': f'船{sid + 1}',
                        '开始时间': start,
                        '结束时间': actual_end,
                        '服务时长': service_t,
                        '载箱量TEU': int(1500 + rng.random() * 4000)
                    })
        berth_df = pd.DataFrame(berth_records) if berth_records else pd.DataFrame(
            columns=['泊位编号', '船名', '开始时间', '结束时间', '服务时长', '载箱量TEU']
        )

        total_ships = len(ships)
        rejected_ships = sum(1 for s in ships if s.rejected)
        served_ships = len(completed_ships)

        wait_times = [s.wait_time for s in completed_ships if s.wait_time > 0 or s.start_time > s.arrival_time]
        all_wait_times = [s.start_time - s.arrival_time for s in completed_ships]
        avg_wait = np.mean(all_wait_times) if all_wait_times else 0.0
        max_wait = np.max(all_wait_times) if all_wait_times else 0.0

        service_times = [s.service_time for s in completed_ships]
        avg_service = np.mean(service_times) if service_times else 0.0

        total_busy_hours = sum(
            min(berth_end_times[i], sim_duration) - 0 if berth_end_times[i] > 0 else 0
            for i in range(num_berths)
        )
        berth_timeline_saved = {}
        for berth_id, intervals in berth_timeline.items():
            total_busy = 0
            for start, end, _ in intervals:
                if start < sim_duration:
                    total_busy += min(end, sim_duration) - start
            berth_util = total_busy / max(1, sim_duration)
            berth_timeline_saved[berth_id] = (intervals, berth_util)

        total_berth_hours = num_berths * sim_duration
        total_busy_all = 0
        for berth_id in range(num_berths):
            intervals_data = berth_timeline_saved.get(berth_id, ([], 0))
            interval_list = intervals_data[0]
            for start, end, _ in interval_list:
                if start < sim_duration:
                    total_busy_all += min(end, sim_duration) - start
        avg_berth_util = total_busy_all / total_berth_hours if total_berth_hours > 0 else 0.0

        reject_rate = rejected_ships / max(1, total_ships)

        stats = {
            'avg_wait_time': float(avg_wait),
            'max_wait_time': float(max_wait),
            'avg_berth_utilization': float(avg_berth_util),
            'avg_service_time': float(avg_service),
            'throughput': served_ships,
            'total_arrivals': total_ships,
            'rejected_count': rejected_ships,
            'reject_rate': float(reject_rate),
        }

        result = SimulationResult(
            ships=ships,
            timeline_data=timeline_df,
            berth_occupancy=berth_df,
            stats=stats,
            params={
                'arrival_mean': arrival_mean,
                'service_mean': service_mean,
                'service_std': service_std,
                'num_berths': num_berths,
                'max_anchor': max_anchor,
                'sim_duration': sim_duration,
            },
            strategy=strategy,
            berth_timeline=berth_timeline_saved
        )

        return result

    def _enqueue_ship(self, anchor_queue: List[int], ships: List[ShipRecord],
                      ship: ShipRecord, strategy: str, current_time: float):
        """
        根据策略将船舶加入等待队列
        """
        if strategy == STRATEGY_FCFS:
            anchor_queue.append(ship.ship_id)
        elif strategy == STRATEGY_SJF:
            inserted = False
            for i, sid in enumerate(anchor_queue):
                q_ship = next((s for s in ships if s.ship_id == sid), None)
                if q_ship and ship.estimated_service_time < q_ship.estimated_service_time:
                    anchor_queue.insert(i, ship.ship_id)
                    inserted = True
                    break
            if not inserted:
                anchor_queue.append(ship.ship_id)
        elif strategy == STRATEGY_LWF:
            inserted = False
            for i, sid in enumerate(anchor_queue):
                q_ship = next((s for s in ships if s.ship_id == sid), None)
                if q_ship and ship.estimated_service_time > q_ship.estimated_service_time:
                    anchor_queue.insert(i, ship.ship_id)
                    inserted = True
                    break
            if not inserted:
                anchor_queue.append(ship.ship_id)
        else:
            anchor_queue.append(ship.ship_id)

    def _dequeue_ship(self, anchor_queue: List[int], ships: List[ShipRecord],
                      strategy: str, current_time: float) -> Optional[int]:
        """
        根据策略从等待队列中选择下一艘船
        返回选中的船舶ID
        """
        if not anchor_queue:
            return None

        if strategy == STRATEGY_FCFS:
            return anchor_queue.pop(0)
        elif strategy == STRATEGY_SJF:
            min_idx = 0
            min_time = float('inf')
            for i, sid in enumerate(anchor_queue):
                q_ship = next((s for s in ships if s.ship_id == sid), None)
                if q_ship and q_ship.estimated_service_time < min_time:
                    min_time = q_ship.estimated_service_time
                    min_idx = i
            return anchor_queue.pop(min_idx)
        elif strategy == STRATEGY_LWF:
            max_idx = 0
            max_time = -1
            for i, sid in enumerate(anchor_queue):
                q_ship = next((s for s in ships if s.ship_id == sid), None)
                if q_ship and q_ship.estimated_service_time > max_time:
                    max_time = q_ship.estimated_service_time
                    max_idx = i
            return anchor_queue.pop(max_idx)
        else:
            return anchor_queue.pop(0)

    def _assign_berth(self, ship: ShipRecord, berth_id: int, current_time: float,
                      service_mean: float, service_std: float, rng: np.random.Generator):
        """分配泊位给船舶"""
        ship.berth_id = berth_id
        ship.start_time = current_time
        ship.wait_time = current_time - ship.arrival_time
        service_time = max(1.0, rng.normal(service_mean, service_std))
        ship.service_time = service_time
        ship.departure_time = current_time + service_time

    def _build_timeline_data(self, events: List[Tuple[float, int, int]], sim_duration: float) -> pd.DataFrame:
        """构建时间轴数据"""
        if not events:
            return pd.DataFrame(columns=['时间_小时', '锚地等待数', '在泊作业数'])

        sorted_events = sorted(events, key=lambda x: x[0])
        records = []
        anchor_count = 0
        berth_count = 0

        for time, anchor, berth in sorted_events:
            anchor_count = anchor
            berth_count = berth
            records.append({
                '时间_小时': time,
                '锚地等待数': anchor_count,
                '在泊作业数': berth_count
            })

        records.append({
            '时间_小时': sim_duration,
            '锚地等待数': anchor_count,
            '在泊作业数': berth_count
        })

        return pd.DataFrame(records)

    def run_sensitivity_analysis(self,
                                 arrival_means: List[float],
                                 service_mean: float = 36.0,
                                 service_std: float = 6.0,
                                 num_berths: int = 4,
                                 max_anchor: int = 20,
                                 sim_duration: float = 720.0,
                                 base_seed: int = 42,
                                 strategy: str = STRATEGY_FCFS) -> pd.DataFrame:
        """
        敏感性分析：批量运行不同到港间隔下的仿真

        Args:
            arrival_means: 到港间隔均值列表
            service_mean: 装卸作业时长均值
            service_std: 装卸作业时长标准差
            num_berths: 泊位数量
            max_anchor: 最大锚地容量
            sim_duration: 仿真时长
            base_seed: 基础随机种子
            strategy: 泊位分配策略

        Returns:
            DataFrame 包含各配置下的指标
        """
        results = []

        for i, arr_mean in enumerate(arrival_means):
            sim = ShipQueueSimulator(seed=base_seed + i)
            result = sim.run(
                arrival_mean=arr_mean,
                service_mean=service_mean,
                service_std=service_std,
                num_berths=num_berths,
                max_anchor=max_anchor,
                sim_duration=sim_duration,
                strategy=strategy
            )

            results.append({
                '到港间隔_小时': arr_mean,
                '平均等待时长_小时': result.stats['avg_wait_time'],
                '最大等待时长_小时': result.stats['max_wait_time'],
                '泊位平均利用率': result.stats['avg_berth_utilization'],
                '平均服务时间_小时': result.stats['avg_service_time'],
                '吞吐量_艘': result.stats['throughput'],
                '拒绝率': result.stats['reject_rate'],
                '总到港数': result.stats['total_arrivals'],
            })

        return pd.DataFrame(results)

    @staticmethod
    def get_stats_at_time(result: SimulationResult, time_point: float) -> Dict:
        """
        计算截至某一时刻的统计指标（用于仿真回放）

        Args:
            result: 仿真结果
            time_point: 时间点 (小时)

        Returns:
            截至该时刻的统计指标字典
        """
        sim_duration = result.params.get('sim_duration', 0)
        num_berths = result.params.get('num_berths', 0)

        completed_ships = [
            s for s in result.ships
            if not s.rejected
            and s.departure_time > 0
            and s.departure_time <= time_point
        ]

        in_service_ships = [
            s for s in result.ships
            if not s.rejected
            and s.start_time > 0
            and s.start_time <= time_point
            and s.departure_time > time_point
        ]

        waiting_ships = [
            s for s in result.ships
            if not s.rejected
            and s.arrival_time <= time_point
            and (s.start_time == 0 or s.start_time > time_point)
        ]

        all_served = completed_ships + in_service_ships
        wait_times = []
        for s in all_served:
            actual_start = min(s.start_time, time_point) if s.start_time > 0 else time_point
            wt = actual_start - s.arrival_time
            if wt > 0:
                wait_times.append(wt)

        avg_wait = float(np.mean(wait_times)) if wait_times else 0.0
        max_wait = float(np.max(wait_times)) if wait_times else 0.0

        total_busy = 0.0
        for s in result.ships:
            if s.rejected or s.start_time == 0:
                continue
            if s.start_time >= time_point:
                continue
            start = s.start_time
            end = min(s.departure_time, time_point) if s.departure_time > 0 else time_point
            if end > start:
                total_busy += end - start

        total_berth_hours = num_berths * time_point
        berth_util = total_busy / total_berth_hours if total_berth_hours > 0 else 0.0

        throughput = len(completed_ships)

        rejected_before = sum(
            1 for s in result.ships
            if s.rejected and s.arrival_time <= time_point
        )

        return {
            'avg_wait_time': avg_wait,
            'max_wait_time': max_wait,
            'avg_berth_utilization': float(berth_util),
            'throughput': throughput,
            'total_arrivals': sum(
                1 for s in result.ships if s.arrival_time <= time_point
            ),
            'rejected_count': rejected_before,
            'waiting_count': len(waiting_ships),
            'in_service_count': len(in_service_ships),
        }
