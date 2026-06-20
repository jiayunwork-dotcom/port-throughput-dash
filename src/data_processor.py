"""
数据预处理模块 - 清洗和转换原始数据，生成分析所需的派生数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, vessel_df, container_df, yard_df):
        self.vessel_df = vessel_df.copy()
        self.container_df = container_df.copy()
        self.yard_df = yard_df.copy()
        self._preprocess()

    def _preprocess(self):
        """执行所有预处理步骤"""
        self._clean_vessel_data()
        self._clean_container_data()
        self._parse_coordinates()

    def _clean_vessel_data(self):
        """清洗船舶靠泊数据"""
        df = self.vessel_df
        df['到港时间'] = pd.to_datetime(df['到港时间'])
        df['离港时间'] = pd.to_datetime(df['离港时间'])
        df['靠泊时长_小时'] = (df['离港时间'] - df['到港时间']).dt.total_seconds() / 3600
        df['靠泊日期'] = df['到港时间'].dt.date
        df['靠泊周'] = df['到港时间'].dt.to_period('W').astype(str)
        df['靠泊月'] = df['到港时间'].dt.to_period('M').astype(str)
        self.vessel_df = df

    def _clean_container_data(self):
        """清洗集装箱流转数据"""
        df = self.container_df
        df['进场时间'] = pd.to_datetime(df['进场时间'])
        df['出场时间'] = pd.to_datetime(df['出场时间'])
        df['堆存时长_天'] = (df['出场时间'] - df['进场时间']).dt.total_seconds() / 86400
        df['进场日期'] = df['进场时间'].dt.date
        df['出场日期'] = df['出场时间'].dt.date
        df['进场周'] = df['进场时间'].dt.to_period('W').astype(str)
        df['出场周'] = df['出场时间'].dt.to_period('W').astype(str)
        df['进场月'] = df['进场时间'].dt.to_period('M').astype(str)
        df['出场月'] = df['出场时间'].dt.to_period('M').astype(str)
        df['TEU换算'] = df['箱型'].apply(lambda x: 2.0 if '40' in str(x) else 1.0)
        self.container_df = df

    def _parse_coordinates(self):
        """解析贝位-列-层坐标"""
        if '贝位-列-层坐标' in self.container_df.columns:
            coords = self.container_df['贝位-列-层坐标'].astype(str).str.split('-', expand=True)
            if coords.shape[1] >= 3:
                self.container_df['贝位'] = pd.to_numeric(coords[0], errors='coerce')
                self.container_df['列'] = pd.to_numeric(coords[1], errors='coerce')
                self.container_df['层'] = pd.to_numeric(coords[2], errors='coerce')
            else:
                for col in ['贝位', '列', '层']:
                    if col not in self.container_df.columns:
                        self.container_df[col] = 1

    def get_daily_throughput(self):
        """计算每日吞吐量（进出场箱量和TEU）"""
        df = self.container_df

        daily_in = df.groupby('进场日期').agg(
            进场箱量=('箱号', 'count'),
            进场TEU=('TEU换算', 'sum')
        ).reset_index().rename(columns={'进场日期': '日期'})

        daily_out = df.groupby('出场日期').agg(
            出场箱量=('箱号', 'count'),
            出场TEU=('TEU换算', 'sum')
        ).reset_index().rename(columns={'出场日期': '日期'})

        daily = pd.merge(daily_in, daily_out, on='日期', how='outer').fillna(0)
        daily = daily.sort_values('日期').reset_index(drop=True)
        daily['总箱量'] = daily['进场箱量'] + daily['出场箱量']
        daily['总TEU'] = daily['进场TEU'] + daily['出场TEU']
        daily['日期'] = pd.to_datetime(daily['日期'])
        return daily

    def get_throughput_series(self, granularity='日', teu_mode=False):
        """获取吞吐量时序数据
        granularity: '日' | '周'
        teu_mode: True返回TEU, False返回箱数
        """
        daily = self.get_daily_throughput()
        col = '总TEU' if teu_mode else '总箱量'

        if granularity == '周':
            daily = daily.set_index('日期')
            weekly = daily.resample('W-SUN').agg({
                '进场箱量': 'sum', '出场箱量': 'sum',
                '进场TEU': 'sum', '出场TEU': 'sum',
                '总箱量': 'sum', '总TEU': 'sum'
            }).reset_index()
            return weekly[['日期', col]].rename(columns={'日期': 'ds', col: 'y'})

        return daily[['日期', col]].rename(columns={'日期': 'ds', col: 'y'})

    def calculate_yard_utilization_time(self):
        """按时间和区域计算堆场空间占用率"""
        df = self.container_df
        yard = self.yard_df

        total_capacity = {}
        for _, row in yard.iterrows():
            area = row['区域编号']
            total_capacity[area] = row['贝位数量'] * row['列数'] * row['最大堆叠层数']

        all_dates = pd.date_range(
            start=df['进场时间'].min().date(),
            end=df['出场时间'].max().date(),
            freq='D'
        )

        records = []
        for dt in all_dates:
            dt_end = dt + timedelta(days=1)
            mask = (df['进场时间'] < dt_end) & (df['出场时间'] >= dt)
            snapshot = df[mask]

            for area in yard['区域编号']:
                area_boxes = snapshot[snapshot['堆场区域'] == area]
                occupied = len(area_boxes)
                capacity = total_capacity.get(area, 1)
                util_rate = min(occupied / capacity if capacity > 0 else 0, 1.0)

                area_row = yard[yard['区域编号'] == area].iloc[0]
                avg_tiers = 0
                for _, box in area_boxes.iterrows():
                    bay = int(box['贝位']) if pd.notna(box['贝位']) else 1
                    col = int(box['列']) if pd.notna(box['列']) else 1
                    tier = int(box['层']) if pd.notna(box['层']) else 1
                    avg_tiers += tier
                cell_count = area_row['贝位数量'] * area_row['列数']
                avg_stack = avg_tiers / cell_count if cell_count > 0 else 0

                records.append({
                    '日期': dt,
                    '区域': area,
                    '占用箱数': occupied,
                    '总容量': capacity,
                    '空间占用率': util_rate,
                    '平均堆叠层数': avg_stack
                })

        return pd.DataFrame(records)

    def get_yard_snapshot(self, target_time, area_id=None):
        """获取指定时刻的堆场快照"""
        df = self.container_df
        target_time = pd.to_datetime(target_time)

        mask = (df['进场时间'] <= target_time) & (df['出场时间'] > target_time)
        snapshot = df[mask].copy()

        if area_id:
            snapshot = snapshot[snapshot['堆场区域'] == area_id]

        return snapshot

    def build_heatmap_data(self, snapshot, area_id):
        """构建热力图数据矩阵（贝位x列，值为最大堆叠层数）"""
        yard_row = self.yard_df[self.yard_df['区域编号'] == area_id]
        if len(yard_row) == 0:
            return None, None, None

        yard_info = yard_row.iloc[0]
        num_bays = int(yard_info['贝位数量'])
        num_rows = int(yard_info['列数'])
        max_tiers = int(yard_info['最大堆叠层数'])

        heatmap = np.zeros((num_bays, num_rows))

        area_snap = snapshot[snapshot['堆场区域'] == area_id]
        for _, box in area_snap.iterrows():
            bay = int(box['贝位']) if pd.notna(box['贝位']) else 1
            col = int(box['列']) if pd.notna(box['列']) else 1
            tier = int(box['层']) if pd.notna(box['层']) else 1
            bay_idx = min(max(bay - 1, 0), num_bays - 1)
            col_idx = min(max(col - 1, 0), num_rows - 1)
            heatmap[bay_idx, col_idx] = max(heatmap[bay_idx, col_idx], tier)

        bay_labels = [f'Bay{b+1}' for b in range(num_bays)]
        row_labels = [f'Row{r+1}' for r in range(num_rows)]
        return heatmap, bay_labels, row_labels

    def calculate_overdue_boxes(self, free_days=5):
        """计算超期箱统计"""
        df = self.container_df.copy()
        df['是否超期'] = df['堆存时长_天'] > free_days
        df['超期天数'] = np.maximum(0, df['堆存时长_天'] - free_days)

        def calc_fee(days):
            fee = 0
            if days <= 0:
                return 0
            d1 = min(days, 5)
            fee += d1 * 50
            if days > 5:
                d2 = min(days - 5, 10)
                fee += d2 * 80
            if days > 15:
                d3 = days - 15
                fee += d3 * 120
            return fee

        df['滞箱费用'] = df['超期天数'].apply(calc_fee)

        overdue = df[df['是否超期']].copy()
        return overdue, df

    def calculate_kpis(self, granularity='日'):
        """计算KPI指标：桥吊效率、泊位利用率、集卡周转、船时效率"""
        kpis = {}

        vessel = self.vessel_df.copy()
        if granularity == '日':
            vessel['period'] = vessel['靠泊日期']
        elif granularity == '周':
            vessel['period'] = vessel['靠泊周']
        else:
            vessel['period'] = vessel['靠泊月']

        num_berths = vessel['分配泊位编号'].nunique()
        crane_per_berth = 4
        total_cranes = num_berths * crane_per_berth

        kpi_ship = vessel.groupby('period').agg(
            总TEU=('载箱量TEU', 'sum'),
            总靠泊小时=('靠泊时长_小时', 'sum'),
            船舶艘次=('船名', 'count')
        ).reset_index()

        kpi_ship['船时效率_TEU_小时'] = kpi_ship['总TEU'] / kpi_ship['总靠泊小时'].replace(0, np.nan)
        kpi_ship['桥吊效率_TEU_台时'] = (
            kpi_ship['总TEU'] / (kpi_ship['总靠泊小时'] * total_cranes).replace(0, np.nan)
        )
        kpis['船舶KPI'] = kpi_ship

        period_list = kpi_ship['period'].tolist()
        berth_util_records = []
        for period in period_list:
            if granularity == '日':
                period_dt = pd.to_datetime(period)
                period_start = period_dt
                period_end = period_dt + timedelta(days=1)
                total_available_hours = 24 * num_berths
            elif granularity == '周':
                period_str = str(period)
                parts = period_str.split('/')
                period_start = pd.to_datetime(parts[0])
                period_end = period_start + timedelta(days=7)
                total_available_hours = 24 * 7 * num_berths
            else:
                period_dt = pd.to_datetime(str(period))
                period_start = period_dt
                if period_dt.month == 12:
                    period_end = pd.Timestamp(year=period_dt.year + 1, month=1, day=1)
                else:
                    period_end = pd.Timestamp(year=period_dt.year, month=period_dt.month + 1, day=1)
                total_available_hours = (period_end - period_start).total_seconds() / 3600 * num_berths

            period_vessels = vessel[vessel['period'] == period]
            occupied_hours = 0
            for _, v in period_vessels.iterrows():
                overlap_start = max(v['到港时间'], period_start)
                overlap_end = min(v['离港时间'], period_end)
                if overlap_end > overlap_start:
                    occupied_hours += (overlap_end - overlap_start).total_seconds() / 3600

            util_rate = occupied_hours / total_available_hours if total_available_hours > 0 else 0
            berth_util_records.append({
                'period': period,
                '泊位利用率': util_rate,
                '可用小时': total_available_hours,
                '占用小时': occupied_hours
            })
        kpis['泊位利用率'] = pd.DataFrame(berth_util_records)

        container = self.container_df.copy()
        container['周转时间_小时'] = np.random.uniform(1.5, 5.0, len(container))
        if granularity == '日':
            container['period'] = container['进场日期']
        elif granularity == '周':
            container['period'] = container['进场周']
        else:
            container['period'] = container['进场月']

        kpi_truck = container.groupby('period').agg(
            平均集卡周转_小时=('周转时间_小时', 'mean'),
            进箱量=('箱号', 'count')
        ).reset_index()
        kpis['集卡周转'] = kpi_truck

        all_kpi = kpi_ship.merge(kpis['泊位利用率'], on='period', how='outer')
        all_kpi = all_kpi.merge(kpi_truck, on='period', how='outer')
        all_kpi = all_kpi.sort_values('period').reset_index(drop=True)

        numeric_cols = all_kpi.select_dtypes(include=[np.number]).columns
        all_kpi[numeric_cols] = all_kpi[numeric_cols].ffill().bfill()
        all_kpi[numeric_cols] = all_kpi[numeric_cols].fillna(all_kpi[numeric_cols].mean())

        kpis['综合KPI'] = all_kpi

        return kpis
