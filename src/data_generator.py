"""
数据生成模块 - 生成模拟的港口运营数据
包括：船舶靠泊记录、集装箱流转记录、堆场布局定义
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os


class PortDataGenerator:
    """港口模拟数据生成器"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.start_date = datetime(2026, 3, 1)
        self.days = 90

    def generate_yard_layout(self):
        """生成堆场布局定义"""
        areas = []
        area_configs = [
            ('A1', 20, 12, 7),
            ('A2', 20, 12, 7),
            ('B1', 18, 15, 6),
            ('B2', 18, 15, 6),
            ('C1', 25, 10, 8),
            ('C2', 25, 10, 8),
            ('D1', 15, 18, 5),
            ('R1', 10, 8, 4),
        ]
        for area_id, bays, rows, max_tiers in area_configs:
            areas.append({
                '区域编号': area_id,
                '贝位数量': bays,
                '列数': rows,
                '最大堆叠层数': max_tiers
            })
        return pd.DataFrame(areas)

    def generate_vessel_calls(self):
        """生成船舶靠泊记录"""
        vessels = []
        vessel_names = [
            '中远海运银河', '马士基埃德蒙顿', '地中海奥斯陆', '达飞泰特斯',
            '长赐轮', '海王星号', '东方海外香港', '阳明基隆',
            '万海春', '现代商船首尔', '太平新加坡', '以星纽约'
        ]
        berths = ['B1', 'B2', 'B3', 'B4']

        num_vessels = 300
        current_time = self.start_date

        for i in range(num_vessels):
            name = random.choice(vessel_names) + f'-{i+1:03d}'
            arrival = current_time + timedelta(
                hours=np.random.uniform(0.5, 12),
                minutes=np.random.randint(0, 60)
            )
            teu = int(np.random.normal(2500, 800))
            teu = max(400, min(teu, 6000))
            duration_hours = teu / (np.random.uniform(25, 40))
            departure = arrival + timedelta(hours=duration_hours)
            berth = random.choice(berths)

            vessels.append({
                '船名': name,
                '到港时间': arrival,
                '离港时间': departure,
                '载箱量TEU': teu,
                '分配泊位编号': berth
            })
            current_time = departure + timedelta(hours=np.random.uniform(0.1, 3))

        return pd.DataFrame(vessels)

    def generate_container_movements(self, vessel_df, yard_df):
        """生成集装箱流转记录"""
        containers = []
        box_types = ['20尺', '40尺']
        box_statuses = ['重箱', '空箱', '冷藏', '危品']
        status_weights = [0.65, 0.25, 0.07, 0.03]
        customers = ['马士基物流', '中远集运', '地中海航运', '达飞物流',
                     '东方海外', '长荣海运', '阳明海运', '万海航运', '太平船务']
        routes = ['欧洲-远东', '跨太平洋东行', '跨太平洋西行', '亚洲区内',
                  '澳新航线', '中东印巴', '南美航线', '非洲航线']

        area_list = yard_df['区域编号'].tolist()
        area_capacity = {}
        for _, row in yard_df.iterrows():
            area_capacity[row['区域编号']] = {
                'bays': row['贝位数量'],
                'rows': row['列数'],
                'tiers': row['最大堆叠层数']
            }

        total_containers = 120000
        date_range = pd.date_range(
            start=self.start_date,
            end=self.start_date + timedelta(days=self.days),
            freq='h'
        )

        daily_pattern = np.array([
            0.5, 0.3, 0.2, 0.15, 0.1, 0.15,
            0.4, 0.8, 1.2, 1.4, 1.5, 1.3,
            1.4, 1.5, 1.4, 1.3, 1.2, 1.1,
            1.0, 0.9, 0.8, 0.7, 0.6, 0.5
        ])
        daily_pattern = daily_pattern / daily_pattern.mean()

        container_id = 1
        for i in range(total_containers):
            hour_idx = np.random.randint(0, len(date_range) - 48)
            base_dt = date_range[hour_idx]
            hour_of_day = base_dt.hour
            factor = daily_pattern[hour_of_day]
            if np.random.random() > factor * 0.7:
                continue

            in_time = base_dt + timedelta(minutes=np.random.randint(0, 60))
            stay_days = max(1, int(np.random.exponential(4) + 2))
            if np.random.random() < 0.15:
                stay_days = np.random.randint(8, 25)
            out_time = in_time + timedelta(days=stay_days, hours=np.random.randint(0, 24))

            area = random.choice(area_list)
            cap = area_capacity[area]
            bay = np.random.randint(1, cap['bays'] + 1)
            row = np.random.randint(1, cap['rows'] + 1)
            tier = np.random.randint(1, cap['tiers'] + 1)

            btype = np.random.choice(box_types, p=[0.55, 0.45])
            bstatus = np.random.choice(box_statuses, p=status_weights)
            customer = random.choice(customers)
            route = random.choice(routes)

            prefix = random.choice(['MSKU', 'CBHU', 'MEDU', 'CMAU', 'OOLU', 'EGLV', 'YMLU', 'WHLU', 'PSSU'])
            box_no = f'{prefix}{container_id:08d}'

            containers.append({
                '箱号': box_no,
                '进场时间': in_time,
                '出场时间': out_time,
                '堆场区域': area,
                '贝位': bay,
                '列': row,
                '层': tier,
                '箱型': btype,
                '状态': bstatus,
                '客户': customer,
                '航线': route
            })
            container_id += 1

        result_df = pd.DataFrame(containers)
        result_df['贝位-列-层坐标'] = (
            result_df['贝位'].astype(str) + '-' +
            result_df['列'].astype(str) + '-' +
            result_df['层'].astype(str)
        )
        return result_df

    def generate_all(self, output_dir='data'):
        """生成所有数据并保存到CSV文件"""
        os.makedirs(output_dir, exist_ok=True)

        yard_df = self.generate_yard_layout()
        yard_df.to_csv(os.path.join(output_dir, '堆场布局定义.csv'), index=False, encoding='utf-8-sig')
        print(f'已生成堆场布局定义: {len(yard_df)} 条记录')

        vessel_df = self.generate_vessel_calls()
        vessel_df.to_csv(os.path.join(output_dir, '船舶靠泊记录.csv'), index=False, encoding='utf-8-sig')
        print(f'已生成船舶靠泊记录: {len(vessel_df)} 条记录')

        container_df = self.generate_container_movements(vessel_df, yard_df)
        cols = ['箱号', '进场时间', '出场时间', '堆场区域', '贝位-列-层坐标', '箱型', '状态', '客户', '航线']
        container_df[cols].to_csv(
            os.path.join(output_dir, '集装箱流转记录.csv'),
            index=False, encoding='utf-8-sig'
        )
        print(f'已生成集装箱流转记录: {len(container_df)} 条记录')

        return vessel_df, container_df, yard_df


def load_port_data(data_dir='data'):
    """加载港口数据"""
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in [
        '船舶靠泊记录.csv', '集装箱流转记录.csv', '堆场布局定义.csv'
    ]):
        print('数据文件不存在，正在生成模拟数据...')
        generator = PortDataGenerator()
        return generator.generate_all(data_dir)

    vessel_df = pd.read_csv(
        os.path.join(data_dir, '船舶靠泊记录.csv'),
        encoding='utf-8-sig',
        parse_dates=['到港时间', '离港时间']
    )
    container_df = pd.read_csv(
        os.path.join(data_dir, '集装箱流转记录.csv'),
        encoding='utf-8-sig',
        parse_dates=['进场时间', '出场时间']
    )
    yard_df = pd.read_csv(
        os.path.join(data_dir, '堆场布局定义.csv'),
        encoding='utf-8-sig'
    )
    print(f'已加载数据: 船舶{len(vessel_df)}条, 集装箱{len(container_df)}条, 堆场{len(yard_df)}个区域')
    return vessel_df, container_df, yard_df
