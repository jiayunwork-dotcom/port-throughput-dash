"""
港口集装箱吞吐量预测与堆场利用率优化分析面板
主应用入口 - Dash + Plotly交互分析面板
"""

import os
import sys
import traceback
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import dash_bootstrap_components as dbc
from dash.dash_table.Format import Format, Scheme, Symbol

from src.data_generator import load_port_data, PortDataGenerator
from src.data_processor import DataPreprocessor
from src.forecasting import ThroughputForecaster
from src.rehandling import YardStackSimulator, OverdueBoxAnalyzer
from src.whatif import WhatIfScenario
from src.report_generator import PortReportGenerator
from src.charts import ChartBuilder


APP_TITLE = '港口集装箱吞吐量预测与堆场利用率优化分析平台'
APP_PORT = 8050

vessel_df, container_df, yard_df = None, None, None
preprocessor = None
daily_throughput = None
yard_util_time = None
forecast_results_cache = {}
rehandling_results_cache = None
overdue_cache = None
kpi_cache = None


def initialize_data(data_dir='data'):
    """初始化数据"""
    global vessel_df, container_df, yard_df, preprocessor
    global daily_throughput, yard_util_time

    print('正在加载数据...')
    vessel_df, container_df, yard_df = load_port_data(data_dir)
    preprocessor = DataPreprocessor(vessel_df, container_df, yard_df)
    daily_throughput = preprocessor.get_daily_throughput()
    yard_util_time = preprocessor.calculate_yard_utilization_time()
    print(f'数据加载完成: {len(daily_throughput)} 天吞吐量记录')
    return True


try:
    initialize_data()
except Exception as e:
    print(f'数据初始化失败: {e}')
    traceback.print_exc()


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CERULEAN,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css'
    ],
    suppress_callback_exceptions=True,
    title=APP_TITLE
)

server = app.server


def make_header():
    """创建页面顶部导航栏"""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink('数据概览', href='#overview')),
            dbc.NavItem(dbc.NavLink('吞吐量预测', href='#forecast')),
            dbc.NavItem(dbc.NavLink('堆场分析', href='#yard')),
            dbc.NavItem(dbc.NavLink('翻箱率分析', href='#rehandling')),
            dbc.NavItem(dbc.NavLink('KPI面板', href='#kpi')),
            dbc.NavItem(dbc.NavLink('What-if分析', href='#whatif')),
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem('导出PDF报告', id='export-pdf-btn'),
                    dbc.DropdownMenuItem('重新生成模拟数据', id='regenerate-btn'),
                    dbc.DropdownMenuItem('上传自定义数据', id='upload-menu-btn'),
                ],
                nav=True,
                in_navbar=True,
                label='工具',
                align_end=True,
            ),
        ],
        brand=APP_TITLE,
        brand_href='#',
        color='primary',
        dark=True,
        fluid=True,
        className='mb-4 shadow-sm',
        style={'padding': '12px 24px'}
    )


def make_data_upload_section():
    """创建数据上传模块"""
    return dbc.Card([
        dbc.CardHeader([
            html.I(className='fas fa-database me-2'),
            '数据导入模块'
        ], className='bg-primary text-white'),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label('船舶靠泊记录 CSV', className='fw-bold'),
                    dcc.Upload(
                        id='upload-vessel',
                        children=html.Div([
                            html.I(className='fas fa-ship me-2'),
                            '拖拽或 点击上传 船舶靠泊记录.csv'
                        ]),
                        className='border border-secondary border-dashed rounded p-3 text-center text-secondary',
                        accept='.csv',
                        max_size=-1
                    ),
                    html.Div(id='vessel-upload-status', className='mt-2 small text-success')
                ], md=4),
                dbc.Col([
                    html.Label('集装箱流转记录 CSV', className='fw-bold'),
                    dcc.Upload(
                        id='upload-container',
                        children=html.Div([
                            html.I(className='fas fa-box me-2'),
                            '拖拽或 点击上传 集装箱流转记录.csv'
                        ]),
                        className='border border-secondary border-dashed rounded p-3 text-center text-secondary',
                        accept='.csv',
                        max_size=-1
                    ),
                    html.Div(id='container-upload-status', className='mt-2 small text-success')
                ], md=4),
                dbc.Col([
                    html.Label('堆场布局定义 CSV', className='fw-bold'),
                    dcc.Upload(
                        id='upload-yard',
                        children=html.Div([
                            html.I(className='fas fa-warehouse me-2'),
                            '拖拽或 点击上传 堆场布局定义.csv'
                        ]),
                        className='border border-secondary border-dashed rounded p-3 text-center text-secondary',
                        accept='.csv',
                        max_size=-1
                    ),
                    html.Div(id='yard-upload-status', className='mt-2 small text-success')
                ], md=4),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Div(id='data-summary', className='d-flex justify-content-around flex-wrap')
                ])
            ])
        ])
    ], className='mb-4')


def make_overview_section():
    """创建数据概览模块"""
    return html.Div(id='overview', children=[
        html.H3([html.I(className='fas fa-chart-line me-2 text-primary'), '吞吐量统计概览'],
                className='mt-5 mb-3 border-bottom pb-2'),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6('总进场箱量', className='text-muted small'),
                    html.H3(id='kpi-total-in', className='text-primary fw-bold'),
                    html.Small(id='kpi-total-in-sub', className='text-muted')
                ])
            ], className='shadow-sm border-0'), md=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6('总出场箱量', className='text-muted small'),
                    html.H3(id='kpi-total-out', className='text-success fw-bold'),
                    html.Small(id='kpi-total-out-sub', className='text-muted')
                ])
            ], className='shadow-sm border-0'), md=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6('总TEU', className='text-muted small'),
                    html.H3(id='kpi-total-teu', className='text-warning fw-bold'),
                    html.Small(id='kpi-total-teu-sub', className='text-muted')
                ])
            ], className='shadow-sm border-0'), md=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.H6('日均TEU', className='text-muted small'),
                    html.H3(id='kpi-avg-daily-teu', className='text-info fw-bold'),
                    html.Small(id='kpi-stats-sub', className='text-muted')
                ])
            ], className='shadow-sm border-0'), md=3),
        ], className='g-3 mb-4'),

        dbc.Card([
            dbc.CardHeader('每日吞吐量趋势图'),
            dbc.CardBody([
                dcc.Graph(id='throughput-chart')
            ])
        ], className='mb-4')
    ])


def make_forecast_section():
    """创建吞吐量预测模块"""
    return html.Div(id='forecast', children=[
        html.H3([html.I(className='fas fa-brain me-2 text-primary'), '吞吐量时序预测'],
                className='mt-5 mb-3 border-bottom pb-2'),

        dbc.Card([
            dbc.CardHeader('预测参数设置'),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('预测算法'),
                        dcc.Dropdown(
                            id='forecast-algorithm',
                            options=[
                                {'label': 'ARIMA (AIC自动定阶 + 7天季节性)', 'value': 'ARIMA'},
                                {'label': 'Prophet (节假日+周+年周期)', 'value': 'Prophet'},
                                {'label': '简化LSTM (单层128单元, 30天窗口)', 'value': 'LSTM'},
                                {'label': '三种算法对比', 'value': 'ALL'}
                            ],
                            value='ALL',
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label('预测粒度'),
                        dcc.Dropdown(
                            id='forecast-granularity',
                            options=[
                                {'label': '日', 'value': '日'},
                                {'label': '周', 'value': '周'}
                            ],
                            value='日',
                            clearable=False
                        )
                    ], md=2),
                    dbc.Col([
                        html.Label('历史训练长度'),
                        dcc.Dropdown(
                            id='forecast-train-days',
                            options=[
                                {'label': '30天', 'value': 30},
                                {'label': '60天', 'value': 60},
                                {'label': '90天', 'value': 90}
                            ],
                            value=60,
                            clearable=False
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label('预测未来天数'),
                        dcc.Dropdown(
                            id='forecast-steps',
                            options=[
                                {'label': '7天', 'value': 7},
                                {'label': '14天', 'value': 14},
                                {'label': '30天', 'value': 30}
                            ],
                            value=7,
                            clearable=False
                        )
                    ], md=3),
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className='fas fa-rocket me-2'), '运行预测模型'],
                            id='run-forecast-btn', color='primary', className='mt-3'
                        ),
                        dcc.Loading(id='forecast-loading', type='default', children=[])
                    ])
                ])
            ])
        ], className='mb-4'),

        dbc.Row(id='forecast-metrics-row', className='g-3 mb-4'),

        dbc.Card([
            dbc.CardHeader('实际值 vs 预测值 对比（含80%置信区间）'),
            dbc.CardBody([
                dcc.Graph(id='forecast-chart')
            ])
        ], className='mb-4'),

        dbc.Row(id='residuals-row', className='g-3 mb-4'),
    ])


def make_yard_section():
    """创建堆场利用率分析模块"""
    areas = yard_df['区域编号'].tolist() if yard_df is not None else []
    return html.Div(id='yard', children=[
        html.H3([html.I(className='fas fa-warehouse me-2 text-primary'), '堆场利用率分析'],
                className='mt-5 mb-3 border-bottom pb-2'),

        dbc.Card([
            dbc.CardHeader([
                html.I(className='fas fa-layer-group me-2'),
                '堆场俯视热力图（时间轴回放）'
            ]),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('选择堆场区域'),
                        dcc.Dropdown(
                            id='yard-area-select',
                            options=[{'label': f'区域 {a}', 'value': a} for a in areas],
                            value=areas[0] if areas else None,
                            clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label('快照时间'),
                        dcc.Slider(
                            id='yard-time-slider',
                            min=0, max=100, value=50,
                            marks={0: '最早', 25: '1/4', 50: '中期', 75: '3/4', 100: '最新'},
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        ),
                        html.Div(id='yard-snapshot-time', className='text-center mt-2 text-primary fw-bold')
                    ], md=8),
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='yard-heatmap', className='mt-3'),
                    ], md=9),
                    dbc.Col([
                        html.H6('快照统计', className='mt-3 border-bottom pb-2'),
                        html.Div(id='yard-snapshot-stats', className='mt-3'),
                        html.Hr(),
                        html.H6('区域信息', className='border-bottom pb-2'),
                        dash_table.DataTable(
                            id='yard-area-info',
                            columns=[
                                {'name': '属性', 'id': '属性'},
                                {'name': '值', 'id': '值'}
                            ],
                            style_cell={'fontSize': '12px', 'padding': '8px'},
                            style_header={'fontWeight': 'bold',
                                          'backgroundColor': '#f8f9fa'},
                            style_as_list_view=True
                        )
                    ], md=3),
                ])
            ])
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader([
                html.I(className='fas fa-chart-area me-2'),
                '堆场各区域利用率趋势'
            ]),
            dbc.CardBody([
                dcc.Graph(id='yard-utilization-chart')
            ])
        ], className='mb-4'),

        html.H4([html.I(className='fas fa-exclamation-triangle me-2 text-warning'),
                 '超期箱统计分析'], className='mt-4 mb-3 border-bottom pb-2'),

        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('免费堆存期（天）'),
                        dcc.Input(
                            id='free-days-input',
                            type='number', value=5, min=1, max=30, step=1,
                            className='form-control'
                        ),
                        html.Br(),
                        dbc.Button(
                            [html.I(className='fas fa-calculator me-2'), '计算超期箱'],
                            id='calc-overdue-btn', color='warning', outline=True
                        )
                    ], md=3),
                    dbc.Col([
                        html.Div(id='overdue-summary-cards', className='d-flex flex-column gap-2')
                    ], md=9),
                ])
            ])
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('TOP 客户超期箱排行'),
                    dbc.CardBody([
                        dcc.Graph(id='overdue-by-customer-chart'),
                        dash_table.DataTable(
                            id='overdue-customer-table',
                            page_size=8,
                            style_table={'overflowX': 'auto'},
                            style_cell={'fontSize': '12px', 'padding': '6px'},
                            style_header={'fontWeight': 'bold',
                                          'backgroundColor': '#f8f9fa'},
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'},
                                 'backgroundColor': '#f9f9f9'}
                            ]
                        )
                    ])
                ], className='h-100')
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader('各区域超期箱分布'),
                    dbc.CardBody([
                        dcc.Graph(id='overdue-by-area-chart'),
                        dash_table.DataTable(
                            id='overdue-area-table',
                            page_size=8,
                            style_table={'overflowX': 'auto'},
                            style_cell={'fontSize': '12px', 'padding': '6px'},
                            style_header={'fontWeight': 'bold',
                                          'backgroundColor': '#f8f9fa'}
                        )
                    ])
                ], className='h-100')
            ], md=6),
        ], className='g-3 mb-4'),
    ])


def make_rehandling_section():
    """创建翻箱率分析模块"""
    return html.Div(id='rehandling', children=[
        html.H3([html.I(className='fas fa-dice me-2 text-primary'),
                 '翻箱率蒙特卡洛模拟分析'],
                className='mt-5 mb-3 border-bottom pb-2'),

        dbc.Card([
            dbc.CardHeader('模拟参数设置'),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('蒙特卡洛模拟次数'),
                        dcc.Dropdown(
                            id='sim-count',
                            options=[
                                {'label': '500次', 'value': 500},
                                {'label': '1000次（推荐）', 'value': 1000},
                                {'label': '2000次', 'value': 2000},
                            ],
                            value=1000, clearable=False
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label('堆场填充率'),
                        dcc.Slider(id='fill-ratio', min=0.3, max=0.95, step=0.05, value=0.7,
                                   marks={0.3: '30%', 0.5: '50%', 0.7: '70%', 0.9: '90%'},
                                   tooltip={'placement': 'bottom', 'always_visible': True})
                    ], md=5),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className='fas fa-play-circle me-2'),
                             '运行蒙特卡洛模拟'],
                            id='run-simulation-btn', color='info', className='mt-4'
                        )
                    ], md=3, className='d-flex align-items-end justify-content-end'),
                ]),
                dcc.Loading(id='sim-loading', type='default', children=[])
            ])
        ], className='mb-4'),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([html.I(className='fas fa-random me-2 text-muted'), '策略A'],
                             className='text-muted small mb-1'),
                    html.H6('随机堆放', className='fw-bold mb-2'),
                    html.Small('箱子随机分配到可用位置，无优化策略'),
                    html.Hr(),
                    html.Div([
                        html.Span('平均翻箱率: ', className='text-muted'),
                        html.Span(id='strat-a-avg', className='fw-bold text-primary')
                    ]),
                    html.Div([
                        html.Span('P95翻箱率: ', className='text-muted'),
                        html.Span(id='strat-a-p95', className='fw-bold text-secondary')
                    ])
                ])
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([html.I(className='fas fa-sort-amount-down me-2 text-success'),
                              '策略B'], className='text-success small mb-1'),
                    html.H6('按提箱时间排序', className='fw-bold mb-2'),
                    html.Small('预计早提箱的放在上层，减少翻箱'),
                    html.Hr(),
                    html.Div([
                        html.Span('平均翻箱率: ', className='text-muted'),
                        html.Span(id='strat-b-avg', className='fw-bold text-primary')
                    ]),
                    html.Div([
                        html.Span('P95翻箱率: ', className='text-muted'),
                        html.Span(id='strat-b-p95', className='fw-bold text-secondary')
                    ])
                ])
            ]), md=4),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([html.I(className='fas fa-object-group me-2 text-info'), '策略C'],
                             className='text-info small mb-1'),
                    html.H6('按航线聚堆', className='fw-bold mb-2'),
                    html.Small('同航线箱子集中堆放同列，提升作业效率'),
                    html.Hr(),
                    html.Div([
                        html.Span('平均翻箱率: ', className='text-muted'),
                        html.Span(id='strat-c-avg', className='fw-bold text-primary')
                    ]),
                    html.Div([
                        html.Span('P95翻箱率: ', className='text-muted'),
                        html.Span(id='strat-c-p95', className='fw-bold text-secondary')
                    ])
                ])
            ]), md=4),
        ], className='g-3 mb-4'),

        dbc.Card([
            dbc.CardHeader('翻箱率策略对比图'),
            dbc.CardBody([
                dcc.Graph(id='rehandling-chart')
            ])
        ], className='mb-4'),

        dbc.Card([
            dbc.CardHeader('各区域翻箱率明细'),
            dbc.CardBody([
                dash_table.DataTable(
                    id='rehandling-detail-table',
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={'fontSize': '12px', 'padding': '8px'},
                    style_header={'fontWeight': 'bold',
                                  'backgroundColor': '#f8f9fa'},
                    style_data_conditional=[
                        {'if': {'filter_query': '{策略} contains "策略B"'},
                         'backgroundColor': '#e6fffa'}
                    ]
                )
            ])
        ], className='mb-4'),
    ])


def make_kpi_section():
    """创建KPI指标面板"""
    return html.Div(id='kpi', children=[
        html.H3([html.I(className='fas fa-tachometer-alt me-2 text-primary'),
                 'KPI运营指标仪表板'],
                className='mt-5 mb-3 border-bottom pb-2'),

        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label('聚合粒度'),
                        dcc.Dropdown(
                            id='kpi-granularity',
                            options=[
                                {'label': '日', 'value': '日'},
                                {'label': '周', 'value': '周'},
                                {'label': '月', 'value': '月'}
                            ],
                            value='日', clearable=False
                        )
                    ], md=3),
                ])
            ])
        ], className='mb-4'),

        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className='fas fa-ship me-2 text-info'),
                        html.Span('船时效率', className='text-muted small')
                    ]),
                    html.H3(id='kpi-vessel-hour', className='text-info fw-bold mt-1'),
                    html.Small('TEU / 靠泊小时', className='text-muted'),
                    html.Hr(),
                    html.Div(id='kpi-vessel-hour-trend', className='small')
                ])
            ]), md=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className='fas fa-cogs me-2 text-success'),
                        html.Span('桥吊效率', className='text-muted small')
                    ]),
                    html.H3(id='kpi-crane-hour', className='text-success fw-bold mt-1'),
                    html.Small('TEU / 台时', className='text-muted'),
                    html.Hr(),
                    html.Div(id='kpi-crane-hour-trend', className='small')
                ])
            ]), md=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className='fas fa-anchor me-2 text-warning'),
                        html.Span('泊位利用率', className='text-muted small')
                    ]),
                    html.H3(id='kpi-berth-util', className='text-warning fw-bold mt-1'),
                    html.Small('占用小时 / 可用小时', className='text-muted'),
                    html.Hr(),
                    html.Div(id='kpi-berth-util-trend', className='small')
                ])
            ]), md=3),
            dbc.Col(dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className='fas fa-truck me-2 text-purple'),
                        html.Span('集卡周转时间', className='text-muted small')
                    ]),
                    html.H3(id='kpi-truck-time', className='fw-bold mt-1',
                            style={'color': '#805ad5'}),
                    html.Small('进闸 → 堆场落地 (小时)', className='text-muted'),
                    html.Hr(),
                    html.Div(id='kpi-truck-time-trend', className='small')
                ])
            ]), md=3),
        ], className='g-3 mb-4'),

        dbc.Card([
            dbc.CardHeader([
                html.I(className='fas fa-chart-line me-2'),
                'KPI趋势（连续3天低于均值-1σ标红告警）'
            ]),
            dbc.CardBody([
                dcc.Graph(id='kpi-trend-chart')
            ])
        ], className='mb-4'),
    ])


def make_whatif_section():
    """创建What-if情景分析模块"""
    return html.Div(id='whatif', children=[
        html.H3([html.I(className='fas fa-sliders-h me-2 text-primary'),
                 'What-if 情景模拟分析'],
                className='mt-5 mb-3 border-bottom pb-2'),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-boxes me-2 text-primary'),
                        '情景1：吞吐量增长模拟'
                    ]),
                    dbc.CardBody([
                        html.Label('箱量增长幅度'),
                        dcc.Dropdown(
                            id='whatif-growth-rate',
                            options=[
                                {'label': '+10%', 'value': 0.1},
                                {'label': '+20%', 'value': 0.2},
                                {'label': '+30%', 'value': 0.3}
                            ],
                            value=0.2, clearable=False
                        ),
                        html.Hr(),
                        dbc.Button(
                            [html.I(className='fas fa-play me-2'), '运行情景1'],
                            id='run-whatif-1', color='primary', outline=True, size='sm'
                        )
                    ])
                ], className='h-100')
            ], md=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-warehouse me-2 text-success'),
                        '情景2：新增堆场区域'
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label('贝位数'),
                                dcc.Input(id='new-bays', type='number',
                                          value=10, min=5, max=30, className='form-control')
                            ], md=4),
                            dbc.Col([
                                html.Label('列数'),
                                dcc.Input(id='new-rows', type='number',
                                          value=8, min=3, max=20, className='form-control')
                            ], md=4),
                            dbc.Col([
                                html.Label('层数'),
                                dcc.Input(id='new-tiers', type='number',
                                          value=6, min=2, max=10, className='form-control')
                            ], md=4),
                        ]),
                        html.Hr(),
                        dbc.Button(
                            [html.I(className='fas fa-play me-2'), '运行情景2'],
                            id='run-whatif-2', color='success', outline=True, size='sm'
                        )
                    ])
                ], className='h-100')
            ], md=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-hourglass-half me-2 text-warning'),
                        '情景3：缩短堆存天数'
                    ]),
                    dbc.CardBody([
                        html.Label('平均堆存天数减少'),
                        dcc.Dropdown(
                            id='whatif-reduce-days',
                            options=[
                                {'label': '减少1天', 'value': 1},
                                {'label': '减少2天', 'value': 2},
                                {'label': '减少3天', 'value': 3}
                            ],
                            value=2, clearable=False
                        ),
                        html.Hr(),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className='fas fa-play me-2'), '运行情景3'],
                                    id='run-whatif-3', color='warning', outline=True, size='sm'
                                )
                            ]),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className='fas fa-bolt me-2'), '一键运行全部'],
                                    id='run-whatif-all', color='danger', size='sm'
                                )
                            ])
                        ])
                    ])
                ], className='h-100')
            ], md=4),
        ], className='g-3 mb-4'),

        dcc.Loading(id='whatif-loading', type='default', children=[]),

        dbc.Card([
            dbc.CardHeader('情景分析结果对比'),
            dbc.CardBody([
                dcc.Graph(id='whatif-chart'),
                html.Hr(),
                dash_table.DataTable(
                    id='whatif-result-table',
                    page_size=8,
                    style_table={'overflowX': 'auto'},
                    style_cell={'fontSize': '12px', 'padding': '8px',
                                'whiteSpace': 'normal', 'height': 'auto'},
                    style_header={'fontWeight': 'bold',
                                  'backgroundColor': '#f8f9fa'},
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'},
                         'backgroundColor': '#f9f9f9'}
                    ]
                )
            ])
        ], className='mb-4'),

        html.Div([
            html.Hr(),
            html.H6(id='export-status', className='text-muted text-center'),
            dcc.Download(id='download-pdf')
        ], className='mb-5'),
    ])


def make_footer():
    """创建页脚"""
    return html.Footer([
        html.Div([
            html.P([
                '© 2026 港口智能运营系统 | ',
                '基于 Dash + Plotly + PyTorch 构建 | ',
                html.Small(f'数据更新: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
            ], className='text-center text-muted small')
        ], className='container py-4 border-top')
    ])


app.layout = html.Div([
    make_header(),
    dcc.Store(id='app-data-store'),
    dbc.Container([
        make_data_upload_section(),
        make_overview_section(),
        make_forecast_section(),
        make_yard_section(),
        make_rehandling_section(),
        make_kpi_section(),
        make_whatif_section(),
    ], fluid=True, className='px-4'),
    make_footer()
])

from callbacks import register_callbacks
register_callbacks(app)

if __name__ == '__main__':
    print(f'=' * 70)
    print(f'{APP_TITLE}')
    print(f'启动地址: http://localhost:{APP_PORT}')
    print(f'=' * 70)
    app.run_server(debug=True, host='0.0.0.0', port=APP_PORT)

