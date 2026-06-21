"""
Dash回调函数 - 处理所有交互逻辑（修复版 v2）
核心修复：
1. 统一使用 src.state 模块管理全局状态，解决跨模块变量不共享问题
2. init-trigger 组件保证页面加载后自动触发所有初始渲染
3. 所有回调加入异常捕获+打印，避免静默失败
4. 按钮判断统一采用 n_clicks is not None and n_clicks > 0
"""

import os
import io
import base64
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table, no_update
from dash.dash_table.Format import Format, Scheme
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src import state as S
from src.data_generator import PortDataGenerator
from src.data_processor import DataPreprocessor
from src.forecasting import ThroughputForecaster
from src.rehandling import YardStackSimulator, OverdueBoxAnalyzer
from src.whatif import WhatIfScenario
from src.report_generator import PortReportGenerator
from src.charts import ChartBuilder
from src.simulation import ShipQueueSimulator, STRATEGY_FCFS, STRATEGY_SJF, STRATEGY_LWF, STRATEGY_ALL


def empty_figure(title='暂无数据，请先加载数据或运行相应模块'):
    """生成统一的空占位图"""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref='paper', yref='paper',
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color='#a0aec0')
    )
    fig.update_layout(
        height=400,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='white', paper_bgcolor='white'
    )
    return fig


def _debug(msg):
    """调试打印 - 方便排查回调是否触发"""
    print(f'[callback] {msg}', flush=True)


def register_callbacks(app):
    """注册所有回调函数"""

    # =============================================
    # 回调1: 概览区 + 吞吐量趋势图 + 堆场利用率趋势
    # 触发: init-trigger (初始化) + app-data-store (数据变化)
    # =============================================
    @app.callback(
        Output('data-summary', 'children'),
        Output('throughput-chart', 'figure'),
        Output('kpi-total-in', 'children'),
        Output('kpi-total-out', 'children'),
        Output('kpi-total-teu', 'children'),
        Output('kpi-avg-daily-teu', 'children'),
        Output('kpi-total-in-sub', 'children'),
        Output('kpi-total-out-sub', 'children'),
        Output('kpi-total-teu-sub', 'children'),
        Output('kpi-stats-sub', 'children'),
        Output('yard-utilization-chart', 'figure'),
        [Input('init-trigger', 'n_intervals'),
         Input('app-data-store', 'data')]
    )
    def update_overview(n_intervals, _store):
        """更新概览区数据 - 初始加载+数据变化时自动触发"""
        _debug(f'update_overview触发 n_intervals={n_intervals} initialized={S.is_initialized()}')

        if not S.is_initialized():
            ef = empty_figure('系统初始化中，请稍候...')
            return (['数据加载中...'], ef, '-', '-', '-', '-', '', '', '', '', ef)

        df = S.daily_throughput
        total_in = int(df['进场箱量'].sum())
        total_out = int(df['出场箱量'].sum())
        total_teu = float(df['总TEU'].sum())
        days = len(df)
        avg_daily_teu = total_teu / max(1, days)

        summary = [
            dbc.Badge([
                html.I(className='fas fa-ship me-1'),
                f'船舶靠泊记录: {len(S.vessel_df):,} 条'
            ], color='primary', className='me-2 mb-2', style={'fontSize': '14px'}),
            dbc.Badge([
                html.I(className='fas fa-box me-1'),
                f'集装箱流转: {len(S.container_df):,} 条'
            ], color='success', className='me-2 mb-2', style={'fontSize': '14px'}),
            dbc.Badge([
                html.I(className='fas fa-warehouse me-1'),
                f'堆场区域: {len(S.yard_df)} 个'
            ], color='info', className='me-2 mb-2', style={'fontSize': '14px'}),
            dbc.Badge([
                html.I(className='fas fa-calendar me-1'),
                f'数据跨度: {df["日期"].min().strftime("%Y-%m-%d")} ~ {df["日期"].max().strftime("%Y-%m-%d")}'
            ], color='warning', className='me-2 mb-2', style={'fontSize': '14px'}),
        ]

        throughput_fig = ChartBuilder.create_throughput_chart(df)
        yard_util_fig = ChartBuilder.create_yard_utilization_chart(S.yard_util_time)

        _debug(f'update_overview完成: {days}天数据, 总TEU {total_teu:,.0f}')

        return (
            summary, throughput_fig,
            f'{total_in:,}', f'{total_out:,}', f'{total_teu:,.0f}', f'{avg_daily_teu:,.0f}',
            f'占总吞吐 {total_in/(total_in+total_out)*100:.1f}%',
            f'占总吞吐 {total_out/(total_in+total_out)*100:.1f}%',
            f'总量 {total_in+total_out:,} 箱',
            f'共 {days} 天数据' if days > 1 else '单日数据',
            yard_util_fig
        )

    # =============================================
    # 辅助: 解析上传的CSV
    # =============================================
    def _parse_upload(contents, filename):
        """解析上传的CSV文件"""
        if contents is None:
            return None
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            if '.csv' in filename.lower():
                return pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
        except Exception as e:
            print(f'文件解析失败 {filename}: {e}')
            traceback.print_exc()
        return None

    # =============================================
    # 回调2: 数据上传 + 重新生成模拟数据
    # =============================================
    @app.callback(
        Output('vessel-upload-status', 'children'),
        Output('container-upload-status', 'children'),
        Output('yard-upload-status', 'children'),
        Output('app-data-store', 'data', allow_duplicate=True),
        [Input('upload-vessel', 'contents'),
         Input('upload-container', 'contents'),
         Input('upload-yard', 'contents'),
         Input('regenerate-btn', 'n_clicks')],
        [State('upload-vessel', 'filename'),
         State('upload-container', 'filename'),
         State('upload-yard', 'filename')],
        prevent_initial_call='initial_duplicate'
    )
    def handle_upload_and_regenerate(v_c, c_c, y_c, regen_n, v_f, c_f, y_f):
        """处理数据上传和重新生成"""
        _debug(f'handle_upload_and_regenerate触发 regen_n={regen_n}')

        ctx = callback_context
        triggered = ctx.triggered_id

        # --- 分支A: 点击重新生成按钮 ---
        if triggered == 'regenerate-btn' and regen_n is not None and regen_n > 0:
            try:
                gen = PortDataGenerator(seed=np.random.randint(1, 9999))
                S.vessel_df, S.container_df, S.yard_df = gen.generate_all('data')
                S.preprocessor = DataPreprocessor(S.vessel_df, S.container_df, S.yard_df)
                S.daily_throughput = S.preprocessor.get_daily_throughput()
                S.yard_util_time = S.preprocessor.calculate_yard_utilization_time()
                S.forecast_results_cache.clear()
                S.rehandling_results_cache = None
                S.overdue_cache = None
                S.kpi_cache = None
                S.set_initialized(True)
                new_store = {'regenerated': True, 'ts': datetime.now().isoformat()}
                _debug('重新生成数据完成')
                return ('✓ 使用新模拟数据', '✓ 使用新模拟数据', '✓ 使用新模拟数据', new_store)
            except Exception as e:
                traceback.print_exc()
                return (f'✗ 失败: {str(e)[:30]}', '', '', no_update)

        # --- 分支B: 数据文件上传 ---
        statuses = ['', '', '']
        changed = False

        try:
            if v_c is not None and v_f:
                df = _parse_upload(v_c, v_f)
                if df is not None:
                    S.vessel_df = df
                    statuses[0] = f'✓ 已加载 {len(df)} 条记录'
                    changed = True
                else:
                    statuses[0] = '✗ 解析失败'
        except Exception as e:
            statuses[0] = f'✗ {str(e)[:30]}'

        try:
            if c_c is not None and c_f:
                df = _parse_upload(c_c, c_f)
                if df is not None:
                    S.container_df = df
                    statuses[1] = f'✓ 已加载 {len(df)} 条记录'
                    changed = True
                else:
                    statuses[1] = '✗ 解析失败'
        except Exception as e:
            statuses[1] = f'✗ {str(e)[:30]}'

        try:
            if y_c is not None and y_f:
                df = _parse_upload(y_c, y_f)
                if df is not None:
                    S.yard_df = df
                    statuses[2] = f'✓ 已加载 {len(df)} 个区域'
                    changed = True
                else:
                    statuses[2] = '✗ 解析失败'
        except Exception as e:
            statuses[2] = f'✗ {str(e)[:30]}'

        # 上传完成后重建预处理
        if changed and S.vessel_df is not None and S.container_df is not None and S.yard_df is not None:
            try:
                S.preprocessor = DataPreprocessor(S.vessel_df, S.container_df, S.yard_df)
                S.daily_throughput = S.preprocessor.get_daily_throughput()
                S.yard_util_time = S.preprocessor.calculate_yard_utilization_time()
                S.forecast_results_cache.clear()
                S.rehandling_results_cache = None
                S.overdue_cache = None
                S.kpi_cache = None
                S.set_initialized(True)
                new_store = {'uploaded': True, 'ts': datetime.now().isoformat()}
                _debug('上传数据并预处理完成')
                return (statuses[0], statuses[1], statuses[2], new_store)
            except Exception as e:
                traceback.print_exc()
                return (statuses[0], statuses[1], f'✗ 预处理失败: {str(e)[:30]}', no_update)

        # 默认: 无变化 (首次触发时)
        return (statuses[0], statuses[1], statuses[2], no_update)

    # =============================================
    # 回调3: 运行吞吐量预测
    # =============================================
    @app.callback(
        Output('forecast-chart', 'figure'),
        Output('forecast-metrics-row', 'children'),
        Output('residuals-row', 'children'),
        Output('forecast-loading', 'children'),
        [Input('run-forecast-btn', 'n_clicks')],
        [State('forecast-algorithm', 'value'),
         State('forecast-granularity', 'value'),
         State('forecast-train-days', 'value'),
         State('forecast-steps', 'value')],
        prevent_initial_call=False
    )
    def run_forecast(n_clicks, algorithm, granularity, train_days, steps):
        """运行吞吐量预测"""
        _debug(f'run_forecast触发 n_clicks={n_clicks}, algo={algorithm}')

        ef = empty_figure('点击"运行预测模型"按钮开始时序预测')

        if not S.is_initialized():
            return ef, [], [], []

        # 初始加载: 展示空占位
        if n_clicks is None or n_clicks == 0:
            return ef, [], [], []

        # 按钮点击: 执行预测
        try:
            series = S.preprocessor.get_throughput_series(
                granularity=granularity, teu_mode=False
            )

            algos = ['ARIMA', 'Prophet', 'LSTM'] if algorithm == 'ALL' else [algorithm]
            results = {}

            for algo in algos:
                try:
                    forecaster = ThroughputForecaster(algorithm=algo)
                    forecaster.fit(series, train_days=train_days)
                    result = forecaster.forecast(steps=steps, confidence=0.8)
                    results[algo] = result
                    _debug(f'  {algo} 完成: MAPE={result.get("mape", "?")}')
                except Exception as e:
                    _debug(f'  {algo} 失败: {e}')
                    traceback.print_exc()
                    results[algo] = {'error': str(e)}

            S.forecast_results_cache = results

            fig = ChartBuilder.create_forecast_comparison(results)

            cards = []
            for algo, res in results.items():
                if isinstance(res, dict) and 'error' not in res:
                    mape = res.get('mape', np.nan)
                    rmse = res.get('rmse', np.nan)
                    mape_str = f'{mape:.2f}%' if not np.isnan(mape) else '-'
                    rmse_str = f'{rmse:,.0f}' if not np.isnan(rmse) else '-'

                    if not np.isnan(mape) and mape < 10:
                        color = 'success'
                    elif not np.isnan(mape) and mape < 20:
                        color = 'warning'
                    else:
                        color = 'danger'

                    cards.append(dbc.Col(dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.Span(algo, className='fw-bold'),
                                dbc.Badge('MAPE', color=color,
                                          className='float-end', pill=True)
                            ], className='mb-2'),
                            html.Div([
                                html.Span(mape_str,
                                          className=f'text-{color} fw-bold',
                                          style={'fontSize': '28px'})
                            ]),
                            html.Small(f'RMSE: {rmse_str} 箱', className='text-muted'),
                            html.Br(),
                            html.Small('平均绝对百分比误差', className='text-muted')
                        ])
                    ], className='shadow-sm'), md=3))

            residuals_figs = []
            for algo, res in results.items():
                if isinstance(res, dict) and 'residuals' in res:
                    rfig = ChartBuilder.create_residual_histogram(res.get('residuals'), algo)
                    residuals_figs.append(dbc.Col([
                        dbc.Card([
                            dbc.CardBody([dcc.Graph(figure=rfig)])
                        ], className='shadow-sm')
                    ], md=4))

            _debug(f'预测完成: {len(results)} 个算法')
            return fig, cards, residuals_figs, []

        except Exception as e:
            traceback.print_exc()
            return empty_figure(f'预测失败: {str(e)[:50]}'), [], [], []

    # =============================================
    # 回调4: 堆场热力图 - 切换区域+拖动滑块立即响应
    # =============================================
    @app.callback(
        Output('yard-heatmap', 'figure'),
        Output('yard-snapshot-time', 'children'),
        Output('yard-snapshot-stats', 'children'),
        Output('yard-area-info', 'data'),
        [Input('yard-area-select', 'value'),
         Input('yard-time-slider', 'value'),
         Input('init-trigger', 'n_intervals')]
    )
    def update_yard_heatmap(area_id, slider_val, n_intervals):
        """更新堆场热力图 - 下拉/滑块变化立即响应"""
        _debug(f'update_yard_heatmap触发 area={area_id}, slider={slider_val}, '
               f'n_intervals={n_intervals}, init={S.is_initialized()}')

        ef = empty_figure('请选择堆场区域并稍候')

        if not S.is_initialized() or area_id is None:
            return ef, '-', [], []

        try:
            date_min = pd.to_datetime(S.container_df['进场时间'].min())
            date_max = pd.to_datetime(S.container_df['出场时间'].max())
            total_seconds = (date_max - date_min).total_seconds()

            if slider_val is None:
                slider_val = 50
            if total_seconds <= 0:
                total_seconds = 1

            target_ts = date_min + timedelta(seconds=total_seconds * (slider_val / 100))

            snapshot = S.preprocessor.get_yard_snapshot(target_ts)
            heatmap, bay_labels, row_labels = S.preprocessor.build_heatmap_data(
                snapshot, area_id
            )

            yard_row = S.yard_df[S.yard_df['区域编号'] == area_id]
            if len(yard_row) == 0:
                return ef, str(target_ts), [], []

            yr = yard_row.iloc[0]
            max_tiers = int(yr['最大堆叠层数'])
            area_snap = snapshot[snapshot['堆场区域'] == area_id]
            total_boxes = len(area_snap)
            cap = int(yr['贝位数量']) * int(yr['列数']) * max_tiers
            util = total_boxes / cap * 100 if cap > 0 else 0

            heavy = len(area_snap[area_snap['状态'] == '重箱'])
            empty_cnt = len(area_snap[area_snap['状态'] == '空箱'])
            reefer = len(area_snap[area_snap['状态'] == '冷藏'])
            danger = len(area_snap[area_snap['状态'] == '危品'])

            fig = ChartBuilder.create_yard_heatmap(
                heatmap, bay_labels, row_labels, max_tiers, area_id
            )
            time_str = target_ts.strftime('%Y-%m-%d %H:%M') + f' (进度{slider_val}%)'

            stats = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Small('占用箱数', className='text-muted'),
                        html.H6(f'{total_boxes:,}', className='fw-bold text-primary')
                    ]),
                    dbc.Col([
                        html.Small('占用率', className='text-muted'),
                        html.H6(f'{util:.1f}%', className='fw-bold text-success')
                    ]),
                ], className='mb-2'),
                html.Hr(className='my-2'),
                dbc.Row([
                    dbc.Col([
                        html.Small('重箱', className='text-muted'),
                        html.H6(f'{heavy}', className='text-secondary')
                    ]),
                    dbc.Col([
                        html.Small('空箱', className='text-muted'),
                        html.H6(f'{empty_cnt}', className='text-secondary')
                    ]),
                ], className='mb-1'),
                dbc.Row([
                    dbc.Col([
                        html.Small('冷藏', className='text-muted'),
                        html.H6(f'{reefer}', className='text-info')
                    ]),
                    dbc.Col([
                        html.Small('危品', className='text-muted'),
                        html.H6(f'{danger}', className='text-danger')
                    ]),
                ])
            ])

            area_info = [
                {'属性': '区域编号', '值': area_id},
                {'属性': '贝位数量', '值': f"{int(yr['贝位数量'])} 个"},
                {'属性': '列数', '值': f"{int(yr['列数'])} 列"},
                {'属性': '最大层数', '值': f"{max_tiers} 层"},
                {'属性': '总容量', '值': f"{cap:,} 箱"},
                {'属性': '当前占用', '值': f'{util:.1f}%'},
            ]

            _debug(f'热力图完成: {area_id} {total_boxes}箱 {util:.1f}%')
            return fig, time_str, stats, area_info

        except Exception as e:
            traceback.print_exc()
            return empty_figure(f'热力图生成失败: {str(e)[:50]}'), '-', [], []

    # =============================================
    # 回调5: 超期箱分析
    # =============================================
    @app.callback(
        Output('overdue-summary-cards', 'children'),
        Output('overdue-by-customer-chart', 'figure'),
        Output('overdue-by-area-chart', 'figure'),
        Output('overdue-customer-table', 'data'),
        Output('overdue-customer-table', 'columns'),
        Output('overdue-area-table', 'data'),
        Output('overdue-area-table', 'columns'),
        [Input('calc-overdue-btn', 'n_clicks'),
         Input('init-trigger', 'n_intervals')],
        [State('free-days-input', 'value')],
        prevent_initial_call=False
    )
    def update_overdue_analysis(n_clicks, n_intervals, free_days):
        """更新超期箱分析 - 初始化(默认5天)+按钮点击"""
        _debug(f'update_overdue_analysis触发 n_clicks={n_clicks}, '
               f'n_intervals={n_intervals}, init={S.is_initialized()}')

        ef_cust = empty_figure('点击"计算超期箱"或稍候自动加载')
        ef_area = empty_figure('点击"计算超期箱"或稍候自动加载')
        default_cards = dbc.Alert(
            '数据加载中，或点击上方"计算超期箱"按钮', color='secondary'
        )

        if not S.is_initialized():
            return default_cards, ef_cust, ef_area, [], [], [], []

        # 初始加载自动跑一次 (默认5天)
        run_auto = (n_intervals is not None and n_intervals > 0 and n_clicks is None)
        run_click = (n_clicks is not None and n_clicks > 0)

        if not run_auto and not run_click:
            return default_cards, ef_cust, ef_area, [], [], [], []

        try:
            free_days = int(free_days) if free_days else 5
            analyzer = OverdueBoxAnalyzer(free_days=free_days)
            overdue_df, by_customer, by_area, by_route = analyzer.analyze(S.container_df)
            S.overdue_cache = {
                'df': overdue_df, 'customer': by_customer,
                'area': by_area, 'free_days': free_days
            }

            total_overdue = len(overdue_df)
            total_fee = overdue_df['滞箱费用'].sum()
            avg_overdue_days = overdue_df['超期天数'].mean() if total_overdue > 0 else 0

            cards = dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Small('超期箱总数', className='text-muted'),
                        html.H4(f'{total_overdue:,}', className='fw-bold text-danger'),
                        html.Small(f'免费期 {free_days} 天', className='text-muted')
                    ])
                ], color='danger', outline=True), md=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Small('预估滞箱费用合计', className='text-muted'),
                        html.H4(f'¥ {total_fee:,.0f}', className='fw-bold text-warning'),
                        html.Small('按阶梯费率计算', className='text-muted')
                    ])
                ], color='warning', outline=True), md=4),
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.Small('平均超期天数', className='text-muted'),
                        html.H4(f'{avg_overdue_days:.1f} 天', className='fw-bold text-info'),
                        html.Small(
                            f'超期比例 {total_overdue/len(S.container_df)*100:.1f}%',
                            className='text-muted'
                        )
                    ])
                ], color='info', outline=True), md=4),
            ])

            cust_chart = ChartBuilder.create_overdue_bar_chart(
                by_customer, 'TOP 10 客户 - 超期箱数量',
                '超期箱数量', '客户', top_n=10
            )
            area_chart = ChartBuilder.create_overdue_bar_chart(
                by_area, '各区域 - 预估滞箱费用 (元)',
                '预估滞箱费用', '堆场区域', top_n=10
            )

            cust_cols = [
                {'name': '客户名称', 'id': '客户'},
                {'name': '超期箱数', 'id': '超期箱数量',
                 'type': 'numeric', 'format': Format(group=',')},
                {'name': '平均超期天', 'id': '平均超期天数',
                 'format': Format(precision=1, scheme=Scheme.fixed)},
                {'name': '预估滞箱费', 'id': '预估滞箱费用',
                 'type': 'numeric', 'format': Format(group=',', precision=0)},
            ]

            area_cols = [
                {'name': '堆场区域', 'id': '堆场区域'},
                {'name': '超期箱数', 'id': '超期箱数量',
                 'type': 'numeric', 'format': Format(group=',')},
                {'name': '平均超期天', 'id': '平均超期天数',
                 'format': Format(precision=1, scheme=Scheme.fixed)},
                {'name': '预估滞箱费', 'id': '预估滞箱费用',
                 'type': 'numeric', 'format': Format(group=',', precision=0)},
            ]

            cust_records = by_customer.copy()
            if '平均超期天数' in cust_records.columns:
                cust_records['平均超期天数'] = cust_records['平均超期天数'].apply(lambda v: round(float(v), 1))
            if '平均堆存天数' in cust_records.columns:
                cust_records['平均堆存天数'] = cust_records['平均堆存天数'].apply(lambda v: round(float(v), 1))
            if '预估滞箱费用' in cust_records.columns:
                cust_records['预估滞箱费用'] = cust_records['预估滞箱费用'].apply(lambda v: int(round(float(v), 0)))

            area_records = by_area.copy()
            if '平均超期天数' in area_records.columns:
                area_records['平均超期天数'] = area_records['平均超期天数'].apply(lambda v: round(float(v), 1))
            if '预估滞箱费用' in area_records.columns:
                area_records['预估滞箱费用'] = area_records['预估滞箱费用'].apply(lambda v: int(round(float(v), 0)))

            _debug(f'超期箱分析完成: {total_overdue}个, ¥{total_fee:,.0f}')
            return (cards, cust_chart, area_chart,
                    cust_records.to_dict('records'), cust_cols,
                    area_records.to_dict('records'), area_cols)
        except Exception as e:
            traceback.print_exc()
            return (dbc.Alert(f'计算失败: {str(e)[:60]}', color='danger'),
                    ef_cust, ef_area, [], [], [], [])

    # =============================================
    # 辅助: 格式化翻箱率结果
    # =============================================
    def _format_rehandling_results(result_df):
        """格式化翻箱率结果（内部函数）"""
        _debug(f'  _format_rehandling_results 输入: {len(result_df)} 行, 列={list(result_df.columns)}')

        if result_df.empty:
            _debug('  ⚠️ result_df 为空！')
            ef = empty_figure('模拟结果为空，请重试')
            return (ef, [], [], [], *(['-'] * 6))

        try:
            fig = ChartBuilder.create_rehandling_comparison(result_df)
        except Exception as e:
            _debug(f'  ⚠️ 图表生成失败: {e}')
            fig = empty_figure(f'图表生成失败: {str(e)[:40]}')

        strategy_order = [
            '策略A-随机堆放', '策略B-按提箱时间排序', '策略C-按航线聚堆'
        ]

        try:
            strat_avg = result_df.groupby('策略').agg(
                平均翻箱率=('平均翻箱率', 'mean'),
                P95翻箱率=('P95翻箱率', 'mean')
            )
            _debug(f'  分组后策略: {list(strat_avg.index)}')
            _debug(f'  分组值:\n{strat_avg}')
        except Exception as e:
            _debug(f'  ⚠️ 分组聚合失败: {e}')
            return (fig, result_df.to_dict('records'), [], [], *(['-'] * 6))

        vals = []
        for s in strategy_order:
            if s in strat_avg.index:
                avg_val = float(strat_avg.loc[s, '平均翻箱率']) * 100
                p95_val = float(strat_avg.loc[s, 'P95翻箱率']) * 100
                vals.append(f'{avg_val:.2f}%')
                vals.append(f'{p95_val:.2f}%')
                _debug(f'  策略[{s}] avg={avg_val:.4f}%, p95={p95_val:.4f}%')
            else:
                vals += ['-', '-']
                _debug(f'  策略[{s}] 未找到！')

        cols = [
            {'name': '区域', 'id': '区域'},
            {'name': '策略', 'id': '策略'},
            {'name': '平均翻箱率(倍)', 'id': '平均翻箱率',
             'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
            {'name': 'P95翻箱率(倍)', 'id': 'P95翻箱率',
             'type': 'numeric', 'format': Format(precision=2, scheme=Scheme.fixed)},
            {'name': '总提箱次数', 'id': '总提箱次数',
             'type': 'numeric', 'format': Format(group=',')},
            {'name': '总翻箱次数', 'id': '总翻箱次数',
             'type': 'numeric', 'format': Format(group=',')},
        ]

        detail_records = result_df.copy()
        for c in ['平均翻箱率', 'P95翻箱率']:
            if c in detail_records.columns:
                detail_records[c] = detail_records[c].apply(lambda v: round(float(v), 2))
        for c in ['总提箱次数', '总翻箱次数']:
            if c in detail_records.columns:
                detail_records[c] = detail_records[c].apply(lambda v: int(round(float(v), 0)))

        _debug(f'  最终卡片值: {vals}')
        return (fig, detail_records.to_dict('records'), cols, [], *vals)

    # =============================================
    # 回调6: 翻箱率蒙特卡洛模拟
    # =============================================
    @app.callback(
        Output('rehandling-chart', 'figure'),
        Output('rehandling-detail-table', 'data'),
        Output('rehandling-detail-table', 'columns'),
        Output('sim-loading', 'children'),
        Output('strat-a-avg', 'children'),
        Output('strat-a-p95', 'children'),
        Output('strat-b-avg', 'children'),
        Output('strat-b-p95', 'children'),
        Output('strat-c-avg', 'children'),
        Output('strat-c-p95', 'children'),
        [Input('run-simulation-btn', 'n_clicks')],
        [State('sim-count', 'value'), State('fill-ratio', 'value')],
        prevent_initial_call=False
    )
    def run_rehandling_simulation(n_clicks, sim_count, fill_ratio):
        """运行翻箱率蒙特卡洛模拟"""
        _debug(f'run_rehandling_simulation触发 n_clicks={n_clicks}, sim_count={sim_count}, fill_ratio={fill_ratio}')

        ef = empty_figure(
            '点击"运行蒙特卡洛模拟"开始分析（首次需数分钟）'
        )
        dashes = ['-'] * 6

        if not S.is_initialized():
            _debug('  ⚠️ 数据未初始化')
            return (ef, [], [], [], *dashes)

        # 初始加载: 显示提示
        if n_clicks is None or n_clicks == 0:
            if S.rehandling_results_cache is not None:
                _debug(f'  使用缓存结果: {len(S.rehandling_results_cache)} 条')
                return _format_rehandling_results(S.rehandling_results_cache)
            _debug('  初始状态: 显示占位')
            return (ef, [], [], [], *dashes)

        # 点击运行: 执行模拟
        try:
            _sim_count = int(sim_count) if sim_count else 1000
            _fill_ratio = float(fill_ratio) if fill_ratio else 0.7
            _debug(f'  开始模拟: sim_count={_sim_count}, fill_ratio={_fill_ratio}')

            simulator = YardStackSimulator(
                S.yard_df, num_simulations=_sim_count,
                seed=np.random.randint(1, 9999)
            )
            all_results = []
            areas = S.yard_df['区域编号'].tolist()[:4]
            _debug(f'  模拟区域: {areas}')

            for area_id in areas:
                _debug(f'    正在模拟区域 {area_id}...')
                area_results = simulator.run_single_area_simulation(
                    area_id, fill_ratio=_fill_ratio
                )
                _debug(f'    {area_id}: {len(area_results)} 个策略')
                for strategy, result in area_results.items():
                    all_results.append({
                        '区域': area_id,
                        '策略': strategy,
                        '平均翻箱率': float(result.avg_relocation_rate),
                        'P95翻箱率': float(result.p95_relocation_rate),
                        '总提箱次数': int(result.total_pickups),
                        '总翻箱次数': int(result.total_relocations)
                    })

            _debug(f'  模拟完成，共 {len(all_results)} 条记录')
            result_df = pd.DataFrame(all_results)
            S.rehandling_results_cache = result_df
            return _format_rehandling_results(result_df)
        except Exception as e:
            _debug(f'  ⚠️ 模拟异常: {e}')
            traceback.print_exc()
            return (empty_figure(f'模拟失败: {str(e)[:60]}'), [], [], [], *dashes)

    # =============================================
    # 回调7: KPI仪表板 (初始加载自动跑)
    # =============================================
    @app.callback(
        Output('kpi-vessel-hour', 'children'),
        Output('kpi-crane-hour', 'children'),
        Output('kpi-berth-util', 'children'),
        Output('kpi-truck-time', 'children'),
        Output('kpi-vessel-hour-trend', 'children'),
        Output('kpi-crane-hour-trend', 'children'),
        Output('kpi-berth-util-trend', 'children'),
        Output('kpi-truck-time-trend', 'children'),
        Output('kpi-trend-chart', 'figure'),
        [Input('kpi-granularity', 'value'),
         Input('init-trigger', 'n_intervals')],
        prevent_initial_call=False
    )
    def update_kpi_dashboard(granularity, n_intervals):
        """更新KPI仪表板 - 初始化+切换粒度自动更新"""
        _debug(f'update_kpi_dashboard触发 granularity={granularity}, '
               f'n_intervals={n_intervals}, init={S.is_initialized()}')

        ef = empty_figure('KPI数据加载中...')
        dashes = ['-'] * 8

        if not S.is_initialized():
            return (*dashes, ef)

        if granularity is None:
            granularity = '日'

        try:
            kpi_data = S.preprocessor.calculate_kpis(granularity=granularity)
            S.kpi_cache = kpi_data
            df = kpi_data.get('综合KPI', pd.DataFrame())
            if df.empty:
                return (*dashes, ef)

            latest = df.iloc[-1]

            def make_trend_badge(col, is_higher_better=True):
                if len(df) < 2 or col not in df.columns:
                    return ''
                curr = df[col].iloc[-1]
                prev = df[col].iloc[-2]
                if pd.isna(curr) or pd.isna(prev) or abs(prev) < 1e-9:
                    return ''
                change = (curr - prev) / prev * 100
                if is_higher_better:
                    up = change > 0
                else:
                    up = change < 0
                color = 'success' if up else 'danger'
                icon = 'fa-arrow-up' if up else 'fa-arrow-down'
                return html.Span([
                    html.I(className=f'fas {icon} me-1'),
                    f'{abs(change):.1f}% vs 上期'
                ], className=f'text-{color}')

            ship_eff = latest.get('船时效率_TEU_小时', np.nan)
            crane_eff = latest.get('桥吊效率_TEU_台时', np.nan)
            berth_util = latest.get('泊位利用率', np.nan)
            truck_time = latest.get('平均集卡周转_小时', np.nan)

            kpi_fig = ChartBuilder.create_kpi_trend_chart(df)

            _debug(f'KPI计算完成: 船时{ship_eff:.1f}, 泊位{berth_util*100:.1f}%')
            return (
                f'{ship_eff:.1f}' if not pd.isna(ship_eff) else '-',
                f'{crane_eff:.2f}' if not pd.isna(crane_eff) else '-',
                f'{berth_util*100:.1f}%' if not pd.isna(berth_util) else '-',
                f'{truck_time:.1f} h' if not pd.isna(truck_time) else '-',
                make_trend_badge('船时效率_TEU_小时', True),
                make_trend_badge('桥吊效率_TEU_台时', True),
                make_trend_badge('泊位利用率', True),
                make_trend_badge('平均集卡周转_小时', False),
                kpi_fig
            )
        except Exception as e:
            traceback.print_exc()
            return (*dashes, ef)

    # =============================================
    # 回调8: What-if情景分析
    # =============================================
    @app.callback(
        Output('whatif-chart', 'figure'),
        Output('whatif-result-table', 'data'),
        Output('whatif-result-table', 'columns'),
        Output('whatif-loading', 'children'),
        [Input('run-whatif-1', 'n_clicks'),
         Input('run-whatif-2', 'n_clicks'),
         Input('run-whatif-3', 'n_clicks'),
         Input('run-whatif-all', 'n_clicks')],
        [State('whatif-growth-rate', 'value'),
         State('new-bays', 'value'),
         State('new-rows', 'value'),
         State('new-tiers', 'value'),
         State('whatif-reduce-days', 'value')],
        prevent_initial_call=False
    )
    def run_whatif_analysis(n1, n2, n3, n_all, growth, bays, rows, tiers, reduce_days):
        """运行What-if情景分析"""
        _debug(f'run_whatif_analysis触发 n1={n1} n2={n2} n3={n3} n_all={n_all}')

        ef = empty_figure('点击下方任意"运行"按钮，或点击"一键运行全部"')

        if not S.is_initialized():
            return ef, [], [], []

        ctx = callback_context
        triggered = ctx.triggered_id

        # 初始化: 不运行
        if triggered is None:
            return ef, [], [], []

        try:
            scenario = WhatIfScenario(S.preprocessor)
            results = []

            if triggered in ['run-whatif-1', 'run-whatif-all']:
                results.append(scenario.scenario_throughput_growth(growth or 0.2))

            if triggered in ['run-whatif-2', 'run-whatif-all']:
                bays_i = int(bays) if bays else 10
                rows_i = int(rows) if rows else 8
                tiers_i = int(tiers) if tiers else 6
                results.append(scenario.scenario_new_yard_area(bays_i, rows_i, tiers_i))

            if triggered in ['run-whatif-3', 'run-whatif-all']:
                reduce_i = int(reduce_days) if reduce_days else 2
                results.append(scenario.scenario_reduce_storage_days(reduce_i))

            # 一键运行全部: 额外加完整组合
            if triggered == 'run-whatif-all':
                for g in [0.1, 0.3]:
                    results.append(scenario.scenario_throughput_growth(g))
                for d in [1, 3]:
                    results.append(scenario.scenario_reduce_storage_days(d))

            if not results:
                return ef, [], [], []

            result_df = pd.DataFrame(results)
            fig = ChartBuilder.create_whatif_comparison_chart(result_df)

            cols = [{'name': c, 'id': c} for c in result_df.columns]
            for c in cols:
                n = c['name']
                if any(k in n for k in ['利用率', '比例', '幅度', '增长率']):
                    c['type'] = 'numeric'
                    c['format'] = Format(
                        precision=2, scheme=Scheme.percent
                    )
                elif any(k in n for k in ['费用', '容量', '数量', '天数']):
                    c['type'] = 'numeric'
                    c['format'] = Format(group=',', precision=0)

            _debug(f'What-if分析完成: {len(results)} 个情景')
            return fig, result_df.to_dict('records'), cols, []
        except Exception as e:
            traceback.print_exc()
            return (empty_figure(f'分析失败: {str(e)[:50]}'), [], [], [])

    # =============================================
    # 回调9: 导出PDF报告
    # =============================================
    @app.callback(
        Output('download-pdf', 'data'),
        Output('export-status', 'children'),
        [Input('export-pdf-btn', 'n_clicks')],
        prevent_initial_call=False
    )
    def export_pdf_report(n_clicks):
        """导出PDF运营分析报告"""
        _debug(f'export_pdf_report触发 n_clicks={n_clicks}')

        if n_clicks is None or n_clicks == 0:
            return None, ''

        if not S.is_initialized():
            return None, html.Span('✗ 暂无数据可导出', className='text-danger')

        try:
            os.makedirs('reports', exist_ok=True)
            generator = PortReportGenerator('reports')
            start_date = S.daily_throughput['日期'].min()
            end_date = S.daily_throughput['日期'].max()

            by_customer = S.overdue_cache['customer'] if S.overdue_cache else pd.DataFrame()
            by_area = S.overdue_cache['area'] if S.overdue_cache else pd.DataFrame()
            rehandling_df = (
                S.rehandling_results_cache
                if S.rehandling_results_cache is not None
                else pd.DataFrame()
            )

            whatif_df = pd.DataFrame()
            try:
                scenario = WhatIfScenario(S.preprocessor)
                whatif_df = scenario.run_all_scenarios()
            except Exception:
                pass

            figures = {}
            try:
                figures['throughput_chart'] = (
                    ChartBuilder.create_throughput_chart(S.daily_throughput)
                )
            except Exception:
                pass
            try:
                figures['forecast_chart'] = (
                    ChartBuilder.create_forecast_comparison(
                        S.forecast_results_cache or {}
                    )
                )
            except Exception:
                pass
            try:
                figures['yard_util_chart'] = (
                    ChartBuilder.create_yard_utilization_chart(S.yard_util_time)
                )
            except Exception:
                pass
            try:
                figures['rehandling_chart'] = (
                    ChartBuilder.create_rehandling_comparison(rehandling_df)
                )
            except Exception:
                pass

            filepath = generator.generate_report(
                report_name='港口运营分析',
                start_date=start_date, end_date=end_date,
                daily_throughput=S.daily_throughput,
                forecast_results=S.forecast_results_cache or {},
                yard_util=S.yard_util_time,
                rehandling_results=rehandling_df,
                overdue_by_customer=by_customer,
                overdue_by_area=by_area,
                kpi_data=S.kpi_cache or S.preprocessor.calculate_kpis('日'),
                whatif_results=whatif_df,
                figures=figures
            )

            _debug(f'PDF报告生成: {filepath}')
            return (dcc.send_file(filepath),
                    html.Span([
                        html.I(className='fas fa-check-circle me-1 text-success'),
                        f'✓ 报告已生成并下载: {os.path.basename(filepath)}'
                    ], className='text-success fw-bold'))
        except Exception as e:
            traceback.print_exc()
            return None, html.Span(
                f'✗ 导出失败: {str(e)[:60]}', className='text-danger'
            )

    # =============================================
    # 辅助: 检测泊位冲突
    # =============================================
    def _detect_berth_conflicts(vessel_df):
        """检测同一泊位的时间重叠冲突"""
        if vessel_df is None or vessel_df.empty:
            return pd.DataFrame(), set()

        df = vessel_df.copy()
        df['到港时间'] = pd.to_datetime(df['到港时间'])
        df['离港时间'] = pd.to_datetime(df['离港时间'])

        conflicts = []
        conflict_vessels = set()

        berths = df['分配泊位编号'].unique()
        for berth in berths:
            berth_df = df[df['分配泊位编号'] == berth].sort_values('到港时间')
            vessels = berth_df.to_dict('records')

            for i in range(len(vessels)):
                for j in range(i + 1, len(vessels)):
                    v1 = vessels[i]
                    v2 = vessels[j]

                    overlap_start = max(v1['到港时间'], v2['到港时间'])
                    overlap_end = min(v1['离港时间'], v2['离港时间'])

                    if overlap_start < overlap_end:
                        overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
                        conflicts.append({
                            '泊位号': berth,
                            '船名A': v1['船名'],
                            '船名B': v2['船名'],
                            '重叠开始': overlap_start.strftime('%Y-%m-%d %H:%M'),
                            '重叠结束': overlap_end.strftime('%Y-%m-%d %H:%M'),
                            '重叠时长_小时': round(overlap_hours, 2)
                        })
                        conflict_vessels.add(v1['船名'])
                        conflict_vessels.add(v2['船名'])

        return pd.DataFrame(conflicts), conflict_vessels

    # =============================================
    # 辅助: 计算泊位效率指标
    # =============================================
    def _calculate_berth_efficiency(vessel_df):
        """计算每个泊位的效率指标"""
        if vessel_df is None or vessel_df.empty:
            return pd.DataFrame()

        df = vessel_df.copy()
        df['到港时间'] = pd.to_datetime(df['到港时间'])
        df['离港时间'] = pd.to_datetime(df['离港时间'])

        results = []
        berths = sorted(df['分配泊位编号'].unique())

        for berth in berths:
            berth_df = df[df['分配泊位编号'] == berth].sort_values('到港时间')

            turnaround_times = []
            for i in range(1, len(berth_df)):
                prev_depart = berth_df.iloc[i - 1]['离港时间']
                curr_arrive = berth_df.iloc[i]['到港时间']
                gap = (curr_arrive - prev_depart).total_seconds() / 3600
                if gap >= 0:
                    turnaround_times.append(gap)

            avg_turnaround = np.mean(turnaround_times) if turnaround_times else np.nan
            total_vessels = len(berth_df)
            total_teu = berth_df['载箱量TEU'].sum()

            results.append({
                '泊位编号': berth,
                '平均周转时间_小时': round(avg_turnaround, 2) if not np.isnan(avg_turnaround) else 0,
                '靠泊船次': total_vessels,
                '累计装卸TEU': int(total_teu)
            })

        result_df = pd.DataFrame(results)
        if not result_df.empty:
            result_df = result_df.sort_values('平均周转时间_小时', ascending=True).reset_index(drop=True)

        return result_df

    # =============================================
    # 辅助: 生成泊位效率卡片
    # =============================================
    def _render_efficiency_cards(efficiency_df):
        """根据效率数据生成卡片列表"""
        if efficiency_df is None or efficiency_df.empty:
            return html.Div('暂无数据', className='text-muted text-center py-3')

        cards = []
        for idx, row in efficiency_df.iterrows():
            berth = row['泊位编号']
            turnaround = float(row['平均周转时间_小时'])
            vessel_count = int(row['靠泊船次'])
            total_teu = int(row['累计装卸TEU'])

            if turnaround < 2:
                card_color = 'success'
                bg_style = {'backgroundColor': '#f0fff4', 'borderColor': '#38a169'}
                text_color = 'text-success'
            elif turnaround > 8:
                card_color = 'danger'
                bg_style = {'backgroundColor': '#fff5f5', 'borderColor': '#e53e3e'}
                text_color = 'text-danger'
            else:
                card_color = 'primary'
                bg_style = {'backgroundColor': '#ebf8ff', 'borderColor': '#3182ce'}
                text_color = 'text-primary'

            rank_icon = ''
            if idx == 0:
                rank_icon = html.I(className='fas fa-crown text-warning me-1')
            elif idx == 1:
                rank_icon = html.I(className='fas fa-medal text-secondary me-1')
            elif idx == 2:
                rank_icon = html.I(className='fas fa-award text-info me-1')

            card = dbc.Card([
                dbc.CardBody([
                    html.H6([
                        rank_icon,
                        f'泊位 {berth}',
                        dbc.Badge(f'第{idx+1}名', color=card_color,
                                  className='float-end', pill=True)
                    ], className=f'fw-bold {text_color} mb-3'),
                    html.Div([
                        html.Small('平均周转时间', className='text-muted d-block mb-1'),
                        html.H4(f'{turnaround:.2f} h', className=f'fw-bold {text_color} mb-3')
                    ]),
                    html.Hr(className='my-2'),
                    dbc.Row([
                        dbc.Col([
                            html.Small('靠泊船次', className='text-muted d-block'),
                            html.Span(f'{vessel_count} 艘', className='fw-bold')
                        ]),
                        dbc.Col([
                            html.Small('累计装卸', className='text-muted d-block'),
                            html.Span(f'{total_teu:,} TEU', className='fw-bold')
                        ]),
                    ]),
                ])
            ], style=bg_style, className=f'mb-3 border-2 border-{card_color} shadow-sm')
            cards.append(card)

        return html.Div(cards)

    # =============================================
    # 回调10: 泊位调度甘特图 - 滑块 + 快捷按钮 + 所有内容
    # =============================================
    @app.callback(
        Output('berth-date-slider', 'value'),
        Output('berth-range-label', 'children'),
        Output('berth-gantt-chart', 'figure'),
        Output('berth-conflict-summary', 'children'),
        Output('berth-conflict-table', 'data'),
        Output('berth-efficiency-cards', 'children'),
        [Input('init-trigger', 'n_intervals'),
         Input('app-data-store', 'data'),
         Input('berth-date-slider', 'value'),
         Input('berth-quick-7d', 'n_clicks'),
         Input('berth-quick-30d', 'n_clicks'),
         Input('berth-quick-all', 'n_clicks')],
        prevent_initial_call=False
    )
    def update_berth_dashboard(n_intervals, store_data, slider_value,
                                q7d, q30d, qall):
        """更新泊位调度甘特图及相关面板"""
        _debug(f'update_berth_dashboard触发 init={S.is_initialized()}, slider={slider_value}')

        ef_gantt = empty_figure('数据加载中...')
        empty_summary = html.Span('加载中...', className='text-muted')
        empty_data = []
        empty_cards = html.Div('加载中...', className='text-muted text-center py-3')

        if not S.is_initialized():
            return ([0, 100], '数据加载中...', ef_gantt, empty_summary, empty_data, empty_cards)

        try:
            ctx = callback_context
            triggered = ctx.triggered_id

            vessel_df = S.vessel_df.copy()
            vessel_df['到港时间'] = pd.to_datetime(vessel_df['到港时间'])
            vessel_df['离港时间'] = pd.to_datetime(vessel_df['离港时间'])

            min_dt = vessel_df['到港时间'].min()
            max_dt = vessel_df['离港时间'].max()
            total_seconds = (max_dt - min_dt).total_seconds()

            def pct_to_dt(pct):
                return min_dt + timedelta(seconds=total_seconds * pct / 100)

            def dt_to_pct(dt):
                return (dt - min_dt).total_seconds() / total_seconds * 100

            def get_pct_range_for_quick(days):
                end = max_dt
                start = end - timedelta(days=days)
                start_pct = max(0, dt_to_pct(start))
                end_pct = 100
                return [start_pct, end_pct]

            new_slider = slider_value

            if triggered in ['init-trigger', 'app-data-store']:
                new_slider = get_pct_range_for_quick(7)
            elif triggered == 'berth-quick-7d' and q7d is not None and q7d > 0:
                new_slider = get_pct_range_for_quick(7)
            elif triggered == 'berth-quick-30d' and q30d is not None and q30d > 0:
                new_slider = get_pct_range_for_quick(30)
            elif triggered == 'berth-quick-all' and qall is not None and qall > 0:
                new_slider = [0, 100]

            if new_slider is None or len(new_slider) != 2:
                new_slider = [0, 100]

            start_pct = max(0, min(100, new_slider[0]))
            end_pct = max(0, min(100, new_slider[1]))
            if end_pct < start_pct:
                start_pct, end_pct = end_pct, start_pct

            start_dt = pct_to_dt(start_pct)
            end_dt = pct_to_dt(end_pct)

            range_label = (
                f'{start_dt.strftime("%Y-%m-%d %H:%M")}  ~  {end_dt.strftime("%Y-%m-%d %H:%M")}'
                f'  (共 {(end_dt - start_dt).days}天{(end_dt - start_dt).seconds//3600}小时)'
            )

            conflicts_df, conflict_vessel_ids = _detect_berth_conflicts(vessel_df)

            filtered_df = vessel_df[
                (vessel_df['离港时间'] >= start_dt) &
                (vessel_df['到港时间'] <= end_dt)
            ].copy()

            filtered_conflicts = conflicts_df.copy()
            if not filtered_conflicts.empty:
                filtered_conflicts['_start'] = pd.to_datetime(filtered_conflicts['重叠开始'])
                filtered_conflicts['_end'] = pd.to_datetime(filtered_conflicts['重叠结束'])
                filtered_conflicts = filtered_conflicts[
                    (filtered_conflicts['_end'] >= start_dt) &
                    (filtered_conflicts['_start'] <= end_dt)
                ]
                filtered_conflicts = filtered_conflicts.drop(columns=['_start', '_end'])
            else:
                filtered_conflicts = pd.DataFrame()

            filtered_conflict_vessels = set()
            if not filtered_conflicts.empty:
                filtered_conflict_vessels = set(filtered_conflicts['船名A'].tolist() +
                                                filtered_conflicts['船名B'].tolist())

            gantt_fig = ChartBuilder.create_berth_gantt_chart(
                vessel_df,
                date_start=start_dt,
                date_end=end_dt,
                conflict_vessel_ids=filtered_conflict_vessels
            )

            if filtered_conflicts.empty:
                conflict_summary = dbc.Alert(
                    [html.I(className='fas fa-check-circle me-2'),
                     '未检测到泊位冲突，调度正常'],
                    color='success', className='mb-0'
                )
            else:
                conflict_summary = dbc.Alert(
                    [html.I(className='fas fa-exclamation-triangle me-2'),
                     f'检测到 {len(filtered_conflicts)} 处泊位冲突，请及时处理！'],
                    color='danger', className='mb-0'
                )

            efficiency_df = _calculate_berth_efficiency(filtered_df)
            efficiency_cards = _render_efficiency_cards(efficiency_df)

            conflict_data = filtered_conflicts.to_dict('records') if not filtered_conflicts.empty else []

            _debug(f'泊位甘特图更新完成: {len(filtered_df)}条记录, {len(filtered_conflicts)}处冲突')
            return (
                [start_pct, end_pct],
                range_label,
                gantt_fig,
                conflict_summary,
                conflict_data,
                efficiency_cards
            )

        except Exception as e:
            traceback.print_exc()
            err_fig = empty_figure(f'甘特图生成失败: {str(e)[:50]}')
            err_summary = dbc.Alert(f'错误: {str(e)[:80]}', color='danger')
            return (slider_value or [0, 100], '错误', err_fig, err_summary, [], empty_cards)

    # =============================================
    # 回调11: 船舶排队调度仿真 - 单策略/三策略对比
    # =============================================
    @app.callback(
        Output('sim-timeline-chart', 'figure'),
        Output('sim-berth-gantt', 'figure'),
        Output('sim-wait-histogram', 'figure'),
        Output('sim-avg-wait', 'children'),
        Output('sim-max-wait', 'children'),
        Output('sim-berth-util', 'children'),
        Output('sim-avg-service', 'children'),
        Output('sim-throughput', 'children'),
        Output('sim-reject-rate', 'children'),
        Output('sim-card-avg-wait', 'style'),
        Output('sim-card-reject', 'style'),
        Output('sim-stats-container', 'style'),
        Output('sim-multi-stats-container', 'style'),
        Output('sim-multi-stats-container', 'children'),
        Output('sim-replay-controls', 'style'),
        Output('replay-slider', 'max'),
        Output('replay-slider', 'value'),
        Output('replay-time-label', 'children'),
        Output('sim-queue-loading', 'children'),
        [Input('run-sim-btn', 'n_clicks')],
        [State('sim-strategy', 'value'),
         State('sim-arrival-mean', 'value'),
         State('sim-service-mean', 'value'),
         State('sim-service-std', 'value'),
         State('sim-num-berths', 'value'),
         State('sim-max-anchor', 'value'),
         State('sim-duration-days', 'value')],
        prevent_initial_call=False
    )
    def run_simulation(n_clicks, strategy, arrival_mean, service_mean, service_std,
                        num_berths, max_anchor, duration_days):
        """运行船舶排队调度仿真（支持单策略和三策略对比）"""
        _debug(f'run_simulation触发 n_clicks={n_clicks}, strategy={strategy}')

        ef_timeline = empty_figure('点击"运行仿真"按钮开始船舶排队调度仿真')
        ef_gantt = empty_figure('点击"运行仿真"按钮开始船舶排队调度仿真')
        ef_hist = empty_figure('点击"运行仿真"按钮开始船舶排队调度仿真')
        dashes = ['-'] * 6
        normal_style = {}
        single_stats_style = {'display': 'block'}
        multi_stats_style = {'display': 'none'}
        multi_stats_children = []
        replay_style = {'display': 'none'}
        slider_max = 100
        slider_val = 0
        time_label = ''

        if n_clicks is None or n_clicks == 0:
            if S.simulation_result is not None and S.simulation_result.strategy != STRATEGY_ALL:
                return _format_simulation_results(S.simulation_result)
            elif S.multi_strategy_results is not None:
                return _format_multi_strategy_results(S.multi_strategy_results)
            return (ef_timeline, ef_gantt, ef_hist, *dashes, normal_style, normal_style,
                    single_stats_style, multi_stats_style, multi_stats_children,
                    replay_style, slider_max, slider_val, time_label, [])

        try:
            arrival_mean = float(arrival_mean) if arrival_mean else 8.0
            service_mean = float(service_mean) if service_mean else 36.0
            service_std = float(service_std) if service_std else 6.0
            num_berths = int(num_berths) if num_berths else 4
            max_anchor = int(max_anchor) if max_anchor else 20
            duration_days = int(duration_days) if duration_days else 30
            sim_duration = duration_days * 24.0

            base_seed = np.random.randint(1, 99999)

            if strategy == STRATEGY_ALL:
                strategies = [STRATEGY_FCFS, STRATEGY_SJF, STRATEGY_LWF]
                results = {}
                for i, strat in enumerate(strategies):
                    simulator = ShipQueueSimulator(seed=base_seed + i)
                    result = simulator.run(
                        arrival_mean=arrival_mean,
                        service_mean=service_mean,
                        service_std=service_std,
                        num_berths=num_berths,
                        max_anchor=max_anchor,
                        sim_duration=sim_duration,
                        strategy=strat
                    )
                    results[strat] = result

                S.multi_strategy_results = results
                S.simulation_result = None
                S.replay_enabled = False
                return _format_multi_strategy_results(results)
            else:
                simulator = ShipQueueSimulator(seed=base_seed)
                result = simulator.run(
                    arrival_mean=arrival_mean,
                    service_mean=service_mean,
                    service_std=service_std,
                    num_berths=num_berths,
                    max_anchor=max_anchor,
                    sim_duration=sim_duration,
                    strategy=strategy
                )

                S.simulation_result = result
                S.multi_strategy_results = None
                S.replay_enabled = True
                S.replay_current_time = 0.0
                S.replay_playing = False
                S.replay_speed = 1.0

                return _format_simulation_results(result)

        except Exception as e:
            traceback.print_exc()
            return (
                empty_figure(f'仿真失败: {str(e)[:50]}'),
                empty_figure(f'仿真失败: {str(e)[:50]}'),
                empty_figure(f'仿真失败: {str(e)[:50]}'),
                *['-'] * 6, normal_style, normal_style,
                single_stats_style, multi_stats_style, multi_stats_children,
                replay_style, slider_max, slider_val, time_label, []
            )

    def _format_simulation_results(result, current_time=None):
        """格式化单策略仿真结果用于展示"""
        sim_duration = result.params.get('sim_duration', 0)

        if current_time is not None and current_time > 0:
            timeline_fig = ChartBuilder.create_replay_timeline(result.timeline_data, current_time)
            berth_gantt_fig = ChartBuilder.create_replay_berth_gantt(
                result.berth_occupancy, sim_duration, current_time
            )
            stats = ShipQueueSimulator.get_stats_at_time(result, current_time)
            wait_hist_fig = ChartBuilder.create_wait_time_histogram(
                [s for s in result.ships if s.departure_time > 0 and s.departure_time <= current_time
                 and not s.rejected]
            )
            avg_service = result.stats['avg_service_time']
        else:
            timeline_fig = ChartBuilder.create_simulation_timeline_chart(result.timeline_data)
            berth_gantt_fig = ChartBuilder.create_simulation_berth_gantt(
                result.berth_occupancy, sim_duration
            )
            wait_hist_fig = ChartBuilder.create_wait_time_histogram(result.ships)
            stats = result.stats
            avg_service = stats['avg_service_time']

        avg_wait = stats['avg_wait_time']
        max_wait = stats['max_wait_time']
        berth_util = stats['avg_berth_utilization'] * 100
        throughput = stats['throughput']
        reject_rate = (stats.get('rejected_count', 0) / max(1, stats.get('total_arrivals', 1))) * 100

        avg_wait_style = {}
        reject_style = {}

        if avg_wait > 24:
            avg_wait_style = {
                'border': '2px solid #e53e3e',
                'backgroundColor': '#fff5f5'
            }

        if reject_rate > 5:
            reject_style = {
                'border': '2px solid #e53e3e',
                'backgroundColor': '#fff5f5'
            }

        single_stats_style = {'display': 'block'}
        multi_stats_style = {'display': 'none'}
        multi_stats_children = []

        replay_style = {'display': 'block'} if S.replay_enabled else {'display': 'none'}
        slider_max = sim_duration
        slider_val = current_time if current_time is not None else 0
        if current_time is not None:
            time_label = f'{current_time:.1f} h / {sim_duration:.0f} h'
        else:
            time_label = f'0.0 h / {sim_duration:.0f} h'

        return (
            timeline_fig, berth_gantt_fig, wait_hist_fig,
            f'{avg_wait:.1f}',
            f'{max_wait:.1f}',
            f'{berth_util:.1f}',
            f'{avg_service:.1f}',
            f'{throughput}',
            f'{reject_rate:.2f}',
            avg_wait_style,
            reject_style,
            single_stats_style,
            multi_stats_style,
            multi_stats_children,
            replay_style,
            slider_max,
            slider_val,
            time_label,
            []
        )

    def _format_multi_strategy_results(results):
        """格式化三策略对比仿真结果"""
        timeline_fig = ChartBuilder.create_multi_strategy_timeline(results)
        wait_hist_fig = ChartBuilder.create_multi_strategy_wait_histogram(results)

        first_key = list(results.keys())[0]
        first_result = results[first_key]
        berth_gantt_fig = ChartBuilder.create_simulation_berth_gantt(
            first_result.berth_occupancy, first_result.params['sim_duration']
        )

        multi_stats_children = _build_multi_strategy_cards(results)

        single_stats_style = {'display': 'none'}
        multi_stats_style = {'display': 'block'}
        replay_style = {'display': 'none'}
        normal_style = {}

        dashes = ['-'] * 6

        return (
            timeline_fig, berth_gantt_fig, wait_hist_fig,
            *dashes, normal_style, normal_style,
            single_stats_style, multi_stats_style, multi_stats_children,
            replay_style, 100, 0, '', []
        )

    def _build_multi_strategy_cards(results):
        """构建三策略对比的指标卡片（三列）"""
        strategy_order = [STRATEGY_FCFS, STRATEGY_SJF, STRATEGY_LWF]
        strategy_names = {
            STRATEGY_FCFS: '先到先服务 (FCFS)',
            STRATEGY_SJF: '最短作业优先 (SJF)',
            STRATEGY_LWF: '最长等待优先 (LWF)',
        }
        strategy_colors = {
            STRATEGY_FCFS: '#2b6cb0',
            STRATEGY_SJF: '#38a169',
            STRATEGY_LWF: '#e53e3e',
        }

        metrics = [
            ('avg_wait_time', '平均等待时长', '小时', True),
            ('max_wait_time', '最大等待时长', '小时', True),
            ('avg_berth_utilization', '泊位利用率', '%', False),
            ('throughput', '吞吐量', '艘', False),
        ]

        metric_values = {}
        for strat in strategy_order:
            if strat not in results:
                continue
            stats = results[strat].stats
            metric_values[strat] = {}
            for key, label, unit, is_lower_better in metrics:
                val = stats.get(key, 0)
                if key == 'avg_berth_utilization':
                    val = val * 100
                metric_values[strat][key] = val

        best_metrics = {}
        for key, label, unit, is_lower_better in metrics:
            vals = []
            for strat in strategy_order:
                if strat in metric_values:
                    vals.append(metric_values[strat][key])
            if not vals:
                continue
            if is_lower_better:
                best_metrics[key] = min(vals)
            else:
                best_metrics[key] = max(vals)

        cards = []
        for strat in strategy_order:
            if strat not in results:
                continue
            color = strategy_colors.get(strat, '#3182ce')
            name = strategy_names.get(strat, strat)

            metric_cards = []
            for key, label, unit, is_lower_better in metrics:
                val = metric_values[strat][key]
                is_best = val == best_metrics.get(key)

                if is_best:
                    value_style = {
                        'color': '#38a169',
                        'fontWeight': 'bold',
                        'fontSize': '24px'
                    }
                    label_prefix = '👑 '
                else:
                    value_style = {'color': color}
                    label_prefix = ''

                if key == 'avg_berth_utilization':
                    val_str = f'{val:.1f}'
                elif key == 'throughput':
                    val_str = f'{int(val)}'
                else:
                    val_str = f'{val:.1f}'

                metric_cards.append(
                    dbc.Col([
                        html.Small(f'{label_prefix}{label}', className='text-muted d-block mb-1'),
                        html.Div(val_str, style=value_style),
                        html.Small(unit, className='text-muted')
                    ], md=6, className='mb-3')
                )

            card = dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-ship me-2'),
                        name
                    ], style={'backgroundColor': color, 'color': 'white'}),
                    dbc.CardBody([
                        dbc.Row(metric_cards)
                    ])
                ], className='shadow-sm h-100')
            ], md=4)
            cards.append(card)

        return html.Div([
            html.H5([html.I(className='fas fa-trophy me-2 text-warning'),
                     '三策略指标对比（最优指标绿色加粗）'],
                    className='mb-3 text-center'),
            dbc.Row(cards, className='g-3 mb-4')
        ])

    # =============================================
    # 回调12: 敏感性分析
    # =============================================
    @app.callback(
        Output('sim-sensitivity-chart', 'figure'),
        Output('sim-queue-loading', 'children', allow_duplicate=True),
        [Input('run-sensitivity-btn', 'n_clicks')],
        [State('sim-strategy', 'value'),
         State('sim-service-mean', 'value'),
         State('sim-service-std', 'value'),
         State('sim-num-berths', 'value'),
         State('sim-max-anchor', 'value'),
         State('sim-duration-days', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def run_sensitivity_analysis(n_clicks, strategy, service_mean, service_std,
                                  num_berths, max_anchor, duration_days):
        """运行敏感性分析"""
        _debug(f'run_sensitivity_analysis触发 n_clicks={n_clicks}, strategy={strategy}')

        ef = empty_figure('点击"敏感性分析"按钮开始分析')

        if n_clicks is None or n_clicks == 0:
            if S.sensitivity_result is not None:
                sens_fig = ChartBuilder.create_sensitivity_chart(S.sensitivity_result)
                return sens_fig, []
            return ef, []

        try:
            service_mean = float(service_mean) if service_mean else 36.0
            service_std = float(service_std) if service_std else 6.0
            num_berths = int(num_berths) if num_berths else 4
            max_anchor = int(max_anchor) if max_anchor else 20
            duration_days = int(duration_days) if duration_days else 30
            sim_duration = duration_days * 24.0

            strat = strategy if strategy in [STRATEGY_FCFS, STRATEGY_SJF, STRATEGY_LWF] else STRATEGY_FCFS

            arrival_means = list(range(4, 17, 2))

            simulator = ShipQueueSimulator()
            sensitivity_df = simulator.run_sensitivity_analysis(
                arrival_means=arrival_means,
                service_mean=service_mean,
                service_std=service_std,
                num_berths=num_berths,
                max_anchor=max_anchor,
                sim_duration=sim_duration,
                base_seed=np.random.randint(1, 9999),
                strategy=strat
            )

            S.sensitivity_result = sensitivity_df
            sens_fig = ChartBuilder.create_sensitivity_chart(sensitivity_df)
            return sens_fig, []

        except Exception as e:
            traceback.print_exc()
            return empty_figure(f'敏感性分析失败: {str(e)[:50]}'), []

    # =============================================
    # 回调13: 回放播放按钮
    # =============================================
    @app.callback(
        Output('replay-interval', 'disabled'),
        Output('replay-play-btn', 'color'),
        [Input('replay-play-btn', 'n_clicks'),
         Input('replay-pause-btn', 'n_clicks')],
        prevent_initial_call=False
    )
    def replay_play_pause(play_clicks, pause_clicks):
        """回放播放/暂停控制"""
        ctx = callback_context
        triggered = ctx.triggered_id

        if triggered == 'replay-play-btn' and play_clicks and play_clicks > 0:
            S.replay_playing = True
            return False, 'success'
        elif triggered == 'replay-pause-btn' and pause_clicks and pause_clicks > 0:
            S.replay_playing = False
            return True, 'success'

        return True, 'success'

    # =============================================
    # 回调14: 回放速度选择
    # =============================================
    @app.callback(
        Output('replay-interval', 'interval'),
        [Input('replay-speed', 'value')],
        prevent_initial_call=False
    )
    def replay_speed_change(speed):
        """调整回放速度"""
        if speed is None:
            speed = 1
        S.replay_speed = speed
        base_interval = 200
        return int(base_interval / speed)

    # =============================================
    # 回调15: 回放定时器 - 自动推进时间
    # =============================================
    @app.callback(
        Output('sim-timeline-chart', 'figure', allow_duplicate=True),
        Output('sim-berth-gantt', 'figure', allow_duplicate=True),
        Output('sim-wait-histogram', 'figure', allow_duplicate=True),
        Output('sim-avg-wait', 'children', allow_duplicate=True),
        Output('sim-max-wait', 'children', allow_duplicate=True),
        Output('sim-berth-util', 'children', allow_duplicate=True),
        Output('sim-avg-service', 'children', allow_duplicate=True),
        Output('sim-throughput', 'children', allow_duplicate=True),
        Output('sim-reject-rate', 'children', allow_duplicate=True),
        Output('sim-card-avg-wait', 'style', allow_duplicate=True),
        Output('sim-card-reject', 'style', allow_duplicate=True),
        Output('replay-slider', 'value', allow_duplicate=True),
        Output('replay-time-label', 'children', allow_duplicate=True),
        [Input('replay-interval', 'n_intervals')],
        prevent_initial_call='initial_duplicate'
    )
    def replay_tick(n_intervals):
        """回放定时器回调 - 每次触发推进时间"""
        if not S.replay_playing or S.simulation_result is None:
            return tuple([no_update] * 13)

        try:
            sim_duration = S.simulation_result.params.get('sim_duration', 0)
            step = sim_duration / 200.0

            new_time = S.replay_current_time + step
            if new_time >= sim_duration:
                new_time = sim_duration
                S.replay_playing = False

            S.replay_current_time = new_time

            result = _format_simulation_results(S.simulation_result, current_time=new_time)

            timeline_fig, berth_gantt_fig, wait_hist_fig = result[0], result[1], result[2]
            avg_wait, max_wait, berth_util, avg_service, throughput, reject_rate = result[3:9]
            avg_wait_style, reject_style = result[9], result[10]
            slider_val = result[15]
            time_label = result[16]

            return (
                timeline_fig, berth_gantt_fig, wait_hist_fig,
                avg_wait, max_wait, berth_util, avg_service, throughput, reject_rate,
                avg_wait_style, reject_style,
                slider_val, time_label
            )
        except Exception as e:
            traceback.print_exc()
            return tuple([no_update] * 13)

    # =============================================
    # 回调16: 回放滑块拖动 - 跳转到指定时间
    # =============================================
    @app.callback(
        Output('sim-timeline-chart', 'figure', allow_duplicate=True),
        Output('sim-berth-gantt', 'figure', allow_duplicate=True),
        Output('sim-wait-histogram', 'figure', allow_duplicate=True),
        Output('sim-avg-wait', 'children', allow_duplicate=True),
        Output('sim-max-wait', 'children', allow_duplicate=True),
        Output('sim-berth-util', 'children', allow_duplicate=True),
        Output('sim-avg-service', 'children', allow_duplicate=True),
        Output('sim-throughput', 'children', allow_duplicate=True),
        Output('sim-reject-rate', 'children', allow_duplicate=True),
        Output('sim-card-avg-wait', 'style', allow_duplicate=True),
        Output('sim-card-reject', 'style', allow_duplicate=True),
        Output('replay-time-label', 'children', allow_duplicate=True),
        [Input('replay-slider', 'value')],
        prevent_initial_call='initial_duplicate'
    )
    def replay_slider_change(value):
        """回放滑块拖动 - 手动跳转"""
        if S.simulation_result is None or not S.replay_enabled:
            return tuple([no_update] * 12)

        try:
            S.replay_current_time = float(value) if value else 0.0

            result = _format_simulation_results(S.simulation_result, current_time=S.replay_current_time)

            timeline_fig, berth_gantt_fig, wait_hist_fig = result[0], result[1], result[2]
            avg_wait, max_wait, berth_util, avg_service, throughput, reject_rate = result[3:9]
            avg_wait_style, reject_style = result[9], result[10]
            time_label = result[16]

            return (
                timeline_fig, berth_gantt_fig, wait_hist_fig,
                avg_wait, max_wait, berth_util, avg_service, throughput, reject_rate,
                avg_wait_style, reject_style,
                time_label
            )
        except Exception as e:
            traceback.print_exc()
            return tuple([no_update] * 12)
