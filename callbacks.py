"""
Dash回调函数 - 处理所有交互逻辑
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
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from src.data_generator import load_port_data, PortDataGenerator
from src.data_processor import DataPreprocessor
from src.forecasting import ThroughputForecaster
from src.rehandling import YardStackSimulator, OverdueBoxAnalyzer
from src.whatif import WhatIfScenario
from src.report_generator import PortReportGenerator
from src.charts import ChartBuilder


def register_callbacks(app):
    """注册所有回调函数"""

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
        [Input('app-data-store', 'data')]
    )
    def update_overview(_):
        """更新概览区数据"""
        global preprocessor, daily_throughput, yard_util_time

        ctx = dash.callback_context
        if preprocessor is None or daily_throughput is None:
            empty_fig = go.Figure()
            return ['-'], empty_fig, '-', '-', '-', '-', '', '', '', '', empty_fig

        df = daily_throughput
        total_in = int(df['进场箱量'].sum())
        total_out = int(df['出场箱量'].sum())
        total_teu = float(df['总TEU'].sum())
        days = len(df)
        avg_daily_teu = total_teu / max(1, days)

        summary = [
            dbc.Badge([
                html.I(className='fas fa-ship me-1'),
                f'船舶靠泊记录: {len(preprocessor.vessel_df):,} 条'
            ], color='primary', className='me-2 mb-2', style={'fontSize': '14px'}),
            dbc.Badge([
                html.I(className='fas fa-box me-1'),
                f'集装箱流转: {len(preprocessor.container_df):,} 条'
            ], color='success', className='me-2 mb-2', style={'fontSize': '14px'}),
            dbc.Badge([
                html.I(className='fas fa-warehouse me-1'),
                f'堆场区域: {len(preprocessor.yard_df)} 个'
            ], color='info', className='me-2 mb-2', style={'fontSize': '14px'}),
            dbc.Badge([
                html.I(className='fas fa-calendar me-1'),
                f'数据跨度: {df["日期"].min().strftime("%Y-%m-%d")} ~ {df["日期"].max().strftime("%Y-%m-%d")}'
            ], color='warning', className='me-2 mb-2', style={'fontSize': '14px'}),
        ]

        throughput_fig = ChartBuilder.create_throughput_chart(df)
        yard_util_fig = ChartBuilder.create_yard_utilization_chart(yard_util_time)

        return (
            summary, throughput_fig,
            f'{total_in:,}', f'{total_out:,}', f'{total_teu:,.0f}', f'{avg_daily_teu:,.0f}',
            f'占总吞吐 {total_in/(total_in+total_out)*100:.1f}%',
            f'占总吞吐 {total_out/(total_in+total_out)*100:.1f}%',
            f'总量 {total_in+total_out:,} 箱',
            f'共 {days} 天数据' if days > 1 else '单日数据',
            yard_util_fig
        )

    def _parse_upload(contents, filename):
        """解析上传的CSV文件"""
        if contents is None:
            return None
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if '.csv' in filename.lower():
                return pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
        except Exception as e:
            print(f'文件解析失败 {filename}: {e}')
            return None

    @app.callback(
        Output('vessel-upload-status', 'children'),
        Output('container-upload-status', 'children'),
        Output('yard-upload-status', 'children'),
        Output('app-data-store', 'data'),
        [Input('upload-vessel', 'contents'),
         Input('upload-container', 'contents'),
         Input('upload-yard', 'contents'),
         Input('regenerate-btn', 'n_clicks')],
        [State('upload-vessel', 'filename'),
         State('upload-container', 'filename'),
         State('upload-yard', 'filename')]
    )
    def handle_upload_and_regenerate(v_c, c_c, y_c, regen_n, v_f, c_f, y_f):
        """处理数据上传和重新生成"""
        global vessel_df, container_df, yard_df, preprocessor
        global daily_throughput, yard_util_time, forecast_results_cache
        global rehandling_results_cache, overdue_cache, kpi_cache

        ctx = callback_context
        triggered = ctx.triggered_id

        if triggered == 'regenerate-btn' and regen_n:
            try:
                gen = PortDataGenerator(seed=np.random.randint(1, 9999))
                vessel_df, container_df, yard_df = gen.generate_all('data')
                preprocessor = DataPreprocessor(vessel_df, container_df, yard_df)
                daily_throughput = preprocessor.get_daily_throughput()
                yard_util_time = preprocessor.calculate_yard_utilization_time()
                forecast_results_cache.clear()
                rehandling_results_cache = None
                overdue_cache = None
                kpi_cache = None
                return ('✓ 使用新模拟数据', '✓ 使用新模拟数据', '✓ 使用新模拟数据',
                        {'regenerated': True, 'ts': datetime.now().isoformat()})
            except Exception as e:
                return (f'✗ 失败: {str(e)[:30]}', '', '', {'error': str(e)})

        statuses = ['', '', '']
        all_loaded = True

        try:
            if v_c and v_f:
                df = _parse_upload(v_c, v_f)
                if df is not None:
                    vessel_df = df
                    statuses[0] = f'✓ 已加载 {len(df)} 条记录'
                else:
                    statuses[0] = '✗ 解析失败'
                    all_loaded = False

            if c_c and c_f:
                df = _parse_upload(c_c, c_f)
                if df is not None:
                    container_df = df
                    statuses[1] = f'✓ 已加载 {len(df)} 条记录'
                else:
                    statuses[1] = '✗ 解析失败'
                    all_loaded = False

            if y_c and y_f:
                df = _parse_upload(y_c, y_f)
                if df is not None:
                    yard_df = df
                    statuses[2] = f'✓ 已加载 {len(df)} 个区域'
                else:
                    statuses[2] = '✗ 解析失败'
                    all_loaded = False
        except Exception as e:
            return (statuses[0] or '', statuses[1] or '', statuses[2] or f'✗ {str(e)[:30]}', {})

        if (v_c or c_c or y_c) and vessel_df is not None and container_df is not None and yard_df is not None:
            try:
                preprocessor = DataPreprocessor(vessel_df, container_df, yard_df)
                daily_throughput = preprocessor.get_daily_throughput()
                yard_util_time = preprocessor.calculate_yard_utilization_time()
                forecast_results_cache.clear()
                rehandling_results_cache = None
                overdue_cache = None
                kpi_cache = None
                return (statuses[0] or '', statuses[1] or '', statuses[2] or '',
                        {'uploaded': True, 'ts': datetime.now().isoformat()})
            except Exception as e:
                return (statuses[0] or '', statuses[1] or '', f'✗ 预处理失败: {str(e)[:30]}', {})

        return (statuses[0] or '', statuses[1] or '', statuses[2] or '', {})

    @app.callback(
        Output('forecast-chart', 'figure'),
        Output('forecast-metrics-row', 'children'),
        Output('residuals-row', 'children'),
        Output('forecast-loading', 'children'),
        [Input('run-forecast-btn', 'n_clicks')],
        [State('forecast-algorithm', 'value'),
         State('forecast-granularity', 'value'),
         State('forecast-train-days', 'value'),
         State('forecast-steps', 'value')]
    )
    def run_forecast(n_clicks, algorithm, granularity, train_days, steps):
        """运行吞吐量预测"""
        global preprocessor, forecast_results_cache

        empty_fig = go.Figure()
        empty_fig.update_layout(title='点击"运行预测模型"开始预测', height=400,
                                xaxis=dict(showgrid=False, showticklabels=False),
                                yaxis=dict(showgrid=False, showticklabels=False))

        if preprocessor is None:
            return empty_fig, [], [], []

        ctx = callback_context
        if ctx.triggered_id != 'run-forecast-btn' or not n_clicks:
            return empty_fig, [], [], []

        results = {}
        series = preprocessor.get_throughput_series(granularity=granularity, teu_mode=False)

        algos = ['ARIMA', 'Prophet', 'LSTM'] if algorithm == 'ALL' else [algorithm]

        for algo in algos:
            try:
                forecaster = ThroughputForecaster(algorithm=algo)
                forecaster.fit(series, train_days=train_days)
                result = forecaster.forecast(steps=steps, confidence=0.8)
                results[algo] = result
            except Exception as e:
                print(f'{algo} 预测失败: {e}')
                traceback.print_exc()
                results[algo] = {'error': str(e)}

        forecast_results_cache.update(results)

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
                            html.Span(mape_str, className=f'text-{color} fw-bold', style={'fontSize': '28px'}),
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

        return fig, cards, residuals_figs, []

    @app.callback(
        Output('yard-heatmap', 'figure'),
        Output('yard-snapshot-time', 'children'),
        Output('yard-snapshot-stats', 'children'),
        Output('yard-area-info', 'data'),
        [Input('yard-area-select', 'value'),
         Input('yard-time-slider', 'value')]
    )
    def update_yard_heatmap(area_id, slider_val):
        """更新堆场热力图"""
        global preprocessor, container_df, yard_df

        empty_fig = go.Figure()
        empty_fig.update_layout(title='请选择堆场区域', height=450)

        if preprocessor is None or not area_id:
            return empty_fig, '-', [], []

        date_min = container_df['进场时间'].min()
        date_max = container_df['出场时间'].max()
        total_seconds = (date_max - date_min).total_seconds()
        target_ts = date_min + timedelta(seconds=total_seconds * (slider_val / 100))

        snapshot = preprocessor.get_yard_snapshot(target_ts)
        heatmap, bay_labels, row_labels = preprocessor.build_heatmap_data(snapshot, area_id)

        yard_row = yard_df[yard_df['区域编号'] == area_id]
        if len(yard_row) == 0:
            return empty_fig, str(target_ts), [], []

        yr = yard_row.iloc[0]
        max_tiers = int(yr['最大堆叠层数'])
        area_snap = snapshot[snapshot['堆场区域'] == area_id]
        total_boxes = len(area_snap)
        cap = int(yr['贝位数量']) * int(yr['列数']) * max_tiers
        util = total_boxes / cap * 100 if cap > 0 else 0

        heavy = len(area_snap[area_snap['状态'] == '重箱'])
        empty = len(area_snap[area_snap['状态'] == '空箱'])
        reefer = len(area_snap[area_snap['状态'] == '冷藏'])
        danger = len(area_snap[area_snap['状态'] == '危品'])

        fig = ChartBuilder.create_yard_heatmap(heatmap, bay_labels, row_labels, max_tiers, area_id)
        time_str = target_ts.strftime('%Y-%m-%d %H:%M') + f' ({slider_val}%)'

        stats = html.Div([
            dbc.Row([
                dbc.Col([
                    html.Small('占用箱数', className='text-muted'),
                    html.H6(f'{total_boxes}', className='fw-bold text-primary')
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
                    html.H6(f'{empty}', className='text-secondary')
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

        return fig, time_str, stats, area_info

    @app.callback(
        Output('overdue-summary-cards', 'children'),
        Output('overdue-by-customer-chart', 'figure'),
        Output('overdue-by-area-chart', 'figure'),
        Output('overdue-customer-table', 'data'),
        Output('overdue-customer-table', 'columns'),
        Output('overdue-area-table', 'data'),
        Output('overdue-area-table', 'columns'),
        [Input('calc-overdue-btn', 'n_clicks')],
        [State('free-days-input', 'value')]
    )
    def update_overdue_analysis(n_clicks, free_days):
        """更新超期箱分析"""
        global preprocessor, overdue_cache

        empty_fig = go.Figure()
        empty_fig.update_layout(title='点击"计算超期箱"按钮', height=350)
        empty_cards = dbc.Alert('请先设置免费期天数并点击计算', color='secondary')

        ctx = callback_context
        if ctx.triggered_id != 'calc-overdue-btn' or not n_clicks:
            return empty_cards, empty_fig, empty_fig, [], [], [], []

        if preprocessor is None:
            return empty_cards, empty_fig, empty_fig, [], [], [], []

        try:
            free_days = int(free_days) if free_days else 5
            analyzer = OverdueBoxAnalyzer(free_days=free_days)
            overdue_df, by_customer, by_area, by_route = analyzer.analyze(preprocessor.container_df)
            overdue_cache = {'df': overdue_df, 'customer': by_customer,
                             'area': by_area, 'free_days': free_days}

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
                        html.Small(f'超期比例 {total_overdue/len(preprocessor.container_df)*100:.1f}%',
                                  className='text-muted')
                    ])
                ], color='info', outline=True), md=4),
            ])

            cust_chart = ChartBuilder.create_overdue_bar_chart(
                by_customer, 'TOP 10 客户 - 超期箱数量', '超期箱数量', '客户', top_n=10
            )
            area_chart = ChartBuilder.create_overdue_bar_chart(
                by_area, '各区域 - 预估滞箱费用 (元)', '预估滞箱费用', '堆场区域', top_n=10
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

            return (cards, cust_chart, area_chart,
                    by_customer.to_dict('records'), cust_cols,
                    by_area.to_dict('records'), area_cols)
        except Exception as e:
            traceback.print_exc()
            return dbc.Alert(f'计算失败: {str(e)}', color='danger'), empty_fig, empty_fig, [], [], [], []

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
        [State('sim-count', 'value'), State('fill-ratio', 'value')]
    )
    def run_rehandling_simulation(n_clicks, sim_count, fill_ratio):
        """运行翻箱率蒙特卡洛模拟"""
        global preprocessor, yard_df, rehandling_results_cache

        empty_fig = go.Figure()
        empty_fig.update_layout(title='点击"运行蒙特卡洛模拟"', height=400)
        dashes = ['-'] * 6

        ctx = callback_context
        if ctx.triggered_id != 'run-simulation-btn' or not n_clicks:
            if rehandling_results_cache is not None:
                return _format_rehandling_results(rehandling_results_cache)
            return empty_fig, [], [], [], *dashes

        if preprocessor is None:
            return empty_fig, [], [], [], *dashes

        try:
            simulator = YardStackSimulator(yard_df, num_simulations=sim_count,
                                           seed=np.random.randint(1, 9999))

            all_results = []
            for area_id in yard_df['区域编号'].tolist():
                area_results = simulator.run_single_area_simulation(
                    area_id, fill_ratio=fill_ratio
                )
                for strategy, result in area_results.items():
                    all_results.append({
                        '区域': area_id,
                        '策略': strategy,
                        '平均翻箱率': result.avg_relocation_rate,
                        'P95翻箱率': result.p95_relocation_rate,
                        '总提箱次数': result.total_pickups,
                        '总翻箱次数': result.total_relocations
                    })

            result_df = pd.DataFrame(all_results)
            rehandling_results_cache = result_df
            return _format_rehandling_results(result_df)
        except Exception as e:
            traceback.print_exc()
            return (empty_fig.update_layout(title=f'模拟失败: {str(e)[:50]}'),
                    [], [], [], *dashes)

    def _format_rehandling_results(result_df):
        """格式化翻箱率结果"""
        fig = ChartBuilder.create_rehandling_comparison(result_df)

        strat_avg = result_df.groupby('策略').agg(
            平均翻箱率=('平均翻箱率', 'mean'),
            P95翻箱率=('P95翻箱率', 'mean')
        )

        vals = []
        for s in ['策略A-随机堆放', '策略B-按提箱时间排序', '策略C-按航线聚堆']:
            if s in strat_avg.index:
                vals.append(f'{strat_avg.loc[s, "平均翻箱率"]*100:.2f}%')
                vals.append(f'{strat_avg.loc[s, "P95翻箱率"]*100:.2f}%')
            else:
                vals += ['-', '-']

        cols = [
            {'name': '区域', 'id': '区域'},
            {'name': '策略', 'id': '策略'},
            {'name': '平均翻箱率', 'id': '平均翻箱率',
             'format': Format(precision=4, scheme=Scheme.percent)},
            {'name': 'P95翻箱率', 'id': 'P95翻箱率',
             'format': Format(precision=4, scheme=Scheme.percent)},
            {'name': '总提箱次数', 'id': '总提箱次数',
             'type': 'numeric', 'format': Format(group=',')},
            {'name': '总翻箱次数', 'id': '总翻箱次数',
             'type': 'numeric', 'format': Format(group=',')},
        ]

        display_df = result_df.copy()
        return (fig, display_df.to_dict('records'), cols, [], *vals)

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
        [Input('kpi-granularity', 'value')]
    )
    def update_kpi_dashboard(granularity):
        """更新KPI仪表板"""
        global preprocessor, kpi_cache

        empty_fig = go.Figure()
        empty_fig.update_layout(title='KPI趋势图加载中...', height=550)
        dashes = ['-'] * 8

        if preprocessor is None:
            return *dashes, empty_fig

        try:
            kpi_data = preprocessor.calculate_kpis(granularity=granularity)
            kpi_cache = kpi_data
            df = kpi_data.get('综合KPI', pd.DataFrame())
            if df.empty:
                return *dashes, empty_fig

            latest = df.iloc[-1]

            def make_trend_badge(col, is_higher_better=True):
                if len(df) < 2 or col not in df.columns:
                    return ''
                curr = df[col].iloc[-1]
                prev = df[col].iloc[-2]
                if pd.isna(curr) or pd.isna(prev) or prev == 0:
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
            return *dashes, empty_fig

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
         State('whatif-reduce-days', 'value')]
    )
    def run_whatif_analysis(n1, n2, n3, n_all, growth, bays, rows, tiers, reduce_days):
        """运行What-if情景分析"""
        global preprocessor

        empty_fig = go.Figure()
        empty_fig.update_layout(title='选择情景参数后点击"运行"按钮', height=450)

        ctx = callback_context
        triggered = ctx.triggered_id
        if preprocessor is None or (not n1 and not n2 and not n3 and not n_all):
            return empty_fig, [], [], []

        try:
            scenario = WhatIfScenario(preprocessor)
            results = []

            if triggered in ['run-whatif-1', 'run-whatif-all']:
                results.append(scenario.scenario_throughput_growth(growth))

            if triggered in ['run-whatif-2', 'run-whatif-all']:
                bays_i = int(bays) if bays else 10
                rows_i = int(rows) if rows else 8
                tiers_i = int(tiers) if tiers else 6
                results.append(scenario.scenario_new_yard_area(bays_i, rows_i, tiers_i))

            if triggered in ['run-whatif-3', 'run-whatif-all']:
                reduce_i = int(reduce_days) if reduce_days else 2
                results.append(scenario.scenario_reduce_storage_days(reduce_i))

            if not results and triggered == 'run-whatif-all':
                for g in [0.1, 0.2, 0.3]:
                    results.append(scenario.scenario_throughput_growth(g))
                results.append(scenario.scenario_new_yard_area(10, 8, 6))
                for d in [1, 2, 3]:
                    results.append(scenario.scenario_reduce_storage_days(d))

            if not results:
                return empty_fig, [], [], []

            result_df = pd.DataFrame(results)
            fig = ChartBuilder.create_whatif_comparison_chart(result_df)

            cols = [{'name': c, 'id': c} for c in result_df.columns]
            for c in cols:
                n = c['name']
                if any(k in n for k in ['利用率', '比例', '幅度', '增长率']):
                    c['type'] = 'numeric'
                    c['format'] = Format(precision=2, scheme=Scheme.percent)
                elif any(k in n for k in ['费用', '容量', '数量', '天数']):
                    c['type'] = 'numeric'
                    c['format'] = Format(group=',', precision=0)

            return fig, result_df.to_dict('records'), cols, []
        except Exception as e:
            traceback.print_exc()
            return (empty_fig.update_layout(title=f'分析失败: {str(e)[:50]}'),
                    [], [], [])

    @app.callback(
        Output('download-pdf', 'data'),
        Output('export-status', 'children'),
        [Input('export-pdf-btn', 'n_clicks')]
    )
    def export_pdf_report(n_clicks):
        """导出PDF运营分析报告"""
        global preprocessor, daily_throughput, yard_util_time
        global forecast_results_cache, rehandling_results_cache
        global overdue_cache, kpi_cache

        ctx = callback_context
        if ctx.triggered_id != 'export-pdf-btn' or not n_clicks:
            return None, ''

        if preprocessor is None:
            return None, html.Span('✗ 暂无数据可导出', className='text-danger')

        try:
            generator = PortReportGenerator('reports')
            start_date = daily_throughput['日期'].min()
            end_date = daily_throughput['日期'].max()

            by_customer = overdue_cache['customer'] if overdue_cache else pd.DataFrame()
            by_area = overdue_cache['area'] if overdue_cache else pd.DataFrame()

            rehandling_df = rehandling_results_cache if rehandling_results_cache is not None else pd.DataFrame()

            whatif_df = pd.DataFrame()
            try:
                scenario = WhatIfScenario(preprocessor)
                whatif_df = scenario.run_all_scenarios()
            except Exception:
                pass

            figures = {}
            try:
                figures['throughput_chart'] = ChartBuilder.create_throughput_chart(daily_throughput)
            except Exception:
                pass
            try:
                figures['forecast_chart'] = ChartBuilder.create_forecast_comparison(forecast_results_cache)
            except Exception:
                pass
            try:
                figures['yard_util_chart'] = ChartBuilder.create_yard_utilization_chart(yard_util_time)
            except Exception:
                pass
            try:
                figures['rehandling_chart'] = ChartBuilder.create_rehandling_comparison(rehandling_df)
            except Exception:
                pass

            filepath = generator.generate_report(
                report_name='港口运营分析',
                start_date=start_date, end_date=end_date,
                daily_throughput=daily_throughput,
                forecast_results=forecast_results_cache,
                yard_util=yard_util_time,
                rehandling_results=rehandling_df,
                overdue_by_customer=by_customer,
                overdue_by_area=by_area,
                kpi_data=kpi_cache or preprocessor.calculate_kpis('日'),
                whatif_results=whatif_df,
                figures=figures
            )

            return (dcc.send_file(filepath),
                    html.Span([
                        html.I(className='fas fa-check-circle me-1 text-success'),
                        f'✓ 报告已生成: {os.path.basename(filepath)}'
                    ], className='text-success fw-bold'))
        except Exception as e:
            traceback.print_exc()
            return None, html.Span(f'✗ 导出失败: {str(e)[:60]}', className='text-danger')
