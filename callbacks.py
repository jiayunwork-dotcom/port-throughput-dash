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

            _debug(f'超期箱分析完成: {total_overdue}个, ¥{total_fee:,.0f}')
            return (cards, cust_chart, area_chart,
                    by_customer.to_dict('records'), cust_cols,
                    by_area.to_dict('records'), area_cols)
        except Exception as e:
            traceback.print_exc()
            return (dbc.Alert(f'计算失败: {str(e)[:60]}', color='danger'),
                    ef_cust, ef_area, [], [], [], [])

    # =============================================
    # 辅助: 格式化翻箱率结果
    # =============================================
    def _format_rehandling_results(result_df):
        """格式化翻箱率结果（内部函数）"""
        fig = ChartBuilder.create_rehandling_comparison(result_df)

        strategy_order = [
            '策略A-随机堆放', '策略B-按提箱时间排序', '策略C-按航线聚堆'
        ]
        strat_avg = result_df.groupby('策略').agg(
            平均翻箱率=('平均翻箱率', 'mean'),
            P95翻箱率=('P95翻箱率', 'mean')
        )

        vals = []
        for s in strategy_order:
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

        return (fig, result_df.to_dict('records'), cols, [], *vals)

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
        _debug(f'run_rehandling_simulation触发 n_clicks={n_clicks}')

        ef = empty_figure(
            '点击"运行蒙特卡洛模拟"开始分析（首次需数分钟）'
        )
        dashes = ['-'] * 6

        if not S.is_initialized():
            return (ef, [], [], [], *dashes)

        # 初始加载: 显示提示
        if n_clicks is None or n_clicks == 0:
            if S.rehandling_results_cache is not None:
                return _format_rehandling_results(S.rehandling_results_cache)
            return (ef, [], [], [], *dashes)

        # 点击运行: 执行模拟
        try:
            simulator = YardStackSimulator(
                S.yard_df, num_simulations=sim_count or 1000,
                seed=np.random.randint(1, 9999)
            )
            all_results = []
            # 模拟前4个代表性区域（缩短时间）
            areas = S.yard_df['区域编号'].tolist()[:4]
            for area_id in areas:
                area_results = simulator.run_single_area_simulation(
                    area_id, fill_ratio=fill_ratio or 0.7
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
            S.rehandling_results_cache = result_df
            _debug(f'翻箱率模拟完成: {len(result_df)} 条记录')
            return _format_rehandling_results(result_df)
        except Exception as e:
            traceback.print_exc()
            return (empty_figure(f'模拟失败: {str(e)[:50]}'), [], [], [], *dashes)

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
