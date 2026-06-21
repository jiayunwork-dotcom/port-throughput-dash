"""
Plotly图表创建工具模块
为Dash面板生成各类交互图表
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta


class ChartBuilder:
    """图表构建器"""

    COLORS = {
        'primary': '#1a365d',
        'secondary': '#2b6cb0',
        'accent': '#3182ce',
        'success': '#38a169',
        'warning': '#d69e2e',
        'danger': '#e53e3e',
        'info': '#319795',
        'purple': '#805ad5',
        'gray': '#718096',
        'light_gray': '#e2e8f0'
    }

    PALETTE = ['#2b6cb0', '#38a169', '#d69e2e', '#e53e3e', '#805ad5',
               '#319795', '#dd6b20', '#3182ce', '#2f855a', '#b7791f']

    @classmethod
    def _update_layout(cls, fig, title='', xlabel='', ylabel='', height=500,
                       showlegend=True, legend_orientation='h'):
        """统一更新图表布局"""
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center',
                       font=dict(size=16, color=cls.COLORS['primary'])),
            xaxis=dict(title=xlabel, showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(title=ylabel, showgrid=True, gridcolor='#f0f0f0'),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=height,
            margin=dict(l=60, r=30, t=60, b=60),
            legend=dict(
                orientation=legend_orientation,
                yanchor='bottom',
                y=1.02 if legend_orientation == 'h' else None,
                xanchor='right' if legend_orientation == 'h' else 'left',
                x=1 if legend_orientation == 'h' else None,
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.9)'
            ),
            showlegend=showlegend,
            hovermode='x unified'
        )
        return fig

    @classmethod
    def create_throughput_chart(cls, daily_df: pd.DataFrame) -> go.Figure:
        """创建吞吐量趋势图"""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_df['日期'], y=daily_df['进场箱量'],
            name='进场箱量', marker_color=cls.PALETTE[0], opacity=0.8
        ))
        fig.add_trace(go.Bar(
            x=daily_df['日期'], y=daily_df['出场箱量'],
            name='出场箱量', marker_color=cls.PALETTE[1], opacity=0.8
        ))
        fig.add_trace(go.Scatter(
            x=daily_df['日期'], y=daily_df['总TEU'],
            name='总TEU', yaxis='y2', mode='lines+markers',
            line=dict(color=cls.COLORS['warning'], width=3),
            marker=dict(size=5)
        ))
        fig.update_layout(
            barmode='stack',
            yaxis2=dict(
                title='TEU',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
        return cls._update_layout(fig,
            title='每日吞吐量趋势',
            xlabel='日期', ylabel='箱量', height=450)

    @classmethod
    def create_forecast_comparison(cls, forecast_results: dict,
                                   selected_algos: list = None) -> go.Figure:
        """创建预测对比图（实际值vs预测值）"""
        if selected_algos is None:
            selected_algos = list(forecast_results.keys())

        fig = go.Figure()
        colors = [cls.PALETTE[i % len(cls.PALETTE)] for i in range(len(selected_algos))]

        for idx, algo in enumerate(selected_algos):
            result = forecast_results.get(algo)
            if not isinstance(result, dict):
                continue

            hist_dates = result.get('hist_dates')
            actual = result.get('hist_actual')

            if idx == 0 and hist_dates is not None and actual is not None:
                fig.add_trace(go.Scatter(
                    x=hist_dates, y=actual,
                    name='实际值', mode='lines',
                    line=dict(color=cls.COLORS['primary'], width=2.5),
                    legendgroup='actual'
                ))

            fitted = result.get('hist_fitted')
            fitted_dates = result.get('hist_fitted_dates')
            if fitted is not None and len(fitted) > 0 and fitted_dates is not None:
                min_len = min(len(fitted_dates), len(fitted))
                fig.add_trace(go.Scatter(
                    x=fitted_dates[-min_len:], y=fitted[-min_len:],
                    name=f'{algo} 拟合', mode='lines',
                    line=dict(color=colors[idx], width=1.5, dash='dash'),
                    opacity=0.7
                ))

            fc_dates = result.get('forecast_dates')
            fc_mean = result.get('forecast_mean')
            fc_lower = result.get('forecast_lower')
            fc_upper = result.get('forecast_upper')

            if fc_dates is not None and fc_mean is not None:
                fig.add_trace(go.Scatter(
                    x=fc_dates, y=fc_mean,
                    name=f'{algo} 预测', mode='lines+markers',
                    line=dict(color=colors[idx], width=3),
                    marker=dict(size=7, symbol='diamond')
                ))
                if fc_lower is not None and fc_upper is not None:
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([fc_dates, fc_dates[::-1]]),
                        y=np.concatenate([fc_upper, fc_lower[::-1]]),
                        fill='toself', fillcolor=colors[idx],
                        opacity=0.15, line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False, hoverinfo='skip',
                        name=f'{algo} 80%置信区间'
                    ))

        return cls._update_layout(fig,
            title='吞吐量时序预测对比（含80%置信区间）',
            xlabel='日期', ylabel='吞吐量（箱量）', height=500)

    @classmethod
    def create_residual_histogram(cls, residuals: np.ndarray, algo: str = '') -> go.Figure:
        """创建残差分布直方图"""
        if residuals is None or len(residuals) == 0:
            fig = go.Figure()
            fig.add_annotation(text='残差数据不可用', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title=f'{algo} 残差分布', height=350)

        residuals = np.array(residuals)
        valid = residuals[~np.isnan(residuals)]

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid, nbinsx=30,
            name='残差频次',
            marker=dict(color=cls.PALETTE[0], line=dict(color='white', width=1)),
            histnorm='probability density', opacity=0.8
        ))
        if len(valid) > 1:
            mean_r = np.mean(valid)
            std_r = np.std(valid)
            x_range = np.linspace(valid.min(), valid.max(), 100)
            normal_pdf = (1 / (std_r * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x_range - mean_r) / std_r) ** 2)
            fig.add_trace(go.Scatter(
                x=x_range, y=normal_pdf,
                name='正态分布拟合', mode='lines',
                line=dict(color=cls.COLORS['danger'], width=2, dash='dash')
            ))

        return cls._update_layout(fig,
            title=f'{algo} 残差分布直方图（正态性检验）',
            xlabel='残差值', ylabel='概率密度', height=350,
            showlegend=True, legend_orientation='v')

    @classmethod
    def create_metric_cards(cls, forecast_results: dict) -> pd.DataFrame:
        """生成指标卡片数据"""
        records = []
        for algo, result in forecast_results.items():
            if isinstance(result, dict):
                mape = result.get('mape', np.nan)
                rmse = result.get('rmse', np.nan)
                records.append({
                    '算法': algo,
                    'MAPE (%)': f'{mape:.2f}' if not np.isnan(mape) else '-',
                    'RMSE': f'{rmse:.0f}' if not np.isnan(rmse) else '-'
                })
        return pd.DataFrame(records)

    @classmethod
    def create_yard_heatmap(cls, heatmap_data: np.ndarray,
                            bay_labels: list, row_labels: list,
                            max_tiers: int, area_id: str) -> go.Figure:
        """创建堆场俯视热力图"""
        if heatmap_data is None or heatmap_data.size == 0:
            fig = go.Figure()
            fig.add_annotation(text=f'区域 {area_id} 暂无数据',
                               xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig,
                title=f'堆场区域 {area_id} 状态快照',
                xlabel='列 (Row)', ylabel='贝位 (Bay)', height=450)

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=row_labels,
            y=bay_labels,
            colorscale=[
                [0.0, '#ebf8ff'],
                [0.2, '#bee3f8'],
                [0.4, '#63b3ed'],
                [0.6, '#3182ce'],
                [0.8, '#2c5282'],
                [1.0, '#1a365d']
            ],
            zmin=0, zmax=max_tiers,
            hoverongaps=False,
            colorbar=dict(
                title=dict(text='堆叠层数', side='right'),
                tickmode='linear', dtick=1
            ),
            text=heatmap_data.astype(int),
            texttemplate='%{text}层',
            textfont=dict(size=9, color='white')
        ))

        fig.update_layout(
            xaxis=dict(side='top', fixedrange=True),
            yaxis=dict(autorange='reversed', fixedrange=True)
        )
        return cls._update_layout(fig,
            title=f'堆场区域 {area_id} 俯视热力图（颜色深度=堆叠高度）',
            xlabel='列号', ylabel='贝位号', height=450, showlegend=False)

    @classmethod
    def create_yard_utilization_chart(cls, yard_util_df: pd.DataFrame) -> go.Figure:
        """创建堆场利用率趋势图"""
        fig = go.Figure()
        areas = yard_util_df['区域'].unique()

        for i, area in enumerate(areas):
            area_data = yard_util_df[yard_util_df['区域'] == area].sort_values('日期')
            fig.add_trace(go.Scatter(
                x=area_data['日期'], y=area_data['空间占用率'],
                name=f'{area} 区', mode='lines',
                line=dict(color=cls.PALETTE[i % len(cls.PALETTE)], width=2),
                stackgroup=None
            ))

        overall = yard_util_df.groupby('日期').agg(
            总占用=('占用箱数', 'sum'),
            总容量=('总容量', 'sum')
        ).reset_index()
        overall['整体利用率'] = overall['总占用'] / overall['总容量']
        fig.add_trace(go.Scatter(
            x=overall['日期'], y=overall['整体利用率'],
            name='整体平均', mode='lines',
            line=dict(color=cls.COLORS['danger'], width=3, dash='dash')
        ))
        fig.add_hline(
            y=0.85, line_dash='dash', line_color=cls.COLORS['warning'],
            annotation_text='高负荷线 (85%)',
            annotation_position='bottom right'
        )
        fig.update_yaxes(tickformat=',.0%')
        return cls._update_layout(fig,
            title='堆场各区域空间占用率趋势',
            xlabel='日期', ylabel='占用率', height=450)

    @classmethod
    def create_rehandling_comparison(cls, rehandling_df: pd.DataFrame) -> go.Figure:
        """创建翻箱率策略对比图"""
        if rehandling_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='翻箱率分析中，请稍候...',
                               xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='翻箱率策略对比', height=450)

        strategy_avg = rehandling_df.groupby('策略').agg(
            平均翻箱率=('平均翻箱率', 'mean'),
            P95翻箱率=('P95翻箱率', 'mean')
        ).reset_index()

        fig = go.Figure()
        x = strategy_avg['策略'].values
        fig.add_trace(go.Bar(
            x=x, y=strategy_avg['平均翻箱率'] * 100,
            name='平均翻箱率',
            marker_color=cls.PALETTE[0],
            text=[f'{v:.1f}%' for v in strategy_avg['平均翻箱率'] * 100],
            textposition='outside'
        ))
        fig.add_trace(go.Bar(
            x=x, y=strategy_avg['P95翻箱率'] * 100,
            name='P95翻箱率',
            marker_color=cls.PALETTE[3],
            text=[f'{v:.1f}%' for v in strategy_avg['P95翻箱率'] * 100],
            textposition='outside'
        ))

        return cls._update_layout(fig,
            title='三种堆存策略翻箱率对比（蒙特卡洛1000次模拟）',
            xlabel='堆存策略', ylabel='翻箱率 (%)', height=450)

    @classmethod
    def create_overdue_bar_chart(cls, df: pd.DataFrame, title: str,
                                  value_col: str, label_col: str,
                                  top_n: int = 10) -> go.Figure:
        """创建超期箱分组柱状图"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text='暂无超期箱数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title=title, height=400)

        top_data = df.head(top_n).sort_values(value_col, ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_data[value_col],
            y=top_data[label_col],
            orientation='h',
            marker=dict(
                color=top_data[value_col],
                colorscale=[
                    [0, '#63b3ed'],
                    [0.5, '#ecc94b'],
                    [1.0, '#fc8181']
                ],
                showscale=False
            ),
            text=[f'{v:,.0f}' for v in top_data[value_col]],
            textposition='outside',
            width=0.7
        ))
        return cls._update_layout(fig, title=title,
            xlabel=value_col, ylabel='', height=400, showlegend=False)

    @classmethod
    def create_kpi_trend_chart(cls, kpi_df: pd.DataFrame) -> go.Figure:
        """创建KPI趋势多子图"""
        cols_available = [
            ('船时效率_TEU_小时', '船时效率 (TEU/h)', cls.PALETTE[0]),
            ('桥吊效率_TEU_台时', '桥吊效率 (TEU/台时)', cls.PALETTE[1]),
            ('泊位利用率', '泊位利用率', cls.PALETTE[3]),
            ('平均集卡周转_小时', '集卡周转 (小时)', cls.PALETTE[4])
        ]
        cols_to_plot = [(c, t, col) for c, t, col in cols_available if c in kpi_df.columns]
        n_plots = len(cols_to_plot)
        if n_plots == 0:
            fig = go.Figure()
            return cls._update_layout(fig, title='KPI趋势图', height=500)

        rows = 2 if n_plots >= 2 else 1
        cols = 2 if n_plots >= 3 else min(n_plots, 2)
        fig = make_subplots(rows=rows, cols=cols,
                            subplot_titles=[t for _, t, _ in cols_to_plot],
                            vertical_spacing=0.15, horizontal_spacing=0.12)

        all_periods = pd.to_datetime(kpi_df['period'])
        x_min = all_periods.min()
        x_max = all_periods.max()
        if pd.notnull(x_min) and pd.notnull(x_max) and x_min == x_max:
            x_min = x_min - pd.Timedelta(days=1)
            x_max = x_max + pd.Timedelta(days=1)
        x_pad = (x_max - x_min) * 0.03 if pd.notnull(x_min) and pd.notnull(x_max) and x_max > x_min else pd.Timedelta(days=1)
        x_range = [x_min - x_pad, x_max + x_pad] if pd.notnull(x_min) and pd.notnull(x_max) else None

        for idx, (col_name, title, color) in enumerate(cols_to_plot):
            r = idx // cols + 1
            c = idx % cols + 1

            vals = kpi_df[col_name].dropna()
            periods = pd.to_datetime(kpi_df.loc[vals.index, 'period'])
            mean_v = vals.mean()
            std_v = vals.std()

            fig.add_trace(go.Scatter(
                x=periods, y=vals, name=title, mode='lines+markers',
                line=dict(color=color, width=2.5),
                marker=dict(size=5)
            ), row=r, col=c)

            if col_name == '平均集卡周转_小时':
                threshold_high = mean_v + std_v
                fig.add_hline(
                    y=threshold_high, line_dash='dash',
                    line_color=cls.COLORS['danger'], line_width=1.5,
                    annotation_text='+1σ告警线', annotation_font_size=9,
                    annotation_position='bottom right',
                    row=r, col=c
                )
                fig.add_hline(
                    y=mean_v, line_dash='dot',
                    line_color=cls.COLORS['gray'], line_width=1,
                    row=r, col=c
                )
            else:
                threshold_low = mean_v - std_v
                fig.add_hline(
                    y=threshold_low, line_dash='dash',
                    line_color=cls.COLORS['danger'], line_width=1.5,
                    annotation_text='-1σ告警线', annotation_font_size=9,
                    annotation_position='bottom right',
                    row=r, col=c
                )
                fig.add_hline(
                    y=mean_v, line_dash='dot',
                    line_color=cls.COLORS['gray'], line_width=1,
                    row=r, col=c
                )

            if col_name == '泊位利用率':
                fig.update_yaxes(tickformat=',.0%', row=r, col=c)

            vals_arr = vals.values
            if len(vals_arr) > 3:
                alert_periods = []
                for i in range(2, len(vals_arr)):
                    if col_name == '平均集卡周转_小时':
                        cond = (vals_arr[i] > threshold_high and
                                vals_arr[i-1] > threshold_high and
                                vals_arr[i-2] > threshold_high)
                    else:
                        cond = (vals_arr[i] < threshold_low and
                                vals_arr[i-1] < threshold_low and
                                vals_arr[i-2] < threshold_low)
                    if cond:
                        alert_periods.append(periods.values[i])

                if alert_periods:
                    for ap in alert_periods[:5]:
                        try:
                            ap_idx = list(periods.values).index(ap)
                            fig.add_trace(go.Scatter(
                                x=[ap], y=[vals_arr[ap_idx]],
                                mode='markers', showlegend=False,
                                marker=dict(color=cls.COLORS['danger'], size=14,
                                            symbol='circle-open', line_width=3)
                            ), row=r, col=c)
                        except (ValueError, IndexError):
                            pass

        fig.update_layout(height=600,
                          title=dict(text='KPI关键运营指标趋势（连续3天异常标红）',
                                     x=0.5, xanchor='center',
                                     font=dict(size=16, color=cls.COLORS['primary'])),
                          plot_bgcolor='white', paper_bgcolor='white',
                          showlegend=False, margin=dict(l=60, r=30, t=80, b=60))

        for r in range(1, rows + 1):
            for c in range(1, cols + 1):
                fig.update_xaxes(
                    showgrid=True, gridcolor='#f0f0f0',
                    row=r, col=c
                )
                if x_range is not None:
                    fig.update_xaxes(range=x_range, row=r, col=c)
                fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0', row=r, col=c)

        return fig

    @classmethod
    def create_whatif_comparison_chart(cls, whatif_df: pd.DataFrame) -> go.Figure:
        """创建What-if情景分析对比图"""
        if whatif_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='请先运行情景分析', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='What-if情景对比', height=500)

        scenarios = whatif_df['情景名称'].tolist()
        metrics = []
        for col in whatif_df.columns:
            if col != '情景名称' and pd.api.types.is_numeric_dtype(whatif_df[col]):
                if '利用率' in col or '比例' in col or '幅度' in col:
                    metrics.append(col)
        if not metrics:
            metrics = whatif_df.select_dtypes(include=[np.number]).columns.tolist()[:4]

        metrics_display = [m for m in metrics if m in ['利用率提升幅度', '峰值利用率下降幅度',
                                                        '超期箱减少比例', '翻箱率改善幅度',
                                                        '平均利用率下降幅度', '总容量增长率']]
        if not metrics_display:
            metrics_display = metrics[:4]

        fig = go.Figure()
        for i, metric in enumerate(metrics_display):
            values = whatif_df[metric].fillna(0).values * 100
            fig.add_trace(go.Bar(
                x=scenarios, y=values, name=metric,
                marker_color=cls.PALETTE[i % len(cls.PALETTE)],
                text=[f'{v:.1f}%' for v in values],
                textposition='outside'
            ))

        return cls._update_layout(fig,
            title='What-if情景分析关键指标对比',
            xlabel='模拟情景', ylabel='变化幅度 (%)', height=500)

    @classmethod
    def create_berth_gantt_chart(cls, vessel_df: pd.DataFrame,
                                   date_start=None, date_end=None,
                                   conflict_vessel_ids: set = None) -> go.Figure:
        """创建泊位调度甘特图"""
        if vessel_df is None or vessel_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='暂无靠泊数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='泊位调度甘特图', height=500)

        if conflict_vessel_ids is None:
            conflict_vessel_ids = set()

        df = vessel_df.copy()
        df['到港时间'] = pd.to_datetime(df['到港时间'])
        df['离港时间'] = pd.to_datetime(df['离港时间'])

        if date_start is not None:
            df = df[df['离港时间'] >= pd.to_datetime(date_start)]
        if date_end is not None:
            df = df[df['到港时间'] <= pd.to_datetime(date_end)]

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text='所选时段内无靠泊记录', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='泊位调度甘特图', height=500)

        berths = sorted(df['分配泊位编号'].unique())

        def get_color(teu, is_conflict=False):
            if is_conflict:
                return '#e53e3e'
            if teu < 2000:
                return '#90cdf4'
            elif teu <= 5000:
                return '#3182ce'
            else:
                return '#1a365d'

        fig = go.Figure()

        for berth_idx, berth in enumerate(berths):
            berth_vessels = df[df['分配泊位编号'] == berth]
            for _, row in berth_vessels.iterrows():
                vessel_id = row.get('船名', '')
                is_conflict = vessel_id in conflict_vessel_ids

                hover_text = (
                    f"<b>{row['船名']}</b><br>"
                    f"泊位: {berth}<br>"
                    f"到港: {row['到港时间'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"离港: {row['离港时间'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"载箱量: {row['载箱量TEU']:,} TEU<br>"
                    f"靠泊时长: {(row['离港时间'] - row['到港时间']).total_seconds()/3600:.1f} 小时"
                )
                if is_conflict:
                    hover_text += "<br><span style='color:red'>⚠️ 泊位冲突</span>"

                fig.add_trace(go.Bar(
                    x=[(row['离港时间'] - row['到港时间']).total_seconds() / 3600],
                    y=[berth],
                    base=[row['到港时间']],
                    orientation='h',
                    marker=dict(
                        color=get_color(row['载箱量TEU'], is_conflict),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=hover_text,
                    name=berth,
                    showlegend=False,
                    width=0.6
                ))

        fig.update_layout(
            barmode='overlay',
            xaxis=dict(
                type='date',
                title='时间',
                showgrid=True,
                gridcolor='#f0f0f0',
                tickformat='%m-%d %H:%M',
                tickangle=0
            ),
            yaxis=dict(
                title='泊位',
                showgrid=True,
                gridcolor='#f0f0f0',
                categoryorder='array',
                categoryarray=list(reversed(berths))
            ),
            hovermode='closest'
        )

        if date_start is not None and date_end is not None:
            fig.update_xaxes(range=[pd.to_datetime(date_start), pd.to_datetime(date_end)])

        return cls._update_layout(fig,
            title='泊位调度甘特图（按载箱量分色）',
            xlabel='时间', ylabel='泊位编号', height=500,
            showlegend=False)

    @classmethod
    def create_berth_efficiency_chart(cls, efficiency_df: pd.DataFrame) -> go.Figure:
        """创建泊位效率排行横向条形图"""
        if efficiency_df is None or efficiency_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='暂无效率数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='泊位效率排行', height=400, showlegend=False)

        df = efficiency_df.copy()
        df = df.sort_values('平均周转时间_小时', ascending=True)

        def get_bar_color(turnaround_hours):
            if turnaround_hours < 2:
                return '#38a169'
            elif turnaround_hours > 8:
                return '#e53e3e'
            else:
                return '#3182ce'

        colors = [get_bar_color(t) for t in df['平均周转时间_小时']]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df['平均周转时间_小时'],
            y=df['泊位编号'],
            orientation='h',
            marker=dict(color=colors),
            text=[f'{t:.2f} h' for t in df['平均周转时间_小时']],
            textposition='outside',
            hovertemplate=(
                '<b>%{y}</b><br>'
                '平均周转时间: %{x:.2f} 小时<br>'
                '靠泊船次: %{customdata[0]} 艘<br>'
                '累计装卸: %{customdata[1]:,} TEU'
            ),
            customdata=df[['靠泊船次', '累计装卸TEU']].values
        ))

        fig.update_yaxes(categoryorder='array', categoryarray=df['泊位编号'].tolist())

        return cls._update_layout(fig,
            title='泊位效率排行（按周转时间）',
            xlabel='平均周转时间 (小时)', ylabel='泊位编号',
            height=400, showlegend=False)

    @classmethod
    def create_simulation_timeline_chart(cls, timeline_df: pd.DataFrame) -> go.Figure:
        """创建仿真时间轴折线图：锚地等待数 + 在泊作业数"""
        if timeline_df is None or timeline_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='暂无仿真数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='锚地与在泊船舶数量变化', height=400)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeline_df['时间_小时'], y=timeline_df['锚地等待数'],
            name='锚地等待数', mode='lines',
            line=dict(color=cls.COLORS['warning'], width=2.5),
            fill='tozeroy', fillcolor='rgba(214, 158, 46, 0.15)',
            stackgroup=None
        ))
        fig.add_trace(go.Scatter(
            x=timeline_df['时间_小时'], y=timeline_df['在泊作业数'],
            name='在泊作业数', mode='lines',
            line=dict(color=cls.COLORS['primary'], width=2.5),
            fill='tozeroy', fillcolor='rgba(26, 54, 93, 0.15)',
        ))

        return cls._update_layout(fig,
            title='锚地等待与在泊作业船舶数量变化',
            xlabel='仿真时间 (小时)', ylabel='船舶数量 (艘)',
            height=400)

    @classmethod
    def create_simulation_berth_gantt(cls, berth_df: pd.DataFrame, sim_duration: float) -> go.Figure:
        """创建仿真泊位甘特图（复用现有泊位甘特图风格）"""
        if berth_df is None or berth_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='暂无仿真数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='仿真泊位甘特图', height=400)

        df = berth_df.copy()
        berths = sorted(df['泊位编号'].unique())

        def get_color(teu):
            if teu < 2000:
                return '#90cdf4'
            elif teu <= 5000:
                return '#3182ce'
            else:
                return '#1a365d'

        fig = go.Figure()

        for berth in berths:
            berth_vessels = df[df['泊位编号'] == berth]
            for _, row in berth_vessels.iterrows():
                duration = row['结束时间'] - row['开始时间']
                hover_text = (
                    f"<b>{row['船名']}</b><br>"
                    f"泊位: {berth}<br>"
                    f"开始: {row['开始时间']:.1f} 小时<br>"
                    f"结束: {row['结束时间']:.1f} 小时<br>"
                    f"载箱量: {row['载箱量TEU']:,} TEU<br>"
                    f"作业时长: {duration:.1f} 小时"
                )

                fig.add_trace(go.Bar(
                    x=[duration],
                    y=[berth],
                    base=[row['开始时间']],
                    orientation='h',
                    marker=dict(
                        color=get_color(row['载箱量TEU']),
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate=hover_text,
                    name=berth,
                    showlegend=False,
                    width=0.6
                ))

        fig.update_layout(
            barmode='overlay',
            xaxis=dict(
                title='仿真时间 (小时)',
                showgrid=True,
                gridcolor='#f0f0f0',
            ),
            yaxis=dict(
                title='泊位',
                showgrid=True,
                gridcolor='#f0f0f0',
                categoryorder='array',
                categoryarray=list(reversed(berths))
            ),
            hovermode='closest'
        )

        if sim_duration:
            fig.update_xaxes(range=[0, sim_duration])

        return cls._update_layout(fig,
            title='仿真泊位甘特图（按载箱量分色）',
            xlabel='时间 (小时)', ylabel='泊位编号',
            height=400, showlegend=False)

    @classmethod
    def create_wait_time_histogram(cls, ships: list) -> go.Figure:
        """创建等待时长分布直方图"""
        if ships is None or len(ships) == 0:
            fig = go.Figure()
            fig.add_annotation(text='暂无仿真数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='等待时长分布直方图', height=400)

        wait_times = [s.start_time - s.arrival_time for s in ships
                       if hasattr(s, 'start_time') and s.start_time > 0
                       and hasattr(s, 'departure_time') and s.departure_time > 0
                       and not getattr(s, 'rejected', False)]

        if not wait_times:
            fig = go.Figure()
            fig.add_annotation(text='无等待数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='等待时长分布直方图', height=400)

        wait_times = np.array(wait_times)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=wait_times,
            nbinsx=20,
            name='船舶数量',
            marker=dict(
                color=cls.PALETTE[0],
                line=dict(color='white', width=1),
                opacity=0.85
            ),
            histnorm=''
        ))

        avg_wait = np.mean(wait_times)
        fig.add_vline(
            x=avg_wait, line_dash='dash', line_color=cls.COLORS['danger'], line_width=2,
            annotation_text=f'平均: {avg_wait:.1f} h',
            annotation_position='top right'
        )

        return cls._update_layout(fig,
            title='船舶等待时长分布直方图',
            xlabel='等待时长 (小时)', ylabel='船舶数量 (艘)',
            height=400, showlegend=False)

    @classmethod
    def create_sensitivity_chart(cls, sensitivity_df: pd.DataFrame) -> go.Figure:
        """创建敏感性分析折线图"""
        if sensitivity_df is None or sensitivity_df.empty:
            fig = go.Figure()
            fig.add_annotation(text='暂无敏感性分析数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='敏感性分析', height=450)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=sensitivity_df['到港间隔_小时'],
            y=sensitivity_df['平均等待时长_小时'],
            name='平均等待时长 (小时)',
            mode='lines+markers',
            line=dict(color=cls.PALETTE[0], width=3),
            marker=dict(size=8, symbol='circle'),
            yaxis='y1'
        ))

        fig.add_trace(go.Scatter(
            x=sensitivity_df['到港间隔_小时'],
            y=sensitivity_df['泊位平均利用率'] * 100,
            name='泊位平均利用率 (%)',
            mode='lines+markers',
            line=dict(color=cls.PALETTE[1], width=3),
            marker=dict(size=8, symbol='diamond'),
            yaxis='y2'
        ))

        fig.update_layout(
            yaxis=dict(
                title='平均等待时长 (小时)',
                showgrid=True,
                gridcolor='#f0f0f0',
            ),
            yaxis2=dict(
                title='泊位平均利用率 (%)',
                overlaying='y',
                side='right',
                showgrid=False,
                tickformat=',.1f'
            )
        )

        return cls._update_layout(fig,
            title='敏感性分析：到港间隔 vs 等待时长 & 泊位利用率',
            xlabel='船舶到港间隔均值 (小时)', ylabel='',
            height=450,
            showlegend=True, legend_orientation='h')

    @classmethod
    def create_multi_strategy_timeline(cls, results: dict) -> go.Figure:
        """
        创建三策略对比时间轴折线图
        results: {strategy_name: SimulationResult}
        """
        if not results:
            return cls.create_simulation_timeline_chart(None)

        fig = go.Figure()

        strategy_colors = {
            'FCFS': cls.PALETTE[0],
            'SJF': cls.PALETTE[1],
            'LWF': cls.PALETTE[3],
        }

        strategy_names = {
            'FCFS': '先到先服务 (FCFS)',
            'SJF': '最短作业优先 (SJF)',
            'LWF': '最长作业优先 (LWF)',
        }

        for strategy, result in results.items():
            color = strategy_colors.get(strategy, cls.PALETTE[0])
            name = strategy_names.get(strategy, strategy)
            timeline_df = result.timeline_data

            if timeline_df is None or timeline_df.empty:
                continue

            fig.add_trace(go.Scatter(
                x=timeline_df['时间_小时'], y=timeline_df['锚地等待数'],
                name=f'{name} - 锚地等待', mode='lines',
                line=dict(color=color, width=2, dash='solid'),
                opacity=0.9
            ))
            fig.add_trace(go.Scatter(
                x=timeline_df['时间_小时'], y=timeline_df['在泊作业数'],
                name=f'{name} - 在泊作业', mode='lines',
                line=dict(color=color, width=2.5, dash='dot'),
                opacity=0.9
            ))

        return cls._update_layout(fig,
            title='三策略对比：锚地等待与在泊作业船舶数量变化',
            xlabel='仿真时间 (小时)', ylabel='船舶数量 (艘)',
            height=450,
            showlegend=True, legend_orientation='h')

    @classmethod
    def create_multi_strategy_wait_histogram(cls, results: dict) -> go.Figure:
        """
        创建三策略对比等待时长分组柱状图
        results: {strategy_name: SimulationResult}
        """
        if not results:
            return cls.create_wait_time_histogram(None)

        strategy_names = {
            'FCFS': 'FCFS',
            'SJF': 'SJF',
            'LWF': 'LWF',
        }

        strategy_colors = {
            'FCFS': cls.PALETTE[0],
            'SJF': cls.PALETTE[1],
            'LWF': cls.PALETTE[3],
        }

        all_wait_times = {}
        max_wait = 0
        for strategy, result in results.items():
            wait_times = [s.start_time - s.arrival_time for s in result.ships
                           if hasattr(s, 'start_time') and s.start_time > 0
                           and hasattr(s, 'departure_time') and s.departure_time > 0
                           and not getattr(s, 'rejected', False)]
            all_wait_times[strategy] = wait_times
            if wait_times:
                max_wait = max(max_wait, max(wait_times))

        if not any(all_wait_times.values()):
            fig = go.Figure()
            fig.add_annotation(text='无等待数据', xref='paper', yref='paper',
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            return cls._update_layout(fig, title='等待时长分布对比', height=400)

        num_bins = 20
        bin_width = max_wait / num_bins if max_wait > 0 else 1
        bins = [i * bin_width for i in range(num_bins + 1)]

        fig = go.Figure()

        for strategy, wait_times in all_wait_times.items():
            color = strategy_colors.get(strategy, cls.PALETTE[0])
            name = strategy_names.get(strategy, strategy)

            counts, bin_edges = np.histogram(wait_times, bins=bins)
            bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

            fig.add_trace(go.Bar(
                x=bin_centers,
                y=counts,
                name=name,
                marker_color=color,
                opacity=0.8,
                width=bin_width * 0.28,
            ))

        fig.update_layout(barmode='group')

        return cls._update_layout(fig,
            title='三策略对比：船舶等待时长分布',
            xlabel='等待时长 (小时)', ylabel='船舶数量 (艘)',
            height=400,
            showlegend=True, legend_orientation='h')

    @classmethod
    def create_replay_timeline(cls, timeline_df: pd.DataFrame, current_time: float) -> go.Figure:
        """
        创建带回放时间标记的时间轴图表
        """
        if timeline_df is None or timeline_df.empty:
            return cls.create_simulation_timeline_chart(None)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=timeline_df['时间_小时'], y=timeline_df['锚地等待数'],
            name='锚地等待数', mode='lines',
            line=dict(color=cls.COLORS['warning'], width=2.5),
            fill='tozeroy', fillcolor='rgba(214, 158, 46, 0.15)',
        ))
        fig.add_trace(go.Scatter(
            x=timeline_df['时间_小时'], y=timeline_df['在泊作业数'],
            name='在泊作业数', mode='lines',
            line=dict(color=cls.COLORS['primary'], width=2.5),
            fill='tozeroy', fillcolor='rgba(26, 54, 93, 0.15)',
        ))

        fig.add_vline(
            x=current_time,
            line=dict(color=cls.COLORS['danger'], width=3),
            annotation_text=f'当前: {current_time:.1f}h',
            annotation_position='top right',
            annotation_font=dict(color=cls.COLORS['danger'], size=12),
        )

        return cls._update_layout(fig,
            title='锚地等待与在泊作业船舶数量变化（回放中）',
            xlabel='仿真时间 (小时)', ylabel='船舶数量 (艘)',
            height=400)

    @classmethod
    def create_replay_berth_gantt(cls, berth_df: pd.DataFrame, sim_duration: float,
                                   current_time: float) -> go.Figure:
        """
        创建带回放效果的泊位甘特图
        未发生的占用色块显示为半透明
        """
        if berth_df is None or berth_df.empty:
            return cls.create_simulation_berth_gantt(None, sim_duration)

        df = berth_df.copy()
        berths = sorted(df['泊位编号'].unique())

        def get_color(teu, is_past):
            opacity = 1.0 if is_past else 0.25
            if teu < 2000:
                base = '#90cdf4'
            elif teu <= 5000:
                base = '#3182ce'
            else:
                base = '#1a365d'
            return base, opacity

        fig = go.Figure()

        for berth in berths:
            berth_vessels = df[df['泊位编号'] == berth]
            for _, row in berth_vessels.iterrows():
                duration = row['结束时间'] - row['开始时间']
                is_past = row['结束时间'] <= current_time
                is_current = row['开始时间'] <= current_time < row['结束时间']

                color, opacity = get_color(row['载箱量TEU'], is_past)

                if is_current:
                    actual_duration = current_time - row['开始时间']
                    remaining_duration = row['结束时间'] - current_time

                    hover_text = (
                        f"<b>{row['船名']}</b><br>"
                        f"泊位: {berth}<br>"
                        f"开始: {row['开始时间']:.1f} 小时<br>"
                        f"结束: {row['结束时间']:.1f} 小时<br>"
                        f"载箱量: {row['载箱量TEU']:,} TEU<br>"
                        f"作业时长: {duration:.1f} 小时<br>"
                        f"<span style='color:green'>进度: {(actual_duration/duration)*100:.1f}%</span>"
                    )

                    fig.add_trace(go.Bar(
                        x=[actual_duration],
                        y=[berth],
                        base=[row['开始时间']],
                        orientation='h',
                        marker=dict(
                            color=color,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=hover_text,
                        name=berth,
                        showlegend=False,
                        width=0.6
                    ))
                    fig.add_trace(go.Bar(
                        x=[remaining_duration],
                        y=[berth],
                        base=[current_time],
                        orientation='h',
                        marker=dict(
                            color=color,
                            opacity=0.3,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=hover_text,
                        name=berth,
                        showlegend=False,
                        width=0.6
                    ))
                else:
                    hover_text = (
                        f"<b>{row['船名']}</b><br>"
                        f"泊位: {berth}<br>"
                        f"开始: {row['开始时间']:.1f} 小时<br>"
                        f"结束: {row['结束时间']:.1f} 小时<br>"
                        f"载箱量: {row['载箱量TEU']:,} TEU<br>"
                        f"作业时长: {duration:.1f} 小时"
                    )
                    if not is_past:
                        hover_text += "<br><i>尚未开始</i>"

                    fig.add_trace(go.Bar(
                        x=[duration],
                        y=[berth],
                        base=[row['开始时间']],
                        orientation='h',
                        marker=dict(
                            color=color,
                            opacity=opacity,
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=hover_text,
                        name=berth,
                        showlegend=False,
                        width=0.6
                    ))

        fig.update_layout(
            barmode='overlay',
            xaxis=dict(
                title='仿真时间 (小时)',
                showgrid=True,
                gridcolor='#f0f0f0',
            ),
            yaxis=dict(
                title='泊位',
                showgrid=True,
                gridcolor='#f0f0f0',
                categoryorder='array',
                categoryarray=list(reversed(berths))
            ),
            hovermode='closest'
        )

        fig.add_vline(
            x=current_time,
            line=dict(color=cls.COLORS['danger'], width=3),
            annotation_text=f'当前: {current_time:.1f}h',
            annotation_position='top right',
            annotation_font=dict(color=cls.COLORS['danger'], size=12),
        )

        if sim_duration:
            fig.update_xaxes(range=[0, sim_duration])

        return cls._update_layout(fig,
            title='仿真泊位甘特图（回放中 - 半透明为未发生）',
            xlabel='时间 (小时)', ylabel='泊位编号',
            height=400, showlegend=False)
