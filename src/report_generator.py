"""
PDF报告导出模块
生成运营分析报告，包含吞吐量统计、预测评估、堆场分析、翻箱率、超期箱、KPI、What-if结论
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, Image, HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class PortReportGenerator:
    """港口运营分析PDF报告生成器"""

    def __init__(self, output_dir: str = 'reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _init_styles(self):
        """初始化报告样式"""
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1a365d')
        )

        h1_style = ParagraphStyle(
            'H1',
            parent=styles['Heading1'],
            fontSize=18,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2c5282')
        )

        h2_style = ParagraphStyle(
            'H2',
            parent=styles['Heading2'],
            fontSize=14,
            spaceBefore=12,
            spaceAfter=8,
            textColor=colors.HexColor('#2b6cb0')
        )

        h3_style = ParagraphStyle(
            'H3',
            parent=styles['Heading3'],
            fontSize=12,
            spaceBefore=8,
            spaceAfter=6,
            textColor=colors.HexColor('#3182ce')
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            leading=16,
            spaceAfter=6
        )

        metric_style = ParagraphStyle(
            'Metric',
            parent=styles['Normal'],
            fontSize=11,
            leading=18,
            textColor=colors.HexColor('#2d3748')
        )

        highlight_style = ParagraphStyle(
            'Highlight',
            parent=styles['Normal'],
            fontSize=11,
            leading=18,
            backColor=colors.HexColor('#fefcbf'),
            borderPadding=4
        )

        alert_style = ParagraphStyle(
            'Alert',
            parent=styles['Normal'],
            fontSize=10,
            leading=16,
            textColor=colors.HexColor('#c53030'),
            backColor=colors.HexColor('#fed7d7'),
            borderPadding=4
        )

        return {
            'title': title_style,
            'h1': h1_style,
            'h2': h2_style,
            'h3': h3_style,
            'normal': normal_style,
            'metric': metric_style,
            'highlight': highlight_style,
            'alert': alert_style
        }

    def _format_number(self, val, decimals=2):
        """格式化数字"""
        if pd.isna(val):
            return '-'
        if isinstance(val, float):
            return f'{val:,.{decimals}f}'
        return f'{val:,}'

    def _format_percent(self, val, decimals=2):
        """格式化百分比"""
        if pd.isna(val):
            return '-'
        return f'{val * 100:,.{decimals}f}%'

    def _create_table(self, data, header=True, col_widths=None):
        """创建带样式的表格"""
        tbl = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)

        style_commands = [
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]

        if header:
            style_commands += [
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c5282')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
            ]
            for i in range(1, len(data)):
                if i % 2 == 0:
                    style_commands.append(
                        ('BACKGROUND', (0, i), (-1, i), colors.HexColor('#f7fafc'))
                    )

        tbl.setStyle(TableStyle(style_commands))
        return tbl

    def _save_figure_as_image(self, fig, filename, width=700, height=400):
        """将Plotly图表保存为PNG图片"""
        if not PLOTLY_AVAILABLE:
            return None
        filepath = os.path.join(self.output_dir, filename)
        try:
            pio.write_image(fig, filepath, format='png', width=width, height=height, scale=2)
            return filepath
        except Exception as e:
            print(f"图表保存失败: {e}")
            return None

    def _add_chart_image(self, story, img_path, width_cm=17, height_cm=9):
        """向报告中添加图表图片"""
        if img_path and os.path.exists(img_path):
            story.append(Image(img_path, width=width_cm * cm, height=height_cm * cm))
            story.append(Spacer(1, 8))

    def generate_report(
        self,
        report_name: str,
        start_date: datetime,
        end_date: datetime,
        daily_throughput: pd.DataFrame,
        forecast_results: Dict,
        yard_util: pd.DataFrame,
        rehandling_results: pd.DataFrame,
        overdue_by_customer: pd.DataFrame,
        overdue_by_area: pd.DataFrame,
        kpi_data: Dict,
        whatif_results: pd.DataFrame,
        figures: Optional[Dict] = None
    ) -> str:
        """
        生成完整的运营分析报告
        返回: PDF文件路径
        """
        if not PDF_AVAILABLE:
            raise ImportError("reportlab未安装，无法生成PDF报告")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(self.output_dir, f'{report_name}_{timestamp}.pdf')

        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
            title=f'{report_name} - 港口运营分析报告',
            author='港口智能运营系统'
        )

        styles = self._init_styles()
        story = []

        # ===== 封面 =====
        story.append(Spacer(1, 3 * cm))
        story.append(Paragraph('港口集装箱吞吐量预测与', styles['title']))
        story.append(Paragraph('堆场利用率优化分析报告', styles['title']))
        story.append(Spacer(1, 1.5 * cm))

        meta_data = [
            ['报告期间', f'{start_date.strftime("%Y-%m-%d")} 至 {end_date.strftime("%Y-%m-%d")}'],
            ['生成时间', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['报告编号', f'PRT-{timestamp}'],
            ['版本', 'V1.0']
        ]
        meta_tbl = Table(meta_data, colWidths=[4 * cm, 10 * cm])
        meta_tbl.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#4a5568')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_tbl)
        story.append(PageBreak())

        # ===== 目录 =====
        story.append(Paragraph('目录', styles['h1']))
        toc_items = [
            '一、吞吐量统计概览',
            '二、吞吐量预测与准确度评估',
            '三、堆场利用率峰谷时段分析',
            '四、翻箱率策略对比分析',
            '五、超期箱TOP10客户清单',
            '六、KPI运营指标仪表板',
            '七、What-if情景分析结论摘要'
        ]
        for i, item in enumerate(toc_items, 1):
            story.append(Paragraph(item, styles['normal']))
        story.append(PageBreak())

        # ===== 第一章：吞吐量统计 =====
        story.append(Paragraph('一、吞吐量统计概览', styles['h1']))

        total_in = daily_throughput['进场箱量'].sum()
        total_out = daily_throughput['出场箱量'].sum()
        total_teu = daily_throughput['总TEU'].sum()
        days_count = len(daily_throughput)
        avg_daily_boxes = (total_in + total_out) / max(1, days_count)
        avg_daily_teu = total_teu / max(1, days_count)

        summary_data = [
            ['指标', '数值', '指标', '数值'],
            ['总进场箱量', self._format_number(total_in, 0),
             '总出场箱量', self._format_number(total_out, 0)],
            ['总吞吐量TEU', self._format_number(total_teu, 0),
             '日均箱量', self._format_number(avg_daily_boxes, 1)],
            ['日均TEU', self._format_number(avg_daily_teu, 1),
             '统计天数', f'{days_count} 天']
        ]
        story.append(self._create_table(summary_data))
        story.append(Spacer(1, 8))

        if figures and 'throughput_chart' in figures:
            img = self._save_figure_as_image(figures['throughput_chart'], 'throughput_summary.png')
            self._add_chart_image(story, img)
        story.append(PageBreak())

        # ===== 第二章：吞吐量预测 =====
        story.append(Paragraph('二、吞吐量预测与准确度评估', styles['h1']))

        story.append(Paragraph('2.1 预测算法对比', styles['h2']))
        algo_comparison = [['算法', 'MAPE (%)', 'RMSE (箱量)']]
        for algo, result in forecast_results.items():
            if isinstance(result, dict):
                algo_comparison.append([
                    algo,
                    self._format_number(result.get('mape', np.nan), 2),
                    self._format_number(result.get('rmse', np.nan), 0)
                ])
        story.append(self._create_table(algo_comparison))
        story.append(Spacer(1, 8))

        story.append(Paragraph('2.2 准确度分析结论', styles['h2']))
        if forecast_results:
            best_algo = min(
                forecast_results.items(),
                key=lambda x: x[1].get('mape', float('inf')) if isinstance(x[1], dict) else float('inf')
            )
            story.append(Paragraph(
                f"推荐算法：<b>{best_algo[0]}</b>，该算法在分析期间的预测准确度最高，"
                f"MAPE为 {self._format_number(best_algo[1].get('mape', np.nan))}%，"
                f"RMSE为 {self._format_number(best_algo[1].get('rmse', np.nan), 0)} 箱。",
                styles['highlight']
            ))
        story.append(Spacer(1, 8))

        if figures and 'forecast_chart' in figures:
            img = self._save_figure_as_image(figures['forecast_chart'], 'forecast_chart.png')
            self._add_chart_image(story, img)
        story.append(PageBreak())

        # ===== 第三章：堆场利用率 =====
        story.append(Paragraph('三、堆场利用率峰谷时段分析', styles['h1']))

        overall_util = yard_util.groupby('日期').agg(
            总占用=('占用箱数', 'sum'),
            总容量=('总容量', 'sum')
        ).reset_index()
        overall_util['利用率'] = overall_util['总占用'] / overall_util['总容量']

        peak_date = overall_util.loc[overall_util['利用率'].idxmax()]
        valley_date = overall_util.loc[overall_util['利用率'].idxmin()]
        avg_util = overall_util['利用率'].mean()

        story.append(Paragraph('3.1 利用率关键指标', styles['h2']))
        util_data = [
            ['指标', '数值', '日期'],
            ['峰值利用率', self._format_percent(peak_date['利用率']),
             str(peak_date['日期'])],
            ['谷值利用率', self._format_percent(valley_date['利用率']),
             str(valley_date['日期'])],
            ['平均利用率', self._format_percent(avg_util), '-']
        ]
        story.append(self._create_table(util_data))
        story.append(Spacer(1, 8))

        story.append(Paragraph('3.2 各区域利用率对比', styles['h2']))
        area_util = yard_util.groupby('区域').agg(
            平均占用=('占用箱数', 'mean'),
            总容量=('总容量', 'first'),
            平均空间占用率=('空间占用率', 'mean'),
            峰值占用率=('空间占用率', 'max')
        ).reset_index()
        area_table = [['区域', '平均占用率', '峰值占用率', '总容量']]
        for _, row in area_util.iterrows():
            area_table.append([
                row['区域'],
                self._format_percent(row['平均空间占用率']),
                self._format_percent(row['峰值占用率']),
                self._format_number(int(row['总容量']), 0)
            ])
        story.append(self._create_table(area_table))
        story.append(Spacer(1, 8))

        if peak_date['利用率'] > 0.85:
            story.append(Paragraph(
                f"⚠️ 告警：{str(peak_date['日期'])} 堆场峰值利用率达到"
                f" {self._format_percent(peak_date['利用率'])}，接近容量饱和，建议提前安排堆场资源调度。",
                styles['alert']
            ))

        if figures and 'yard_util_chart' in figures:
            img = self._save_figure_as_image(figures['yard_util_chart'], 'yard_utilization.png')
            self._add_chart_image(story, img)
        story.append(PageBreak())

        # ===== 第四章：翻箱率分析 =====
        story.append(Paragraph('四、翻箱率策略对比分析', styles['h1']))

        story.append(Paragraph('4.1 三种堆存策略模拟结果（蒙特卡洛1000次）', styles['h2']))

        if not rehandling_results.empty:
            strategy_summary = rehandling_results.groupby('策略').agg(
                平均翻箱率=('平均翻箱率', 'mean'),
                P95翻箱率=('P95翻箱率', 'mean')
            ).reset_index()
            rh_table = [['堆存策略', '平均翻箱率', 'P95翻箱率']]
            for _, row in strategy_summary.iterrows():
                rh_table.append([
                    row['策略'],
                    self._format_percent(row['平均翻箱率']),
                    self._format_percent(row['P95翻箱率'])
                ])
            story.append(self._create_table(rh_table))
            story.append(Spacer(1, 8))

            best_strategy = strategy_summary.loc[strategy_summary['平均翻箱率'].idxmin()]
            story.append(Paragraph(
                f"推荐策略：<b>{best_strategy['策略']}</b>，"
                f"平均翻箱率仅为 {self._format_percent(best_strategy['平均翻箱率'])}，"
                f"显著低于其他策略，可有效减少翻箱作业成本。",
                styles['highlight']
            ))
        story.append(Spacer(1, 8))

        if figures and 'rehandling_chart' in figures:
            img = self._save_figure_as_image(figures['rehandling_chart'], 'rehandling_comparison.png')
            self._add_chart_image(story, img)
        story.append(PageBreak())

        # ===== 第五章：超期箱 =====
        story.append(Paragraph('五、超期箱TOP10客户清单', styles['h1']))

        free_days = 5
        total_overdue_boxes = int((overdue_by_customer['超期箱数量'].sum()) if not overdue_by_customer.empty else 0)
        total_overdue_fee = (overdue_by_customer['预估滞箱费用'].sum()) if not overdue_by_customer.empty else 0

        story.append(Paragraph(
            f'免费堆存期：{free_days}天 | 超期箱总数：{self._format_number(total_overdue_boxes, 0)} | '
            f'预估滞箱费用合计：¥{self._format_number(total_overdue_fee, 0)}',
            styles['metric']
        ))
        story.append(Spacer(1, 8))

        story.append(Paragraph('5.1 TOP10客户超期箱统计', styles['h2']))
        if not overdue_by_customer.empty:
            top10 = overdue_by_customer.head(10)
            cust_table = [['排名', '客户名称', '超期箱数', '平均超期天数', '预估滞箱费(元)']]
            for idx, (_, row) in enumerate(top10.iterrows(), 1):
                cust_table.append([
                    str(idx),
                    row['客户'],
                    self._format_number(int(row['超期箱数量']), 0),
                    self._format_number(row['平均超期天数'], 1),
                    self._format_number(row['预估滞箱费用'], 0)
                ])
            story.append(self._create_table(cust_table))
        story.append(Spacer(1, 8))

        story.append(Paragraph('5.2 各区域超期箱分布', styles['h2']))
        if not overdue_by_area.empty:
            area_table2 = [['堆场区域', '超期箱数量', '占比', '预估滞箱费(元)']]
            for _, row in overdue_by_area.iterrows():
                pct = row['超期箱数量'] / max(1, total_overdue_boxes) * 100
                area_table2.append([
                    row['堆场区域'],
                    self._format_number(int(row['超期箱数量']), 0),
                    f'{pct:.1f}%',
                    self._format_number(row['预估滞箱费用'], 0)
                ])
            story.append(self._create_table(area_table2))
        story.append(PageBreak())

        # ===== 第六章：KPI指标 =====
        story.append(Paragraph('六、KPI运营指标仪表板', styles['h1']))

        if '综合KPI' in kpi_data:
            kpi = kpi_data['综合KPI']
            if len(kpi) > 0:
                kpi_summary = [
                    ['KPI指标', '平均值', '标准差', '告警次数'],
                ]

                for col, name in [
                    ('船时效率_TEU_小时', '船时效率 (TEU/小时)'),
                    ('桥吊效率_TEU_台时', '桥吊效率 (TEU/台时)'),
                    ('泊位利用率', '泊位利用率'),
                    ('平均集卡周转_小时', '集卡周转时间 (小时)')
                ]:
                    if col in kpi.columns:
                        vals = kpi[col].dropna()
                        if len(vals) > 0:
                            mean_v = vals.mean()
                            std_v = vals.std()
                            threshold = mean_v - std_v
                            if col == '平均集卡周转_小时':
                                threshold = mean_v + std_v
                                alert_count = int((vals > threshold).sum())
                            else:
                                alert_count = int((vals < threshold).sum())

                            display_mean = self._format_percent(mean_v) if '利用率' in name else self._format_number(mean_v, 2)
                            display_std = self._format_percent(std_v) if '利用率' in name else self._format_number(std_v, 2)

                            kpi_summary.append([
                                name, display_mean, display_std, f'{alert_count} 次'
                            ])

                story.append(self._create_table(kpi_summary))

                alert_texts = []
                for col, name in [
                    ('船时效率_TEU_小时', '船时效率'),
                    ('桥吊效率_TEU_台时', '桥吊效率'),
                    ('泊位利用率', '泊位利用率')
                ]:
                    if col in kpi.columns:
                        vals = kpi[col].dropna()
                        if len(vals) > 3:
                            mean_v = vals.mean()
                            std_v = vals.std()
                            below = vals.values
                            alert_days = 0
                            for i in range(2, len(below)):
                                if (below[i] < mean_v - std_v and
                                    below[i-1] < mean_v - std_v and
                                    below[i-2] < mean_v - std_v):
                                    alert_days += 1
                            if alert_days > 0:
                                alert_texts.append(
                                    f'⚠️ {name}：有 {alert_days} 次连续3天低于历史均值-1标准差的告警'
                                )

                if alert_texts:
                    story.append(Paragraph('6.1 异常告警', styles['h2']))
                    for t in alert_texts:
                        story.append(Paragraph(t, styles['alert']))

        if figures and 'kpi_chart' in figures:
            img = self._save_figure_as_image(figures['kpi_chart'], 'kpi_dashboard.png', height=500)
            self._add_chart_image(story, img, height_cm=11)
        story.append(PageBreak())

        # ===== 第七章：What-if分析 =====
        story.append(Paragraph('七、What-if情景分析结论摘要', styles['h1']))

        if not whatif_results.empty:
            for _, row in whatif_results.iterrows():
                scenario_name = row.get('情景名称', '未知情景')
                story.append(Paragraph(f'情景：{scenario_name}', styles['h3']))

                insights = []
                if '新峰值利用率' in row:
                    insights.append(
                        f"峰值利用率从 {self._format_percent(row.get('原始峰值利用率', np.nan))}"
                        f" 变化至 {self._format_percent(row.get('新峰值利用率', np.nan))}"
                    )
                if '峰值利用率下降幅度' in row and not pd.isna(row.get('峰值利用率下降幅度')):
                    insights.append(
                        f"峰值利用率下降 {self._format_percent(row.get('峰值利用率下降幅度', 0))}"
                    )
                if '超期箱减少数量' in row and not pd.isna(row.get('超期箱减少数量')):
                    insights.append(
                        f"超期箱数量减少 {self._format_number(int(row.get('超期箱减少数量', 0)), 0)} 个"
                        f" ({self._format_percent(row.get('超期箱减少比例', 0))})"
                    )
                if '滞箱费用减少' in row and not pd.isna(row.get('滞箱费用减少')):
                    insights.append(
                        f"滞箱费用减少 ¥{self._format_number(row.get('滞箱费用减少', 0), 0)}"
                    )
                if '翻箱率改善幅度' in row and not pd.isna(row.get('翻箱率改善幅度')):
                    insights.append(
                        f"翻箱率改善 {self._format_percent(row.get('翻箱率改善幅度', 0))}"
                    )

                for ins in insights:
                    story.append(Paragraph(f'• {ins}', styles['normal']))
                story.append(Spacer(1, 4))

        story.append(Spacer(1, 1 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
        story.append(Spacer(1, 6))
        story.append(Paragraph(
            '— 报告结束 —',
            ParagraphStyle('center', parent=styles['normal'], alignment=TA_CENTER, textColor=colors.grey)
        ))

        doc.build(story)
        return filepath
