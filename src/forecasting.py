"""
吞吐量预测模块 - ARIMA、Prophet、LSTM三种时序预测算法
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler


def calculate_metrics(actual, predicted):
    """计算预测评估指标 MAPE 和 RMSE"""
    actual = np.array(actual, dtype=float)
    predicted = np.array(predicted, dtype=float)

    mask = actual != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = np.nan

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return mape, rmse


class ARIMAPredictor:
    """ARIMA时序预测器 - 自动定阶(AIC准则)，季节性周期7天"""

    def __init__(self, seasonal_period=7):
        self.seasonal_period = seasonal_period
        self.model = None
        self.scaler = None
        self.history = None

    def _auto_select_order(self, series):
        """自动选择ARIMA阶数（简化版网格搜索）"""
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, self.seasonal_period)

        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]

        try:
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            mod = SARIMAX(
                                series,
                                order=(p, d, q),
                                seasonal_order=(1, d, 1, self.seasonal_period),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            results = mod.fit(disp=False, maxiter=100)
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                best_seasonal_order = (1, d, 1, self.seasonal_period)
                        except Exception:
                            continue
        except Exception:
            pass

        return best_order, best_seasonal_order

    def fit(self, train_dates, train_values):
        """训练模型"""
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels未安装，无法使用ARIMA")

        self.history = pd.Series(train_values, index=pd.DatetimeIndex(train_dates))

        try:
            order, seasonal_order = self._auto_select_order(self.history.values)
            self.model = SARIMAX(
                self.history,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.results = self.model.fit(disp=False, maxiter=200)
        except Exception:
            try:
                self.model = ARIMA(self.history, order=(1, 1, 1))
                self.results = self.model.fit()
            except Exception as e:
                raise RuntimeError(f"ARIMA训练失败: {e}")

        return self

    def predict(self, steps=7, confidence=0.8):
        """预测未来steps步"""
        if self.results is None:
            raise RuntimeError("模型未训练")

        alpha = 1 - confidence
        forecast = self.results.get_forecast(steps=steps)
        pred_mean = forecast.predicted_mean.values
        pred_ci = forecast.conf_int(alpha=alpha)

        future_dates = pd.date_range(
            start=self.history.index[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        return {
            'dates': future_dates,
            'mean': pred_mean,
            'lower': pred_ci.iloc[:, 0].values,
            'upper': pred_ci.iloc[:, 1].values
        }

    def get_fitted_values(self):
        """获取训练集拟合值（用于残差分析）"""
        if self.results is None:
            return None, None
        return self.history.index, self.results.fittedvalues.values


class ProphetPredictor:
    """Prophet预测器 - 节假日效应+周周期+年周期"""

    def __init__(self):
        self.model = None
        self.train_df = None

    def _create_holidays(self):
        """创建中国主要节假日"""
        holidays = pd.DataFrame({
            'holiday': 'chinese_holiday',
            'ds': pd.to_datetime([
                '2026-01-01', '2026-02-16', '2026-02-17', '2026-02-18',
                '2026-02-19', '2026-02-20', '2026-04-04', '2026-04-05',
                '2026-04-06', '2026-05-01', '2026-05-02', '2026-05-03',
                '2026-06-19', '2026-06-20', '2026-06-21', '2026-09-25',
                '2026-09-26', '2026-09-27', '2026-10-01', '2026-10-02',
                '2026-10-03', '2026-10-04', '2026-10-05', '2026-10-06',
                '2026-10-07',
            ]),
            'lower_window': -1,
            'upper_window': 1,
        })
        return holidays

    def fit(self, train_dates, train_values):
        """训练Prophet模型"""
        if not PROPHET_AVAILABLE:
            raise ImportError("prophet库未安装，无法使用Prophet")

        self.train_df = pd.DataFrame({
            'ds': pd.to_datetime(train_dates),
            'y': train_values
        })

        try:
            self.model = Prophet(
                holidays=self._create_holidays(),
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode='additive',
                changepoint_prior_scale=0.05
            )
            self.model.fit(self.train_df)
        except Exception as e:
            raise RuntimeError(f"Prophet训练失败: {e}")

        return self

    def predict(self, steps=7, confidence=0.8):
        """预测未来steps步"""
        if self.model is None:
            raise RuntimeError("模型未训练")

        self.model.interval_width = confidence
        future = self.model.make_future_dataframe(periods=steps, freq='D')
        forecast = self.model.predict(future)

        future_forecast = forecast.tail(steps)
        hist_forecast = forecast.head(len(self.train_df))

        return {
            'dates': future_forecast['ds'].values,
            'mean': future_forecast['yhat'].values,
            'lower': future_forecast['yhat_lower'].values,
            'upper': future_forecast['yhat_upper'].values,
            'hist_dates': hist_forecast['ds'].values,
            'hist_fitted': hist_forecast['yhat'].values
        }


class LSTMPredictor:
    """简化LSTM预测器 - 单层128单元，滑动窗口30天预测未来7天"""

    def __init__(self, window_size=30, hidden_size=128, epochs=50, batch_size=32):
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        self.device = torch.device('cpu')
        self.train_values = None
        self.train_dates = None

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=128, output_size=7):
            super().__init__()
            self.hidden_size = hidden_size
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=1)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, output_size)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_out = lstm_out[:, -1, :]
            return self.fc(last_out)

    def _create_sequences(self, data, target_steps=7):
        """创建滑动窗口序列"""
        X, y = [], []
        for i in range(len(data) - self.window_size - target_steps + 1):
            X.append(data[i:i + self.window_size])
            y.append(data[i + self.window_size:i + self.window_size + target_steps])
        return np.array(X), np.array(y)

    def fit(self, train_dates, train_values):
        """训练LSTM模型"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch未安装，无法使用LSTM")

        self.train_dates = pd.to_datetime(train_dates)
        self.train_values = np.array(train_values, dtype=float).reshape(-1, 1)

        scaled = self.scaler.fit_transform(self.train_values)

        if len(scaled) < self.window_size + 7:
            raise ValueError(f"训练数据不足，需要至少{self.window_size + 7}个数据点")

        X, y = self._create_sequences(scaled, target_steps=7)

        if len(X) == 0:
            raise ValueError("无法创建训练序列")

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).squeeze(-1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self.SimpleLSTM(
            input_size=1, hidden_size=self.hidden_size, output_size=7
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        return self

    def predict(self, steps=7, confidence=0.8):
        """预测未来steps步（使用蒙特卡洛dropout估计不确定性）"""
        if self.model is None:
            raise RuntimeError("模型未训练")

        self.model.eval()
        last_window = self.scaler.transform(self.train_values[-self.window_size:])
        X_tensor = torch.FloatTensor(last_window.reshape(1, self.window_size, 1)).to(self.device)

        with torch.no_grad():
            point_forecast = self.model(X_tensor).cpu().numpy().flatten()

        n_simulations = 100
        self.model.train()
        simulations = []
        with torch.no_grad():
            for _ in range(n_simulations):
                sim = self.model(X_tensor).cpu().numpy().flatten()
                simulations.append(sim)

        simulations = np.array(simulations)
        alpha = 1 - confidence
        lower = np.percentile(simulations, alpha / 2 * 100, axis=0)
        upper = np.percentile(simulations, (1 - alpha / 2) * 100, axis=0)

        def inv_scale(arr):
            return self.scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

        point_forecast = inv_scale(point_forecast)
        lower = inv_scale(lower)
        upper = inv_scale(upper)

        future_dates = pd.date_range(
            start=self.train_dates.iloc[-1] + pd.Timedelta(days=1),
            periods=steps,
            freq='D'
        )

        hist_dates = self.train_dates[self.window_size:]
        hist_fitted = self._get_fitted_values()

        return {
            'dates': future_dates[:steps],
            'mean': point_forecast[:steps],
            'lower': lower[:steps],
            'upper': upper[:steps],
            'hist_dates': hist_dates,
            'hist_fitted': hist_fitted
        }

    def _get_fitted_values(self):
        """获取训练集拟合值"""
        self.model.eval()
        scaled = self.scaler.transform(self.train_values)
        X, y = self._create_sequences(scaled, target_steps=7)
        if len(X) == 0:
            return np.array([])

        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()

        fitted = []
        for i in range(min(len(preds), len(self.train_values) - self.window_size)):
            fitted.append(self.scaler.inverse_transform(
                preds[i, 0].reshape(-1, 1)
            ).flatten()[0])

        return np.array(fitted)


class ThroughputForecaster:
    """吞吐量预测封装类"""

    ALGORITHMS = ['ARIMA', 'Prophet', 'LSTM']

    def __init__(self, algorithm='ARIMA'):
        self.algorithm = algorithm
        self.predictor = None
        self._init_predictor()

    def _init_predictor(self):
        if self.algorithm == 'ARIMA':
            self.predictor = ARIMAPredictor(seasonal_period=7)
        elif self.algorithm == 'Prophet':
            self.predictor = ProphetPredictor()
        elif self.algorithm == 'LSTM':
            self.predictor = LSTMPredictor(window_size=30, hidden_size=128, epochs=30)
        else:
            raise ValueError(f"未知算法: {self.algorithm}")

    def fit(self, series_df, train_days=None):
        """
        训练模型
        series_df: 包含ds和y列的DataFrame
        train_days: 使用最后N天数据训练，None则使用全部
        """
        df = series_df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        df = df.sort_values('ds').reset_index(drop=True)

        if train_days and len(df) > train_days:
            df = df.tail(train_days).reset_index(drop=True)

        dates = df['ds'].values
        values = df['y'].astype(float).values

        self.train_dates = dates
        self.train_values = values

        self.predictor.fit(dates, values)
        return self

    def forecast(self, steps=7, confidence=0.8):
        """执行预测"""
        result = self.predictor.predict(steps=steps, confidence=confidence)

        hist_dates = pd.to_datetime(self.train_dates)
        actual = self.train_values

        if self.algorithm == 'ARIMA':
            _, fitted = self.predictor.get_fitted_values()
            fitted_dates = hist_dates
        else:
            fitted_dates = pd.to_datetime(result.get('hist_dates', hist_dates))
            fitted = result.get('hist_fitted', None)

        if fitted is not None and len(fitted) > 0:
            min_len = min(len(fitted_dates), len(fitted), len(actual))
            mape, rmse = calculate_metrics(actual[-min_len:], fitted[-min_len:])
        else:
            mape, rmse = np.nan, np.nan

        residuals = None
        if fitted is not None and len(fitted) > 0:
            min_len = min(len(fitted), len(actual))
            residuals = actual[-min_len:] - np.array(fitted[-min_len:])

        return {
            'algorithm': self.algorithm,
            'hist_dates': hist_dates,
            'hist_actual': actual,
            'hist_fitted_dates': fitted_dates if fitted is not None else hist_dates,
            'hist_fitted': fitted,
            'forecast_dates': pd.to_datetime(result['dates']),
            'forecast_mean': result['mean'],
            'forecast_lower': result['lower'],
            'forecast_upper': result['upper'],
            'mape': mape,
            'rmse': rmse,
            'residuals': residuals
        }
