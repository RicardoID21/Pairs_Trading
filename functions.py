import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


def download_data(ticker: str, period: str = '10y'):
    """
    Descarga el precio ajustado de un ticker para el periodo especificado.
    Si no se encuentra la columna 'Adj Close', utiliza 'Close'.
    """
    data = yf.download(ticker, period=period)
    print(f"Columnas para {ticker}: {data.columns}")
    if 'Adj Close' in data.columns:
        return data['Adj Close']
    elif 'Adj. Close' in data.columns:
        return data['Adj. Close']
    else:
        print(f"La columna 'Adj Close' no se encontró para {ticker}. Se usará 'Close'.")
        return data['Close']


def log_transform(series):
    """Aplica la transformación logarítmica a la serie."""
    return np.log(series)


def run_adf_test(series, name: str):
    """Aplica el test ADF a la serie y muestra los resultados."""
    series = series.dropna()
    if len(series) < 30:
        print(f"La serie {name} tiene muy pocos datos ({len(series)} observaciones) para ejecutar el test ADF.")
        return
    result = adfuller(series)
    print(f"{name} ADF Statistic: {result[0]:.4f}")
    print(f"{name} p-value: {result[1]:.4f}")
    print("Valores críticos:")
    for key, value in result[4].items():
        print(f"  {key}: {value:.4f}")


def correlation_analysis(series1, series2, name1: str, name2: str):
    """Combina dos series y calcula la correlación de Pearson."""
    data = pd.concat([series1, series2], axis=1, join='inner')
    data.columns = [name1, name2]
    corr = data.corr().iloc[0, 1]
    print(f"Correlación entre {name1} y {name2}: {corr:.4f}")
    return corr


def engle_granger_cointegration_test(series1, series2, name1="Serie1", name2="Serie2"):
    """Realiza la prueba de cointegración de Engle-Granger."""
    df = pd.concat([series1, series2], axis=1, join='inner').dropna()
    df.columns = [name1, name2]
    df_reg = sm.add_constant(df)
    model = sm.OLS(df_reg[name1], df_reg[[name2, 'const']]).fit()
    residuals = model.resid
    adf_result = adfuller(residuals.dropna())
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    print("\n--- Test de Cointegración (Engle-Granger) ---")
    print(f"Regresión: {name1} ~ {name2}")
    print(f"Beta (coef)  : {model.params[name2]:.4f}")
    print(f"Alpha (const): {model.params['const']:.4f}")
    print(f"ADF Statistic (residual): {adf_stat:.4f}")
    print(f"p-value (residual)      : {p_value:.4f}")
    return p_value, adf_stat, model


def plot_etfs(series1, series2, label1: str, label2: str, title="Comparación de activos (Precios Logarítmicos)"):
    """Grafica dos series para comparar sus movimientos."""
    data = pd.concat([series1, series2], axis=1, join='inner')
    data.columns = [label1, label2]
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[label1], label=label1)
    plt.plot(data.index, data[label2], label=label2)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Precio Logarítmico")
    plt.legend()
    plt.grid(True)
    plt.show()


def johansen_cointegration_test(df, det_order=0, k_ar_diff=1):
    """Realiza el test de cointegración de Johansen sobre df."""
    result = coint_johansen(df, det_order, k_ar_diff)
    print("\n--- Test de Johansen ---")
    print("Eigenvalues:")
    print(result.eig)
    print("\nEigenvectors:")
    print(result.evec)
    print("\nEstadísticos de Rastreo (Trace):")
    print(result.lr1)
    print("\nValores críticos:")
    print(result.cvt)
    return result


def ols_regression_and_plot(series_dep, series_indep, dep_label="Y", indep_label="X"):
    """Realiza una regresión OLS y grafica la dispersión junto con la línea de regresión."""
    combined = pd.concat([series_dep, series_indep], axis=1, join='inner')
    combined.columns = [dep_label, indep_label]
    X = combined[indep_label]
    y = combined[dep_label]
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print(model.summary())


class KalmanFilterReg:
    def __init__(self):
        self.x = np.array([1.0, 1.0])  # Estado: [alpha, beta]
        self.A = np.eye(2)
        self.Q = np.eye(2) * 0.1
        self.R = np.array([[1]]) * 10
        self.P = np.eye(2) * 10

    def predict(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x_val, y_val):
        C = np.array([[1, x_val]])
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)
        y_pred = C @ self.x
        resid = y_val - y_pred
        self.x = self.x + K.ravel() * resid
        self.P = (np.eye(2) - K @ C) @ self.P


def run_kalman_filter(log_x, log_y):
    df = pd.concat([log_x, log_y], axis=1).dropna()
    df.columns = ['x', 'y']
    kf = KalmanFilterReg()
    alphas = []
    betas = []
    preds = []
    for date, row in df.iterrows():
        x_ = row['x']
        y_ = row['y']
        kf.predict()
        kf.update(x_, y_)
        alpha, beta = kf.x
        alphas.append(alpha)
        betas.append(beta)
        preds.append(alpha + beta * x_)
    out = pd.DataFrame({'alpha': alphas, 'beta': betas, 'pred_y': preds}, index=df.index)
    return out


def generate_vecm_signals(log_data_shel, log_data_vlo, det_order=0, k_ar_diff=1, threshold_sigma=1.5):
    """
    Ajusta un VECM a las series logarítmicas y genera señales de trading basadas en ±threshold_sigma * std(ECT).
    Retorna:
      - df_signals: DataFrame con columnas ['ECT', 'signal'] indexado por fecha.
      - vecm_res: Objeto del modelo VECM ajustado.
    """
    df = pd.concat([log_data_shel, log_data_vlo], axis=1).dropna()
    df.columns = ['SHEL_Log', 'VLO_Log']
    joh_res = coint_johansen(df, det_order, k_ar_diff)
    rank = 1
    model = VECM(df, deterministic='co', k_ar_diff=k_ar_diff, coint_rank=rank)
    vecm_res = model.fit()
    beta = vecm_res.beta[:, 0]
    has_const = (beta.shape[0] == 3)
    if has_const:
        coint_const = beta[-1]
        beta_assets = beta[:-1]
    else:
        coint_const = 0.0
        beta_assets = beta
    df_shift = df.shift(1).dropna()
    ect_values = []
    for i in range(len(df_shift)):
        row = df_shift.iloc[i]
        val = beta_assets[0] * row['SHEL_Log'] + beta_assets[1] * row['VLO_Log'] + coint_const
        ect_values.append(val)
    ect_series = pd.Series(ect_values, index=df_shift.index, name='ECT')
    ect_mean = ect_series.mean()
    ect_std = ect_series.std()
    up = ect_mean + threshold_sigma * ect_std
    down = ect_mean - threshold_sigma * ect_std
    signals = []
    for val in ect_series:
        if val > up:
            signals.append(-1)
        elif val < down:
            signals.append(1)
        else:
            signals.append(0)
    df_signals = pd.DataFrame({'ECT': ect_series, 'signal': signals}, index=ect_series.index)
    return df_signals, vecm_res


def backtest_vecm_strategy(
        data,
        df_signals,
        capital_init=1_000_000,
        commission=0.125 / 100,
        margin_req=0.5,
        profit_threshold=0.15,
        stop_loss=0.1
):
    """
    Backtest de la estrategia de pairs trading para SHEL y VLO usando señales de VECM,
    operando "all in" y considerando cuenta de margen. Se cierra la posición cuando:
      - Capital + valor de la posición < 0 (margin call),
      - Ganancia >= profit_threshold * trade_cost (take-profit),
      - Pérdida <= -stop_loss * trade_cost (stop-loss),
      - O la señal cambia (o es 0).

    Retorna:
      - portfolio_series: Serie con la evolución del valor total del portafolio.
      - trades: Lista de diccionarios con el log de cada operación.
    """
    capital = capital_init
    active_trade = None
    portfolio_value = []
    trades = []

    for date, row in data.iterrows():
        signal = df_signals.loc[date, 'signal'] if date in df_signals.index else 0

        if active_trade is None:
            if signal in [-1, 1]:
                entry_SHEL = row['SHEL']
                entry_VLO = row['VLO']
                if signal == -1:
                    cost_long_per_share = entry_VLO * (1 + commission)
                    cost_short_per_share = entry_SHEL * margin_req
                else:
                    cost_long_per_share = entry_SHEL * (1 + commission)
                    cost_short_per_share = entry_VLO * margin_req
                total_cost_per_share = cost_long_per_share + cost_short_per_share
                n_shares = int(np.floor(capital / total_cost_per_share))
                if n_shares > 0:
                    trade_cost = n_shares * total_cost_per_share
                    capital -= trade_cost
                    active_trade = {
                        'open_date': date,
                        'signal': signal,
                        'entry_SHEL': entry_SHEL,
                        'entry_VLO': entry_VLO,
                        'n_shares': n_shares,
                        'trade_cost': trade_cost
                    }
                    trades.append({
                        'open_date': date,
                        'close_date': None,
                        'signal': signal,
                        'entry_SHEL': entry_SHEL,
                        'entry_VLO': entry_VLO,
                        'exit_SHEL': None,
                        'exit_VLO': None,
                        'n_shares': n_shares,
                        'pnl': None,
                        'close_reason': None
                    })
        else:
            n_shares = active_trade['n_shares']
            if active_trade['signal'] == -1:
                current_value_SHEL = n_shares * (active_trade['entry_SHEL'] - row['SHEL'])
                current_value_VLO = n_shares * (row['VLO'] - active_trade['entry_VLO'])
            else:
                current_value_SHEL = n_shares * (row['SHEL'] - active_trade['entry_SHEL'])
                current_value_VLO = n_shares * (active_trade['entry_VLO'] - row['VLO'])
            trade_value = current_value_SHEL + current_value_VLO

            if capital + trade_value < 0:
                close_reason = 'margin_call'
            elif trade_value >= profit_threshold * active_trade['trade_cost']:
                close_reason = 'take_profit'
            elif trade_value <= -stop_loss * active_trade['trade_cost']:
                close_reason = 'stop_loss'
            elif signal == 0 or signal != active_trade['signal']:
                close_reason = 'signal_change'
            else:
                close_reason = None

            if close_reason is not None:
                if active_trade['signal'] == -1:
                    pnl_SHEL = (active_trade['entry_SHEL'] - row['SHEL']) * n_shares
                    pnl_VLO = (row['VLO'] - active_trade['entry_VLO']) * n_shares
                else:
                    pnl_SHEL = (row['SHEL'] - active_trade['entry_SHEL']) * n_shares
                    pnl_VLO = (active_trade['entry_VLO'] - row['VLO']) * n_shares
                commission_exit = (n_shares * row['SHEL'] + n_shares * row['VLO']) * commission
                trade_pnl = pnl_SHEL + pnl_VLO - commission_exit
                capital += active_trade['trade_cost'] + trade_pnl
                last_trade = trades[-1]
                if last_trade['close_date'] is None:
                    last_trade['close_date'] = date
                    last_trade['exit_SHEL'] = row['SHEL']
                    last_trade['exit_VLO'] = row['VLO']
                    last_trade['pnl'] = trade_pnl
                    last_trade['close_reason'] = close_reason
                active_trade = None

        if active_trade is not None:
            n_shares = active_trade['n_shares']
            if active_trade['signal'] == -1:
                current_value_SHEL = n_shares * (active_trade['entry_SHEL'] - row['SHEL'])
                current_value_VLO = n_shares * (row['VLO'] - active_trade['entry_VLO'])
            else:
                current_value_SHEL = n_shares * (row['SHEL'] - active_trade['entry_SHEL'])
                current_value_VLO = n_shares * (active_trade['entry_VLO'] - row['VLO'])
            trade_value = current_value_SHEL + current_value_VLO
            total_value = capital + trade_value
        else:
            total_value = capital

        portfolio_value.append(total_value)

    portfolio_series = pd.Series(portfolio_value, index=data.index)

    # Calcular métricas de rendimiento
    final_capital = portfolio_series.iloc[-1]
    total_return = (final_capital - capital_init) / capital_init
    portfolio_returns = portfolio_series.pct_change().fillna(0)
    days = len(portfolio_series)
    annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
    vol_annualized = portfolio_returns.std() * np.sqrt(252)
    risk_free_rate = 0.02
    excess_return = portfolio_returns - (risk_free_rate / 252)
    sharpe_ratio = (excess_return.mean() / excess_return.std()) * np.sqrt(252) if excess_return.std() != 0 else 0
    rolling_max = portfolio_series.cummax()
    drawdown = (portfolio_series - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    print("\n=== Strategy Performance Metrics ===")
    print(f"Initial Capital      : ${capital_init:,.2f}")
    print(f"Final Capital        : ${final_capital:,.2f}")
    print(f"Total Return         : {total_return:.2%}")
    print(f"Annualized Return    : {annualized_return:.2%}")

    return portfolio_series, trades
