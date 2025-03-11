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


def plot_etfs(series1, series2, label1: str, label2: str, title="Comparación de activos (Precios Ajustados)"):
    """Grafica dos series para comparar sus movimientos."""
    data = pd.concat([series1, series2], axis=1, join='inner')
    data.columns = [label1, label2]
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[label1], label=label1)
    plt.plot(data.index, data[label2], label=label2)
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Precio Ajustado")
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


def run_kalman_filter(x, y):
    """Aplica el filtro de Kalman a las series de precios ajustados."""
    df = pd.concat([x, y], axis=1).dropna()
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


def generate_vecm_signals(data_shel, data_vlo, det_order=0, k_ar_diff=1, threshold_sigma=1.5):
    """
    Ajusta un VECM a las series de precios ajustados y genera señales de trading basadas en ±threshold_sigma * std(ECT).
    Retorna:
      - df_signals: DataFrame con columnas ['ECT', 'signal'] indexado por fecha.
      - vecm_res: Objeto del modelo VECM ajustado.
    """
    df = pd.concat([data_shel, data_vlo], axis=1).dropna()
    df.columns = ['SHEL', 'VLO']
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
        val = beta_assets[0] * row['SHEL'] + beta_assets[1] * row['VLO'] + coint_const
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


def backtest_strategy(hedge_df, trades, capital_init=1_000_000, commission=0.00125, margin_req=0.3, n_shares_shel=50):
    """
    Realiza un backtest de la estrategia de pairs trading.

    Parámetros:
    - hedge_df: DataFrame con columnas ['SHEL', 'VLO', 'Hedge_Ratio'] indexado por fecha.
    - trades: Lista de operaciones con 'date', 'type', 'signal', 'spread', 'hedge_ratio'.
    - capital_init: Capital inicial (default: 1,000,000).
    - commission: Comisión por transacción (default: 0.125%).
    - margin_req: Porcentaje de capital reservado como margen (default: 30%).
    - n_shares_shel: Número de acciones de SHEL por operación (default: 50).

    Retorna:
    - trade_log_df: DataFrame con el registro de operaciones.
    - capital_history_df: DataFrame con el historial del capital.
    """
    capital = capital_init  # Capital inicial
    trade_log = []  # Registro de operaciones
    capital_history = []  # Historial del capital
    cumulative_pnl = 0  # P&L acumulado
    open_positions_with_details = []  # Rastrear posiciones abiertas con precios y cantidades

    for trade in trades:
        date = trade['date']
        trade_type = trade['type']
        signal = trade['signal']
        hedge_ratio = trade['hedge_ratio']
        shel_price = hedge_df.loc[date, 'SHEL']
        vlo_price = hedge_df.loc[date, 'VLO']
        capital_available = capital * (1 - margin_req)  # Capital disponible para operar

        # Calcular número de acciones de VLO usando Hedge_Ratio
        n_shares_vlo = int(n_shares_shel * hedge_ratio)

        if trade_type == 'OPEN':
            if signal == -1:  # LONG VLO, SHORT SHEL
                # LONG VLO: Verificar capital para compra
                vlo_cost = n_shares_vlo * vlo_price * (1 + commission)
                shel_cost = n_shares_shel * shel_price * commission  # Solo costo de comisión para SHORT
                if capital_available >= vlo_cost + shel_cost:
                    # Ejecutar operación
                    capital -= vlo_cost  # Restar costo de LONG VLO
                    open_positions_with_details.append({
                        'open_date': date,
                        'signal': -1,
                        'shel_shares': -n_shares_shel,  # SHORT SHEL
                        'vlo_shares': n_shares_vlo,     # LONG VLO
                        'shel_open_price': shel_price,
                        'vlo_open_price': vlo_price,
                        'shel_commission_open': n_shares_shel * shel_price * commission,
                        'vlo_commission_open': n_shares_vlo * vlo_price * commission
                    })
                    trade_log.append({
                        'Date': date,
                        'Type': 'OPEN',
                        'Signal': -1,
                        'SHEL_Shares': -n_shares_shel,
                        'VLO_Shares': n_shares_vlo,
                        'SHEL_Price': shel_price,
                        'VLO_Price': vlo_price,
                        'P&L': 0,
                        'Cumulative_P&L': cumulative_pnl,
                        'Return (%)': (cumulative_pnl / capital_init) * 100,
                        'Capital': capital
                    })
                else:
                    print(f"No hay capital suficiente para abrir posición en {date}: Capital disponible = {capital_available:.2f}, Costo = {vlo_cost + shel_cost:.2f}")
                    continue

            elif signal == 1:  # LONG SHEL, SHORT VLO
                # LONG SHEL: Verificar capital para compra
                shel_cost = n_shares_shel * shel_price * (1 + commission)
                vlo_cost = n_shares_vlo * vlo_price * commission  # Solo costo de comisión para SHORT
                if capital_available >= shel_cost + vlo_cost:
                    # Ejecutar operación
                    capital -= shel_cost  # Restar costo de LONG SHEL
                    open_positions_with_details.append({
                        'open_date': date,
                        'signal': 1,
                        'shel_shares': n_shares_shel,   # LONG SHEL
                        'vlo_shares': -n_shares_vlo,    # SHORT VLO
                        'shel_open_price': shel_price,
                        'vlo_open_price': vlo_price,
                        'shel_commission_open': n_shares_shel * shel_price * commission,
                        'vlo_commission_open': n_shares_vlo * vlo_price * commission
                    })
                    trade_log.append({
                        'Date': date,
                        'Type': 'OPEN',
                        'Signal': 1,
                        'SHEL_Shares': n_shares_shel,
                        'VLO_Shares': -n_shares_vlo,
                        'SHEL_Price': shel_price,
                        'VLO_Price': vlo_price,
                        'P&L': 0,
                        'Cumulative_P&L': cumulative_pnl,
                        'Return (%)': (cumulative_pnl / capital_init) * 100,
                        'Capital': capital
                    })
                else:
                    print(f"No hay capital suficiente para abrir posición en {date}: Capital disponible = {capital_available:.2f}, Costo = {shel_cost + vlo_cost:.2f}")
                    continue

        else:  # Cierre
            positions_to_close = []
            for j, position in enumerate(open_positions_with_details):
                # Calcular P&L
                shel_shares = position['shel_shares']
                vlo_shares = position['vlo_shares']
                shel_open_price = position['shel_open_price']
                vlo_open_price = position['vlo_open_price']
                shel_commission_open = position['shel_commission_open']
                vlo_commission_open = position['vlo_commission_open']
                shel_commission_close = abs(shel_shares) * shel_price * commission
                vlo_commission_close = abs(vlo_shares) * vlo_price * commission

                # P&L para SHEL
                if shel_shares > 0:  # LONG SHEL
                    shel_pnl = (shel_price - shel_open_price) * shel_shares - (shel_commission_open + shel_commission_close)
                else:  # SHORT SHEL
                    shel_pnl = (shel_open_price - shel_price) * abs(shel_shares) - (shel_commission_open + shel_commission_close)

                # P&L para VLO
                if vlo_shares > 0:  # LONG VLO
                    vlo_pnl = (vlo_price - vlo_open_price) * vlo_shares - (vlo_commission_open + vlo_commission_close)
                else:  # SHORT VLO
                    vlo_pnl = (vlo_open_price - vlo_price) * abs(vlo_shares) - (vlo_commission_open + vlo_commission_close)

                # Total P&L de la operación
                total_pnl = shel_pnl + vlo_pnl
                capital += total_pnl
                cumulative_pnl += total_pnl

                trade_log.append({
                    'Date': date,
                    'Type': 'CLOSE',
                    'Signal': signal,
                    'SHEL_Shares': shel_shares,
                    'VLO_Shares': vlo_shares,
                    'SHEL_Price': shel_price,
                    'VLO_Price': vlo_price,
                    'P&L': total_pnl,
                    'Cumulative_P&L': cumulative_pnl,
                    'Return (%)': (cumulative_pnl / capital_init) * 100,
                    'Capital': capital
                })
                positions_to_close.append(j)

            # Eliminar posiciones cerradas
            for j in sorted(positions_to_close, reverse=True):
                open_positions_with_details.pop(j)

        capital_history.append({'Date': date, 'Capital': capital})

    # Convertir trade_log y capital_history a DataFrames
    trade_log_df = pd.DataFrame(trade_log)
    capital_history_df = pd.DataFrame(capital_history).set_index('Date')

    return trade_log_df, capital_history_df