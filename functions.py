import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


def download_data(ticker: str, period: str = '10y'):
    """
    Descarga el precio ajustado de un ticker para el período especificado.
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
    """
    Aplica la transformación logarítmica a la serie.
    """
    return np.log(series)


def run_adf_test(series, name: str):
    """
    Aplica el test ADF a la serie y muestra los resultados.
    """
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
    """
    Combina dos series en un DataFrame y calcula la correlación de Pearson.
    """
    data = pd.concat([series1, series2], axis=1, join='inner')
    data.columns = [name1, name2]
    corr = data.corr().iloc[0, 1]
    print(f"Correlación entre {name1} y {name2}: {corr:.4f}")
    return corr


def engle_granger_cointegration_test(series1, series2, name1="Serie1", name2="Serie2"):
    """
    Realiza la prueba de cointegración de Engle-Granger:
    1. Regresión OLS: Serie1 ~ Serie2
    2. Obtención de residuales
    3. Test ADF sobre los residuales
    """
    # Alineamos las series
    df = pd.concat([series1, series2], axis=1, join='inner').dropna()
    df.columns = [name1, name2]

    # Agregar constante y ajustar el modelo: Serie1 = alpha + beta * Serie2 + e
    df_reg = sm.add_constant(df)
    model = sm.OLS(df_reg[name1], df_reg[[name2, 'const']]).fit()

    # Extraer residuales y aplicar ADF
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
    """
    Grafica dos series en el mismo gráfico para comparar sus movimientos.
    """
    # Alinear series por fecha
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
    """
    Realiza el test de cointegración de Johansen sobre el DataFrame df,
    que contiene las series (en columnas) a analizar.

    Parámetros:
      - df: DataFrame con las series en columnas.
      - det_order: Orden del determinista (0 para sin tendencia).
      - k_ar_diff: Número de diferencias a usar en el test.

    Imprime y retorna el objeto resultado del test.
    """
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
    """
    Realiza una regresión OLS de la serie dependiente (dep) sobre la serie independiente (indep)
    y grafica la dispersión junto con la línea de regresión.

    Parámetros:
      - series_dep: Serie de la variable dependiente.
      - series_indep: Serie de la variable independiente.
      - dep_label: Etiqueta para la variable dependiente (por defecto "Y").
      - indep_label: Etiqueta para la variable independiente (por defecto "X").
    """
    # Alinear las series por fechas (inner join)
    combined = pd.concat([series_dep, series_indep], axis=1, join='inner')
    combined.columns = [dep_label, indep_label]

    # Variables para la regresión
    X = combined[indep_label]
    y = combined[dep_label]
    X_const = sm.add_constant(X)  # Agrega una constante para el intercepto

    # Ajustar el modelo OLS
    model = sm.OLS(y, X_const).fit()
    print(model.summary())

class KalmanFilterReg:
    def __init__(self):
        # Estado inicial (alpha=1, beta=1 por defecto)
        self.x = np.array([1.0, 1.0])
        self.A = np.eye(2)                # Matriz de transición
        self.Q = np.eye(2) * 0.01          # Covarianza del estado
        self.R = np.array([[1]]) * 10 # Covarianza del error en la observación
        self.P = np.eye(2) * 10           # Covarianza inicial del estado

    def predict(self):
        # Propagación de la incertidumbre
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, x_val, y_val):
        """
        Observación: y_val = alpha + beta * x_val
        """
        C = np.array([[1, x_val]])  # Matriz de observación (1x2)
        S = C @ self.P @ C.T + self.R
        K = self.P @ C.T @ np.linalg.inv(S)  # Ganancia de Kalman

        y_pred = C @ self.x                # Predicción de la observación
        resid = y_val - y_pred            # Residuo

        # Actualizar estado (alpha, beta)
        self.x = self.x + K.ravel() * resid
        # Actualizar covarianza
        self.P = (np.eye(2) - K @ C) @ self.P


def run_kalman_filter(log_x, log_y):
    """
    Aplica el filtro de Kalman en modo rolling para estimar alpha, beta y la predicción de y en cada paso.
    log_x: Serie (SHEL log)
    log_y: Serie (VLO log)
    Retorna: DataFrame con columnas ['alpha', 'beta', 'pred_y'] indexado por fecha.
    """
    # Alinear y limpiar
    df = pd.concat([log_x, log_y], axis=1).dropna()
    df.columns = ['x', 'y']

    # Inicializar el filtro
    kf = KalmanFilterReg()

    alphas = []
    betas = []
    preds = []   # Aquí guardaremos la predicción: alpha + beta * x

    # Iterar sobre las observaciones en orden temporal
    for date, row in df.iterrows():
        x_ = row['x']
        y_ = row['y']

        kf.predict()
        kf.update(x_, y_)

        # Extraer alpha, beta del estado
        alpha, beta = kf.x
        alphas.append(alpha)
        betas.append(beta)
        # Predicción: alpha + beta * x
        preds.append(alpha + beta * x_)

    # Construir el resultado
    out = pd.DataFrame({
        'alpha': alphas,
        'beta': betas,
        'pred_y': preds
    }, index=df.index)

    return out



def generate_vecm_signals(log_data_shel, log_data_vlo, det_order=0, k_ar_diff=1, threshold_sigma=1.5):
    """
    1) Runs Johansen test to find cointegration rank.
    2) Fits a VECM with that rank.
    3) Obtains the Error Correction Term (ECT).
    4) Generates trading signals based on ±threshold_sigma * std(ECT).

    Returns:
      - df_signals: DataFrame with columns [ECT, signal], indexed by date.
      - vecm_res: The fitted VECM model object (for further inspection).
    """

    # Combine log data in a DataFrame, align by date
    df = pd.concat([log_data_shel, log_data_vlo], axis=1).dropna()
    df.columns = ['SHEL_Log', 'VLO_Log']

    # 1) Johansen test
    joh_res = coint_johansen(df, det_order, k_ar_diff)
    # Suppose you see from the trace stats that rank=1 is appropriate
    # (Check joh_res.lr1, joh_res.cvt to confirm)
    rank = 1

    # 2) Fit a VECM
    # deterministic='co' => constant inside cointegration relation
    model = VECM(df, deterministic='co', k_ar_diff=k_ar_diff, coint_rank=rank)
    vecm_res = model.fit()

    # 3) Obtain ECT: The error correction term from the fitted model.
    # Statsmodels provides .plot(), .resid, etc. But for the ECT, we can compute:
    # ECT[t] = beta^T * y[t-1] if rank=1 => single cointegrating relationship.
    # Alternatively, statsmodels >= 0.12 has 'vecm_res.ec_term' or we do it manually:

    # For a single cointegrating vector:
    beta = vecm_res.beta  # shape (2, rank=1) => we’ll flatten it
    beta = beta[:, 0]     # e.g. [beta_shel, beta_vlo]

    # If the model also includes a deterministic constant in cointegration,
    # it appears in the last row of `vecm_res.beta`. For a 2D system + constant,
    # shape might be (3,1). Adjust accordingly.
    # If you see something like beta.shape = (3,1), then the last element is the constant.

    # We'll check if there's a constant inside beta:
    # (If deterministic='co', row -1 might be the constant)
    has_const = (beta.shape[0] == 3)  # means [SHEL_Log, VLO_Log, const]
    if has_const:
        coint_const = beta[-1]
        beta_assets = beta[:-1]
    else:
        coint_const = 0.0
        beta_assets = beta

    # Now we compute ECT for each time t from the data
    # ECT[t] = beta_assets^T * y[t] + coint_const
    # But note: VECM uses y[t-1], so we shift the data by 1.
    # We'll do it in a simple approach:
    df_shift = df.shift(1).dropna()
    ect_values = []
    for i in range(len(df_shift)):
        row = df_shift.iloc[i]
        # y[t-1] = [SHEL_Log, VLO_Log]
        val = beta_assets[0]*row['SHEL_Log'] + beta_assets[1]*row['VLO_Log'] + coint_const
        ect_values.append(val)

    # Align ECT with the same index (shifted by 1)
    ect_series = pd.Series(ect_values, index=df_shift.index, name='ECT')

    # 4) Generate signals: ± threshold_sigma * std(ECT)
    # We can center ECT at its mean, or just rely on raw ECT. Here we do raw.
    ect_mean = ect_series.mean()
    ect_std = ect_series.std()
    up = ect_mean + threshold_sigma * ect_std
    down = ect_mean - threshold_sigma * ect_std

    # if ECT > up => short spread
    # if ECT < down => long spread
    # else => no signal (0)
    signals = []
    for val in ect_series:
        if val > up:
            signals.append(-1)  # short
        elif val < down:
            signals.append(1)   # long
        else:
            signals.append(0)

    df_signals = pd.DataFrame({
        'ECT': ect_series,
        'signal': signals
    }, index=ect_series.index)

    return df_signals, vecm_res

