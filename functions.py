import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


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