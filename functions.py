import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def download_data(ticker: str, period: str = '10y'):
    """
    Descarga el precio ajustado (Adj Close) de un ticker para el período especificado.
    Si no se encuentra la columna 'Adj Close', utiliza 'Close'.
    """
    data = yf.download(ticker, period=period)
    print(f"Columnas para {ticker}: {data.columns}")  # Depuración: muestra las columnas disponibles

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
    Se recomienda asegurarse de que la serie no contenga valores menores o iguales a 0.
    """
    return np.log(series)


def run_adf_test(series, name: str):
    """
    Aplica el test de Dickey-Fuller aumentado (ADF) a la serie y muestra los resultados.
    """
    result = adfuller(series.dropna())
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
    Realiza la prueba de cointegración de Engle-Granger entre dos series:
      1) Regresión OLS de Serie1 sobre Serie2
      2) Se obtienen los residuales
      3) Test ADF a los residuales

    Retorna:
      - El p-value del ADF sobre los residuales
      - El estadístico ADF
      - El modelo de regresión (opcional, por si quieres inspeccionar)

    Si el p-value < 0.05, se concluye que existe cointegración (los residuales son estacionarios).
    """
    # Alineamos ambas series por su fecha
    df = pd.concat([series1, series2], axis=1, join='inner').dropna()
    df.columns = [name1, name2]

    # Paso 1: regresión OLS (Serie1 = alpha + beta * Serie2 + e)
    # Añadimos una constante
    df = sm.add_constant(df)
    # Ajustamos el modelo: Serie1 como dependiente, Serie2 como independiente
    model = sm.OLS(df[name1], df[[name2, 'const']]).fit()

    # Paso 2: obtener los residuales
    residuals = model.resid

    # Paso 3: aplicar test ADF a los residuales
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