import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


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

import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd


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

