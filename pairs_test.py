#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


def descargar_datos(ticker: str, period: str = "10y") -> pd.Series:
    """
    Descarga los precios de cierre ajustado (si está disponible) o 'Close' de un activo.
    Devuelve una serie de pandas con el índice de fechas.
    """
    data = yf.download(ticker, period=period)
    print(f"\n--- Descargando datos para {ticker} ---")
    print(f"Columnas disponibles: {data.columns}")

    if 'Adj Close' in data.columns:
        precios = data['Adj Close']
    elif 'Adj. Close' in data.columns:
        precios = data['Adj. Close']
    else:
        print(f"No se encontró 'Adj Close' para {ticker}. Se usará 'Close'.")
        precios = data['Close']

    return precios.dropna()


def transformacion_logaritmica(serie: pd.Series) -> pd.Series:
    """
    Aplica la transformación logarítmica a la serie de precios.
    """
    return np.log(serie)


def test_adf(serie: pd.Series, nombre: str) -> None:
    """
    Aplica la prueba ADF (Augmented Dickey-Fuller) a la serie y muestra los resultados.
    """
    serie = serie.dropna()
    if len(serie) < 30:
        print(f"La serie {nombre} tiene muy pocos datos ({len(serie)}) para ejecutar ADF.")
        return

    resultado = adfuller(serie)
    estadistico_adf = resultado[0]
    p_value = resultado[1]
    valores_criticos = resultado[4]

    print(f"\n--- Test ADF para {nombre} ---")
    print(f"Estadístico ADF: {estadistico_adf:.4f}")
    print(f"p-value: {p_value:.4f}")
    print("Valores críticos:")
    for clave, valor in valores_criticos.items():
        print(f"  {clave}: {valor:.4f}")


def test_cointegracion_engle_granger(serie1: pd.Series, serie2: pd.Series, nombre1: str, nombre2: str) -> None:
    """
    Realiza la prueba de cointegración de Engle-Granger entre dos series:
      1) Regresión OLS: serie1 ~ serie2
      2) Se obtienen los residuales
      3) Test ADF a los residuales
    """
    df = pd.concat([serie1, serie2], axis=1, join='inner').dropna()
    df.columns = [nombre1, nombre2]

    # Agregamos constante a la regresión
    df_reg = sm.add_constant(df)
    # Regresión: serie1 = alpha + beta * serie2
    modelo = sm.OLS(df_reg[nombre1], df_reg[[nombre2, 'const']]).fit()
    residuales = modelo.resid.dropna()

    # Test ADF sobre residuales
    resultado_adf = adfuller(residuales)
    estadistico_adf = resultado_adf[0]
    p_value = resultado_adf[1]

    print(f"\n--- Test de Cointegración (Engle-Granger) entre {nombre1} y {nombre2} ---")
    print(f"Regresión: {nombre1} ~ {nombre2}")
    print(f"Coeficiente (beta): {modelo.params[nombre2]:.4f}")
    print(f"Constante (alpha) : {modelo.params['const']:.4f}")
    print(f"Estadístico ADF (residuales): {estadistico_adf:.4f}")
    print(f"p-value (residuales)         : {p_value:.4f}")

    if p_value < 0.05:
        print("=> Se rechaza la hipótesis nula. Hay evidencia de cointegración.\n")
    else:
        print("=> No se rechaza la hipótesis nula. No hay evidencia de cointegración.\n")


def analizar_par(ticker1: str, ticker2: str) -> None:
    """
    Descarga datos de dos activos, verifica su estacionariedad individual y la cointegración conjunta.
    """
    # 1. Descargar datos
    precios1 = descargar_datos(ticker1)
    precios2 = descargar_datos(ticker2)

    # 2. Transformar a log
    log1 = transformacion_logaritmica(precios1)
    log2 = transformacion_logaritmica(precios2)

    # 3. Test ADF individual
    test_adf(log1, f"{ticker1} (Log)")
    test_adf(log2, f"{ticker2} (Log)")

    # 4. Cointegración Engle-Granger
    test_cointegracion_engle_granger(log1, log2, f"{ticker1}_Log", f"{ticker2}_Log")


def main():
    """
    En la función main definimos los pares que queremos analizar.
    """
    pares = [
        ("^GSPC", "NDAQ"),  # Ejemplo de par con alta correlación y posible cointegración
        ("LLY", "JNJ"),  # Visa y Mastercard
        ("RTX", "OSK"),  # Coca-Cola y Pepsi
        ("SHEL", "VLO"),  # ExxonMobil y Chevron
        ("META", "MSFT")  # ETFs tecnológicos
    ]

    for p in pares:
        ticker1, ticker2 = p
        print("======================================")
        print(f"Analizando par: {ticker1} y {ticker2}")
        print("======================================")
        analizar_par(ticker1, ticker2)


if __name__ == "__main__":
    main()