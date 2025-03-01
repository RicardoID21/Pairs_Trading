import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from functions import (
    download_data,
    run_adf_test,
    correlation_analysis,
    engle_granger_cointegration_test,
    plot_etfs,
    johansen_cointegration_test,
    ols_regression_and_plot,
    run_kalman_filter,
    generate_vecm_signals,
    backtest_vecm_strategy  # Nuestra función modificada de backtest
)

def main():
    # 1. Descargar datos (10 años) para SHEL y VLO
    data_shel = download_data('SHEL', period='10y')
    data_vlo = download_data('VLO', period='10y')

    # Crear un DataFrame con columnas ["SHEL", "VLO"]
    data = pd.concat([data_shel, data_vlo], axis=1).dropna()
    data.columns = ["SHEL", "VLO"]

    # 2. Transformar a logaritmos de precios (para análisis de cointegración)
    log_data_shel = np.log(data_shel.dropna())
    log_data_vlo = np.log(data_vlo.dropna())

    # 3. Ejecutar pruebas (ADF, correlación, cointegración)
    print("\nPrueba ADF para SHEL (logarítmico):")
    run_adf_test(log_data_shel, "SHEL Log")
    print("\nPrueba ADF para VLO (logarítmico):")
    run_adf_test(log_data_vlo, "VLO Log")
    print("\nAnálisis de correlación (logarítmico) entre SHEL y VLO:")
    correlation_analysis(log_data_shel, log_data_vlo, "SHEL Log", "VLO Log")
    print("\nTest de Cointegración Engle-Granger (SHEL vs VLO, logarítmico):")
    engle_granger_cointegration_test(log_data_shel, log_data_vlo, name1="SHEL_Log", name2="VLO_Log")
    johansen_cointegration_test(pd.concat([log_data_shel, log_data_vlo], axis=1).dropna(), det_order=0, k_ar_diff=1)

    # 4. Normalizar (Min-Max) ambas series para graficar en el subplot superior
    norm_shel = (log_data_shel - log_data_shel.min()) / (log_data_shel.max() - log_data_shel.min()) * 100
    norm_vlo = (log_data_vlo - log_data_vlo.min()) / (log_data_vlo.max() - log_data_vlo.min()) * 100

    # 5. Calcular el spread (Johansen) usando el vector cointegrante
    combined_log = pd.concat([log_data_shel, log_data_vlo], axis=1, join='inner').sort_index()
    combined_log.columns = ["SHEL_Log", "VLO_Log"]
    spread_johansen = 7.98728502 * combined_log["SHEL_Log"] - 4.9034967 * combined_log["VLO_Log"]
    spread_centered = spread_johansen - spread_johansen.mean()

    # 6. Definir umbrales ±1.5 std
    spread_std = spread_centered.std()
    threshold_up = 1.5 * spread_std
    threshold_down = -1.5 * spread_std

    # 7. Identificar señales (basadas en spread estático)
    spread_df = spread_centered.to_frame(name='spread').sort_index()
    short_shel_long_vlo = spread_df[spread_df['spread'] > threshold_up]
    short_vlo_long_shel = spread_df[spread_df['spread'] < threshold_down]

    print("\n--- Regresión OLS: SHEL_Log ~ VLO_Log ---")
    ols_regression_and_plot(log_data_shel, log_data_vlo, dep_label="SHEL_Log", indep_label="VLO_Log")

    # 8. Preparar DataFrame normalizado para el gráfico superior
    norm_df = pd.concat([norm_shel, norm_vlo], axis=1, join='inner').sort_index()
    norm_df.columns = ['SHEL_Norm', 'VLO_Norm']

    # 9. Crear figura con 2 subplots para comparar activos y spread
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(norm_df.index, norm_df['SHEL_Norm'], label="SHEL Normalizado", color='tab:blue')
    ax1.plot(norm_df.index, norm_df['VLO_Norm'], label="VLO Normalizado", color='tab:orange')
    ax1.scatter(short_shel_long_vlo.index, norm_df['SHEL_Norm'].reindex(short_shel_long_vlo.index),
                marker='v', color='red', s=100, label='Short SHEL')
    ax1.scatter(short_shel_long_vlo.index, norm_df['VLO_Norm'].reindex(short_shel_long_vlo.index),
                marker='^', color='green', s=100, label='Long VLO')
    ax1.scatter(short_vlo_long_shel.index, norm_df['VLO_Norm'].reindex(short_vlo_long_shel.index),
                marker='v', color='red', s=100, label='Short VLO')
    ax1.scatter(short_vlo_long_shel.index, norm_df['SHEL_Norm'].reindex(short_vlo_long_shel.index),
                marker='^', color='green', s=100, label='Long SHEL')
    ax1.set_title("Comparación normalizada (Min-Max) SHEL vs VLO (10 años) + Señales Pairs Trading")
    ax1.set_ylabel("Precio Normalizado")
    ax1.grid(True)
    handles1, labels1 = ax1.get_legend_handles_labels()
    unique = list(dict(zip(labels1, handles1)).items())
    ax1.legend([u[1] for u in unique], [u[0] for u in unique], loc='best')

    ax2.plot(spread_df.index, spread_df['spread'], label="Spread (Johansen) centrado", color='magenta')
    ax2.axhline(threshold_up, color='blue', linestyle='--', label='+1.5 Sigma')
    ax2.axhline(threshold_down, color='blue', linestyle='--', label='-1.5 Sigma')
    ax2.axhline(0, color='black', linestyle='--', label='Media 0')
    ax2.set_title("Spread (Johansen) con ±1.5 STD")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Spread")
    ax2.grid(True)
    ax2.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # 10. Kalman Filter: graficar evolución de alpha, beta y comparación de Y real vs. Y predicha
    kalman_df = run_kalman_filter(log_data_shel, log_data_vlo)
    plt.figure(figsize=(12, 6))
    plt.plot(kalman_df.index, kalman_df['alpha'], label='Alpha (Kalman)', color='red')
    plt.plot(kalman_df.index, kalman_df['beta'], label='Beta (Kalman)', color='blue')
    plt.title("Kalman Filter: Evolución de alpha y beta")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    df_compare = pd.concat([log_data_vlo, kalman_df['pred_y']], axis=1).dropna()
    df_compare.columns = ['VLO_log', 'pred_y']
    plt.figure(figsize=(12, 6))
    plt.plot(df_compare.index, df_compare['VLO_log'], label='VLO (log)', color='yellow')
    plt.plot(df_compare.index, df_compare['pred_y'], label='Predicción Kalman', color='red', alpha=0.7)
    plt.title("Kalman Filter: Comparación Y real vs. Y predicha")
    plt.xlabel("Fecha")
    plt.ylabel("Log(Precio)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 11. Generar señales VECM y ejecutar backtest de la estrategia (nuestra función en functions)
    df_signals, vecm_res = generate_vecm_signals(log_data_shel, log_data_vlo)
    print("Señales generadas:")
    print(df_signals.head(10))

    portfolio_series, trades = backtest_vecm_strategy(
        data,
        df_signals,
        capital_init=1_000_000,
        commission=0.125/100,
        margin_req=0.5,
        profit_threshold=0.15,
        stop_loss=0.1
    )

    df_trades = pd.DataFrame(trades)
    df_trades['pnl_filled'] = df_trades['pnl'].fillna(0)
    df_trades['cumulative_pnl'] = df_trades['pnl_filled'].cumsum()
    df_trades.drop(columns='pnl_filled', inplace=True)
    print("=== TRADES LOG with Cumulative PnL ===")
    print(df_trades)

    plt.figure(figsize=(12,6))
    plt.plot(portfolio_series.index, portfolio_series, label="Valor del Portafolio", color='green')
    plt.title("Evolución del Valor del Portafolio (Pairs Trading 'All In': SHEL y VLO)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    return portfolio_series, df_trades

if __name__ == "__main__":
    backtest_df, trades_df = main()
