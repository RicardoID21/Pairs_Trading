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
    backtest_vecm_strategy  # Nuestra función de backtest modificada
)


def main():
    # 1. Descarga de datos (10 años) para SHEL y VLO
    data_shel = download_data('SHEL', period='10y')
    data_vlo = download_data('VLO', period='10y')
    data = pd.concat([data_shel, data_vlo], axis=1).dropna()
    data.columns = ['SHEL', 'VLO']

    # 2. Ejecutar pruebas básicas
    print("\nPrueba ADF para SHEL:")
    run_adf_test(data_shel, "SHEL")
    print("\nPrueba ADF para VLO:")
    run_adf_test(data_vlo, "VLO")
    print("\nCorrelación entre SHEL y VLO:",
          correlation_analysis(data_shel, data_vlo, "SHEL", "VLO"))
    engle_granger_cointegration_test(data_shel, data_vlo, name1="SHEL", name2="VLO")
    johansen_cointegration_test(pd.concat([data_shel, data_vlo], axis=1).dropna(), det_order=0, k_ar_diff=1)

    # 3. Normalizar para graficar
    norm_shel = (data_shel - data_shel.min()) / (data_shel.max() - data_shel.min()) * 100
    norm_vlo = (data_vlo - data_vlo.min()) / (data_vlo.max() - data_vlo.min()) * 100

    # 4. Calcular el spread (Johansen)
    combined = pd.concat([data_shel, data_vlo], axis=1, join='inner').sort_index()
    combined.columns = ["SHEL", "VLO"]
    spread_johansen = 0.19432877 * combined["SHEL"] - 0.06835929 * combined["VLO"]
    spread_centered = spread_johansen - spread_johansen.mean()

    spread_std = spread_centered.std()
    threshold_up = 1.5 * spread_std
    threshold_down = -1.5 * spread_std

    spread_df = spread_centered.to_frame(name='spread').sort_index()
    short_shel_long_vlo = spread_df[spread_df['spread'] > threshold_up]
    short_vlo_long_shel = spread_df[spread_df['spread'] < threshold_down]

    print("\n--- Regresión OLS: SHEL ~ VLO ---")
    ols_regression_and_plot(data_shel, data_vlo, dep_label="SHEL", indep_label="VLO")

    norm_df = pd.concat([norm_shel, norm_vlo], axis=1, join='inner').sort_index()
    norm_df.columns = ['SHEL_Norm', 'VLO_Norm']

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

    # 5. Kalman Filter
    kalman_df = run_kalman_filter(data_shel, data_vlo)

    df_compare = pd.concat([data_vlo, kalman_df['pred_y']], axis=1).dropna()
    df_compare.columns = ['VLO', 'pred_y']
    plt.figure(figsize=(12, 6))
    plt.plot(df_compare.index, df_compare['VLO'], label='VLO', color='yellow')
    plt.plot(df_compare.index, df_compare['pred_y'], label='Predicción Kalman', color='red', alpha=0.7)
    plt.title("Kalman Filter: Comparación Y real vs. Y predicha")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. Generar señales VECM y ejecutar backtest de la estrategia
    df_signals, vecm_res = generate_vecm_signals(data_shel, data_vlo)
    print("Señales generadas:")
    print(df_signals.head(10))

    portfolio_series, trades = backtest_vecm_strategy(
        data,
        df_signals,
        capital_init=1_000_000,
        commission=0.125 / 100,
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

    plt.figure(figsize=(12, 6))
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