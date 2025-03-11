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
    backtest_strategy
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
    norm_df = pd.concat([norm_shel, norm_vlo], axis=1, join='inner').sort_index()
    norm_df.columns = ['SHEL_Norm', 'VLO_Norm']

    # 4. Calcular el spread (Johansen, como en el código original)
    combined = pd.concat([data_shel, data_vlo], axis=1, join='inner').sort_index()
    combined.columns = ["SHEL", "VLO"]
    spread_johansen = 0.19432877 * combined["SHEL"] - 0.06835929 * combined["VLO"]
    spread_centered = spread_johansen - spread_johansen.mean()  # Centrado como en el original
    spread_df = spread_centered.to_frame(name='spread')

    # 4.1. Validar la estacionariedad del spread
    print("\nPrueba ADF para el Spread de Johansen:")
    run_adf_test(spread_centered, "Spread Johansen")

    # 5. Calcular Dynamic Hedge Ratio con Kalman Filter
    kalman_df = run_kalman_filter(data_shel, data_vlo)
    hedge_df = pd.concat([data_shel, data_vlo, kalman_df['beta']], axis=1, join='inner').dropna()
    hedge_df.columns = ['SHEL', 'VLO', 'Hedge_Ratio']
    print("\nPrimeras filas del DataFrame con Dynamic Hedge Ratio:")
    print(hedge_df.head())

    # 6. Generar señales de apertura y cierre basadas en el spread de Johansen
    spread_std = spread_centered.std()
    threshold_up = 1.5 * spread_std  # Restaurado a ±1.5 STD
    threshold_down = -1.5 * spread_std

    signals_df = pd.DataFrame(index=spread_df.index)
    signals_df['spread'] = spread_df['spread']
    signals_df[
        'signal'] = 0  # 0: Sin acción, 1: Apertura Long SHEL/Short VLO, -1: Apertura Long VLO/Short SHEL, 2: Cierre

    # Lista para rastrear posiciones abiertas
    open_positions = []  # Lista de tuplas (fecha_apertura, tipo_señal)
    trades = []  # Lista para almacenar todas las operaciones

    # Lógica para aperturas y cierres
    for i in range(1, len(signals_df)):
        current_spread = signals_df['spread'].iloc[i]
        prev_spread = signals_df['spread'].iloc[i - 1] if i > 0 else 0

        # Detectar aperturas (si el spread cruza el umbral ±1.5 STD)
        if current_spread > threshold_up and prev_spread <= threshold_up:
            signals_df.at[signals_df.index[i], 'signal'] = -1  # Apertura Long VLO/Short SHEL
            open_positions.append((signals_df.index[i], -1))
            trades.append({
                'date': signals_df.index[i],
                'type': 'OPEN',
                'signal': -1,
                'spread': current_spread,
                'hedge_ratio': hedge_df.loc[signals_df.index[i], 'Hedge_Ratio']
            })
            print(f"Apertura en {signals_df.index[i]}: Long VLO, Short SHEL (Spread = {current_spread:.2f})")
        elif current_spread < threshold_down and prev_spread >= threshold_down:
            signals_df.at[signals_df.index[i], 'signal'] = 1  # Apertura Long SHEL/Short VLO
            open_positions.append((signals_df.index[i], 1))
            trades.append({
                'date': signals_df.index[i],
                'type': 'OPEN',
                'signal': 1,
                'spread': current_spread,
                'hedge_ratio': hedge_df.loc[signals_df.index[i], 'Hedge_Ratio']
            })
            print(f"Apertura en {signals_df.index[i]}: Long SHEL, Short VLO (Spread = {current_spread:.2f})")

        # Detectar cierres para cada posición abierta (cuando el spread cruza 0)
        positions_to_close = []
        for j, (open_date, signal_type) in enumerate(open_positions):
            if (prev_spread > 0 and current_spread <= 0) or (prev_spread < 0 and current_spread >= 0):
                signals_df.at[signals_df.index[i], 'signal'] = 2  # Cierre
                trades.append({
                    'date': signals_df.index[i],
                    'type': 'CLOSE',
                    'signal': signal_type,
                    'spread': current_spread,
                    'hedge_ratio': hedge_df.loc[signals_df.index[i], 'Hedge_Ratio']
                })
                positions_to_close.append(j)
                print(f"Cierre en {signals_df.index[i]}: Spread cruzó 0 (Spread = {current_spread:.2f})")

        # Eliminar posiciones cerradas
        for j in sorted(positions_to_close, reverse=True):
            open_positions.pop(j)

    # Separar aperturas y cierres para visualización
    open_short_shel_long_vlo = signals_df[signals_df['signal'] == -1].index
    open_long_shel_short_vlo = signals_df[signals_df['signal'] == 1].index
    close_positions = signals_df[signals_df['signal'] == 2].index

    # 7. Visualización
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(norm_df.index, norm_df['SHEL_Norm'], label="SHEL Normalizado", color='tab:blue')
    ax1.plot(norm_df.index, norm_df['VLO_Norm'], label="VLO Normalizado", color='tab:orange')
    # Aperturas Long VLO, Short SHEL
    ax1.scatter(open_short_shel_long_vlo, norm_df['SHEL_Norm'].reindex(open_short_shel_long_vlo),
                marker='v', color='red', s=100, label='Short SHEL Open')
    ax1.scatter(open_short_shel_long_vlo, norm_df['VLO_Norm'].reindex(open_short_shel_long_vlo),
                marker='^', color='green', s=100, label='Long VLO Open')
    # Aperturas Long SHEL, Short VLO
    ax1.scatter(open_long_shel_short_vlo, norm_df['SHEL_Norm'].reindex(open_long_shel_short_vlo),
                marker='^', color='green', s=100, label='Long SHEL Open')
    ax1.scatter(open_long_shel_short_vlo, norm_df['VLO_Norm'].reindex(open_long_shel_short_vlo),
                marker='v', color='red', s=100, label='Short VLO Open')
    # Cierres
    ax1.scatter(close_positions, norm_df['SHEL_Norm'].reindex(close_positions),
                marker='o', color='black', s=50, label='Close Position')
    ax1.set_title("Comparación normalizada (Min-Max) SHEL vs VLO + Señales (Johansen)")
    ax1.set_ylabel("Precio Normalizado")
    ax1.grid(True)
    handles1, labels1 = ax1.get_legend_handles_labels()
    unique = list(dict(zip(labels1, handles1)).items())
    ax1.legend([u[1] for u in unique], [u[0] for u in unique], loc='best')

    ax2.plot(signals_df.index, signals_df['spread'], label="Spread (Johansen)", color='magenta')
    ax2.axhline(threshold_up, color='blue', linestyle='--', label='+1.5 Sigma')
    ax2.axhline(threshold_down, color='blue', linestyle='--', label='-1.5 Sigma')
    ax2.axhline(0, color='black', linestyle='--', label='Nivel 0 (Cierre)')
    ax2.scatter(open_short_shel_long_vlo, signals_df['spread'].reindex(open_short_shel_long_vlo),
                marker='v', color='red', s=100, label='Open Short SHEL')
    ax2.scatter(open_long_shel_short_vlo, signals_df['spread'].reindex(open_long_shel_short_vlo),
                marker='^', color='green', s=100, label='Open Long SHEL')
    ax2.scatter(close_positions, signals_df['spread'].reindex(close_positions),
                marker='o', color='black', s=50, label='Close')
    ax2.set_title("Spread (Johansen) con ±1.5 STD y Cierres al cruzar 0")
    ax2.set_xlabel("Fecha")
    ax2.set_ylabel("Spread")
    ax2.grid(True)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()

    # 7.1. Gráfico de precios reales de SHEL y VLO
    plt.figure(figsize=(12, 6))
    plt.plot(combined.index, combined['SHEL'], label='SHEL', color='tab:blue')
    plt.plot(combined.index, combined['VLO'], label='VLO', color='tab:orange')
    plt.title("Precios Reales de SHEL y VLO (2015-2025)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Ajustado")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 7.2. Gráfico de precios reales de SHEL y VLO en 2019
    combined_2019 = combined[(combined.index.year >= 2019) & (combined.index.year <= 2019)]
    plt.figure(figsize=(12, 6))
    plt.plot(combined_2019.index, combined_2019['SHEL'], label='SHEL', color='tab:blue')
    plt.plot(combined_2019.index, combined_2019['VLO'], label='VLO', color='tab:orange')
    plt.title("Precios Reales de SHEL y VLO (2019)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio Ajustado")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 8. Backtest usando la función backtest_strategy
    trade_log_df, capital_history_df = backtest_strategy(
        hedge_df=hedge_df,
        trades=trades,
        capital_init=1_000_000,
        commission=0.00125,
        margin_req=0.3,
        n_shares_shel=50
    )

    # Mostrar el registro de operaciones
    print("\nRegistro de operaciones:")
    print(trade_log_df)

    # 9. Gráfico de la evolución del capital
    plt.figure(figsize=(12, 6))
    plt.plot(capital_history_df.index, capital_history_df['Capital'], label="Capital", color='green')
    plt.title("Evolución del Capital (Pairs Trading SHEL vs VLO)")
    plt.xlabel("Fecha")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 10. Kalman Filter comparación (referencia)
    df_compare = pd.concat([data_vlo, kalman_df['pred_y']], axis=1).dropna()
    df_compare.columns = ['VLO', 'pred_y']
    plt.figure(figsize=(12, 6))
    plt.plot(df_compare.index, df_compare['VLO'], label='VLO', color='yellow')
    plt.plot(df_compare.index, df_compare['pred_y'], label='Predicción Kalman', color='red', alpha=0.7)
    plt.title("Kalman Filter: Comparación VLO real vs. Predicha")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 11. Señales VECM (referencia)
    df_signals_vecm, vecm_res = generate_vecm_signals(data_shel, data_vlo)
    print("\nSeñales VECM generadas (referencia):")
    print(df_signals_vecm.head(10))

    return hedge_df, signals_df, trades, trade_log_df, capital_history_df


if __name__ == "__main__":
    hedge_df, signals_df, trades, trade_log_df, capital_history_df = main()