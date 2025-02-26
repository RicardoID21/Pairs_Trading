import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import (
    download_data,
    run_adf_test,
    correlation_analysis,
    engle_granger_cointegration_test,
    plot_etfs,
    johansen_cointegration_test,
    ols_regression_and_plot,
    run_kalman_filter,
    generate_vecm_signals
)

def main():
    # 1. Descarga de datos (10 años) para SHEL y VLO
    data_shel = download_data('SHEL', period='10y')
    data_vlo = download_data('VLO', period='10y')

    # 2. Transformación logarítmica de los precios
    log_data_shel = np.log(data_shel.dropna())
    log_data_vlo = np.log(data_vlo.dropna())

    # 3. ADF, correlación y Engle-Granger (como en tu código)
    print("\nPrueba ADF para SHEL (logarítmico):")
    run_adf_test(log_data_shel, "SHEL Log")
    print("\nPrueba ADF para VLO (logarítmico):")
    run_adf_test(log_data_vlo, "VLO Log")

    print("\nAnálisis de correlación (logarítmico) entre SHEL y VLO:")
    correlation_analysis(log_data_shel, log_data_vlo, "SHEL Log", "VLO Log")

    print("\nTest de Cointegración Engle-Granger (SHEL vs VLO, logarítmico):")
    engle_granger_cointegration_test(log_data_shel, log_data_vlo, name1="SHEL_Log", name2="VLO_Log")

    # 4. Normalizar (Min-Max) ambas series para graficar en el subplot superior
    norm_shel = (log_data_shel - log_data_shel.min()) / (log_data_shel.max() - log_data_shel.min()) * 100
    norm_vlo = (log_data_vlo - log_data_vlo.min()) / (log_data_vlo.max() - log_data_vlo.min()) * 100

    # 5. Calcular el spread (Johansen) usando el vector cointegrante
    #    Ejemplo: u_t = 7.98728502 * log(SHEL) - 4.9034967 * log(VLO)
    combined_log = pd.concat([log_data_shel, log_data_vlo], axis=1, join='inner').sort_index()
    combined_log.columns = ["SHEL_Log", "VLO_Log"]

    spread_johansen = 7.98728502 * combined_log["SHEL_Log"] - 4.9034967 * combined_log["VLO_Log"]
    spread_centered = spread_johansen - spread_johansen.mean()

    # 6. Definir umbrales ±1.5 std
    spread_std = spread_centered.std()
    threshold_up = 1.5 * spread_std   # Spread > +1.5 => Short SHEL, Long VLO
    threshold_down = -1.5 * spread_std # Spread < -1.5 => Short VLO, Long SHEL

    # 7. Identificar señales
    spread_df = spread_centered.to_frame(name='spread').sort_index()
    short_shel_long_vlo = spread_df[spread_df['spread'] > threshold_up]   # Señal
    short_vlo_long_shel = spread_df[spread_df['spread'] < threshold_down] # Señal

    print("\n--- Regresión OLS: SHEL_Log ~ VLO_Log ---")
    ols_regression_and_plot(log_data_shel, log_data_vlo, dep_label="SHEL_Log", indep_label="VLO_Log")

    # 8. Preparar DataFrame normalizado (para subplot superior)
    norm_df = pd.concat([norm_shel, norm_vlo], axis=1, join='inner').sort_index()
    norm_df.columns = ['SHEL_Norm', 'VLO_Norm']

    # 9. Crear figura con 2 subplots (compartiendo el eje x)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})


    # ==============================
    # Subplot superior: SHEL vs VLO normalizados
    # ==============================
    ax1.plot(norm_df.index, norm_df['SHEL_Norm'], label="SHEL Normalizado", color='tab:blue')
    ax1.plot(norm_df.index, norm_df['VLO_Norm'], label="VLO Normalizado", color='tab:orange')

    # Para marcar las señales en el gráfico superior, necesitamos dónde ponerlas.
    # Podemos dibujarlas en la propia línea de SHEL o VLO, o una línea media.
    # Ejemplo: en la misma línea del activo "caro".
    #   - Spread > +1.5 => Short SHEL => marcamos flecha roja en SHEL
    #                    => Long VLO  => marcamos flecha verde en VLO
    #   - Spread < -1.5 => Short VLO  => flecha roja en VLO
    #                    => Long SHEL => flecha verde en SHEL

    # short_shel_long_vlo => spread > threshold_up
    ax1.scatter(short_shel_long_vlo.index,
                norm_df['SHEL_Norm'].reindex(short_shel_long_vlo.index),
                marker='v', color='red', s=100, label='Short SHEL')
    ax1.scatter(short_shel_long_vlo.index,
                norm_df['VLO_Norm'].reindex(short_shel_long_vlo.index),
                marker='^', color='green', s=100, label='Long VLO')

    # short_vlo_long_shel => spread < threshold_down
    ax1.scatter(short_vlo_long_shel.index,
                norm_df['VLO_Norm'].reindex(short_vlo_long_shel.index),
                marker='v', color='red', s=100, label='Short VLO')
    ax1.scatter(short_vlo_long_shel.index,
                norm_df['SHEL_Norm'].reindex(short_vlo_long_shel.index),
                marker='^', color='green', s=100, label='Long SHEL')

    ax1.set_title("Comparación normalizada (Min-Max) SHEL vs VLO (10 años) + Señales Pairs Trading")
    ax1.set_ylabel("Precio Normalizado")
    ax1.grid(True)

    # Manejamos la leyenda sin duplicados
    # (al scatter le pusimos las mismas labels varias veces)
    handles1, labels1 = ax1.get_legend_handles_labels()
    # Filtramos duplicados conservando el orden
    unique = list(dict(zip(labels1, handles1)).items())
    ax1.legend([u[1] for u in unique], [u[0] for u in unique], loc='best')

    # ==============================
    # Subplot inferior: Spread Johansen con ±1.5 STD
    # ==============================
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


    kalman_df = run_kalman_filter(log_data_shel, log_data_vlo)
    # kalman_df tiene columns: ['alpha', 'beta', 'pred_y']

    # Graficar la evolución de alpha y beta
    plt.figure(figsize=(12, 6))
    plt.plot(kalman_df.index, kalman_df['alpha'], label='Alpha (Kalman)', color='red')
    plt.plot(kalman_df.index, kalman_df['beta'], label='Beta (Kalman)', color='blue')
    plt.title("Kalman Filter: Evolución de alpha y beta")
    plt.xlabel("Fecha")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Graficar la predicción vs. y real
    # (Opcional) Comparamos kalman_df['pred_y'] con log_data_vlo
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


    df_signals, vecm_res = generate_vecm_signals(log_data_shel, log_data_vlo)

    # Inspect the signals
    print(df_signals.head(10))

    # Plot the ECT with signals
    plt.figure(figsize=(12,6))
    plt.plot(df_signals.index, df_signals['ECT'], label='ECT', color='purple')
    # Mark up/down lines
    ect_mean = df_signals['ECT'].mean()
    ect_std = df_signals['ECT'].std()
    up_line = ect_mean + 1.5*ect_std
    down_line = ect_mean - 1.5*ect_std

    plt.axhline(up_line, color='blue', linestyle='--', label='Upper Threshold')
    plt.axhline(down_line, color='blue', linestyle='--', label='Lower Threshold')
    plt.axhline(ect_mean, color='red', linestyle='--', label='ECT Mean')

    # Mark signals on the ECT line
    short_signals = df_signals[df_signals['signal'] == -1]
    long_signals = df_signals[df_signals['signal'] == 1]

    plt.scatter(short_signals.index, short_signals['ECT'], marker='v', color='red', s=100, label='Short Spread')
    plt.scatter(long_signals.index, long_signals['ECT'], marker='^', color='green', s=100, label='Long Spread')

    plt.title("VECM: ECT & Trading Signals")
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    main()