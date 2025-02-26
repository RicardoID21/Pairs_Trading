from functions import (
    download_data,
    log_transform,
    run_adf_test,
    correlation_analysis,
    engle_granger_cointegration_test,
    plot_etfs
)
from OLS import ols_regression_and_plot

import pandas as pd
import matplotlib.pyplot as plt
def main():
    # 1. Descarga de datos (10 años) para SHEL y VLO
    data_shel = download_data('SHEL', period='10y')
    data_vlo = download_data('VLO', period='10y')

    # 2. Transformación logarítmica de los precios
    log_data_shel = log_transform(data_shel)
    log_data_vlo = log_transform(data_vlo)

    # 3. Pruebas ADF individuales para cada activo (usando 10 años de datos)
    print("\nPrueba ADF para SHEL (logarítmico):")
    run_adf_test(log_data_shel, "SHEL Log")
    print("\nPrueba ADF para VLO (logarítmico):")
    run_adf_test(log_data_vlo, "VLO Log")

    # 4. Análisis de correlación (usando 10 años, logarítmico)
    print("\nAnálisis de correlación (logarítmico) entre SHEL y VLO:")
    correlation_analysis(log_data_shel, log_data_vlo, "SHEL Log", "VLO Log")

    # 5. Test de Cointegración Engle-Granger (usando 10 años)
    print("\nTest de Cointegración Engle-Granger (SHEL vs VLO, logarítmico):")
    engle_granger_cointegration_test(
        log_data_shel,
        log_data_vlo,
        name1="SHEL_Log",
        name2="VLO_Log"
    )
    # 6. Normalización Min-Max para que ambas series comiencen en el mismo nivel (0 a 100) usando los 10 años
    norm_shel = (log_data_shel - log_data_shel.min()) / (log_data_shel.max() - log_data_shel.min()) * 100
    norm_vlo = (log_data_vlo - log_data_vlo.min()) / (log_data_vlo.max() - log_data_vlo.min()) * 100

    print("\nGenerando gráfico escalado (normalizado) de precios logarítmicos (últimos 10 años):")
    plot_etfs(norm_shel, norm_vlo, "SHEL Normalizado", "VLO Normalizado",
              title="Comparación normalizada (Min-Max): SHEL vs VLO (10 años)")

    # 7. Alinear las series normalizadas y calcular el spread
    combined = pd.concat([norm_shel, norm_vlo], axis=1, join='inner')
    combined.columns = ['SHEL', 'VLO']
    spread = combined['SHEL'] - combined['VLO']

    # 8. Normalizar (estandarizar) el spread usando Z-Score
    spread_normalized = (spread - spread.mean()) / spread.std()
    print("Spread Normalizado - Media:", spread_normalized.mean(), "Std:", spread_normalized.std())

    # 9. Graficar el spread normalizado
    plt.figure(figsize=(12, 6))
    plt.plot(spread_normalized.index, spread_normalized, label='Spread Normalizado (Z-Score)', color='purple')
    plt.axhline(0, color='red', linestyle='--', label='Media = 0')
    plt.axhline(1.50, color='blue', linestyle='--', label='1.50 Sigma')
    plt.axhline(-1.50, color='blue', linestyle='--')

    plt.title("Spread Normalizado (Z-Score) entre SHEL y VLO (10 años)")
    plt.xlabel("Fecha")
    plt.ylabel("Spread (Z-Score)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Regresión OLS: SHEL_Log ~ VLO_Log ---")
    ols_regression_and_plot(log_data_shel, log_data_vlo, dep_label="SHEL_Log", indep_label="VLO_Log")

if __name__ == "__main__":
    main()




