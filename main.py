from functions import (
    download_data,
    log_transform,
    run_adf_test,
    correlation_analysis,
    engle_granger_cointegration_test,
    plot_etfs
)
import pandas as pd

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

    # 6. Filtrar para graficar solo los últimos 5 años
    last5_shel = log_data_shel.loc[log_data_shel.index >= (log_data_shel.index.max() - pd.DateOffset(years=5))]
    last5_vlo = log_data_vlo.loc[log_data_vlo.index >= (log_data_vlo.index.max() - pd.DateOffset(years=5))]

    # 7. Normalizar las series para que ambas comiencen en el mismo nivel (por ejemplo, 100)
    norm_last5_shel = (last5_shel - last5_shel.min()) / (last5_shel.max() - last5_shel.min()) * 100
    norm_last5_vlo = (last5_vlo - last5_vlo.min()) / (last5_vlo.max() - last5_vlo.min()) * 100

    print("\nGenerando gráfico escalado (normalizado) de precios logarítmicos (últimos 5 años):")
    plot_etfs(norm_last5_shel, norm_last5_vlo, "SHEL Normalizado", "VLO Normalizado", title="Comparación escalada: SHEL vs VLO (últimos 5 años)")

if __name__ == "__main__":
    main()