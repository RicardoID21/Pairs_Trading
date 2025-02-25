from functions import (
    download_data,
    log_transform,
    run_adf_test,
    correlation_analysis,
    engle_granger_cointegration_test,
    plot_etfs
)

def main():
    # 1. Descarga de datos para Home Depot (HD) y Lowe's (LOW)
    data_hd = download_data('HD')
    data_low = download_data('LOW')

    # 2. Transformación logarítmica de los precios
    log_data_hd = log_transform(data_hd)
    log_data_low = log_transform(data_low)

    # 3. Pruebas ADF individuales para cada activo
    print("\nPrueba ADF para HD (logarítmico):")
    run_adf_test(log_data_hd, "HD Log")

    print("\nPrueba ADF para LOW (logarítmico):")
    run_adf_test(log_data_low, "LOW Log")

    # 4. Análisis de correlación (logarítmico)
    print("\nAnálisis de correlación (logarítmico) entre HD y LOW:")
    correlation_analysis(log_data_hd, log_data_low, "HD Log", "LOW Log")

    # 5. Test de Cointegración Engle-Granger entre HD y LOW
    print("\nTest de Cointegración Engle-Granger (HD vs LOW, logarítmico):")
    engle_granger_cointegration_test(
        log_data_hd,
        log_data_low,
        name1="HD_Log",
        name2="LOW_Log"
    )

    # 6. Graficar los precios logarítmicos de HD y LOW para ver sus cruces
    print("\nGenerando gráfico de precios logarítmicos:")
    plot_etfs(log_data_hd, log_data_low, "HD Log", "LOW Log", title="Comparación logarítmica: HD vs LOW")

if __name__ == "__main__":
    main()