from functions import (
    download_data,
    log_transform,
    run_adf_test,
    correlation_analysis,
    engle_granger_cointegration_test
)


def main():
    # 1. Descarga de datos
    data_qqq = download_data('QQQ')
    data_xlk = download_data('XLK')

    # 2. Transformación logarítmica
    log_data_qqq = log_transform(data_qqq)
    log_data_xlk = log_transform(data_xlk)

    # 3. Prueba ADF individual
    print("\nPrueba ADF para QQQ (logarítmico):")
    run_adf_test(log_data_qqq, "QQQ Log")

    print("\nPrueba ADF para XLK (logarítmico):")
    run_adf_test(log_data_xlk, "XLK Log")

    # 4. Análisis de correlación
    print("\nAnálisis de correlación (logarítmico):")
    correlation_analysis(log_data_qqq, log_data_xlk, "QQQ Log", "XLK Log")

    # 5. Test de Cointegración Engle-Granger
    print("\nTest de Cointegración Engle-Granger (QQQ vs XLK, logarítmico):")
    engle_granger_cointegration_test(
        log_data_qqq,
        log_data_xlk,
        name1="QQQ_Log",
        name2="XLK_Log"
    )


if __name__ == "__main__":
    main()