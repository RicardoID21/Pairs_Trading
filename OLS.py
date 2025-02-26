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

    # Graficar los datos y la línea de regresión
    plt.figure(figsize=(10, 6))
    plt.scatter(combined[indep_label], combined[dep_label], label="Datos", color="blue", alpha=0.6)
    plt.plot(combined[indep_label], model.predict(X_const), color="red", linewidth=2, label="Línea de Regresión OLS")
    plt.xlabel(f'{indep_label} (Log)')
    plt.ylabel(f'{dep_label} (Log)')
    plt.title(f'Regresión OLS: {dep_label} vs {indep_label}')
    plt.legend()
    plt.grid(True)
    plt.show()

