#!/usr/bin/env python3
"""
Ejemplo simplificado de forecasting de demanda energética
Basado en el paper de Joaquín Amat Rodrigo y Javier Escobar Ortiz

Este script muestra un ejemplo básico de cómo usar el sistema de forecasting
sin todas las funcionalidades avanzadas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skforecast.datasets import fetch_dataset
from skforecast.recursive import ForecasterRecursive
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures

def ejemplo_basico():
    """
    Ejemplo básico de forecasting de demanda energética
    """
    print("=" * 60)
    print("EJEMPLO BÁSICO DE FORECASTING DE DEMANDA ENERGÉTICA")
    print("=" * 60)
    
    # 1. Cargar datos
    print("1. Cargando datos...")
    datos = fetch_dataset(name='vic_electricity', raw=True)
    datos['Time'] = pd.to_datetime(datos['Time'], format='%Y-%m-%dT%H:%M:%SZ')
    datos = datos.set_index('Time')
    datos = datos.asfreq('30min')
    
    # Agregar a datos horarios
    datos = datos.resample(rule="h", closed="left", label="right").agg({
        "Demand": "mean",
        "Temperature": "mean",
        "Holiday": "mean",
    })
    
    # Filtrar datos
    datos = datos.loc['2012-01-01 00:00:00':'2014-12-30 23:00:00', :].copy()
    
    print(f"Datos cargados: {datos.shape[0]} registros")
    print(f"Período: {datos.index.min()} a {datos.index.max()}")
    
    # 2. Dividir datos
    print("\n2. Dividiendo datos...")
    fin_train = '2013-12-31 23:59:00'
    fin_validacion = '2014-11-30 23:59:00'
    
    datos_train = datos.loc[: fin_train, :].copy()
    datos_val   = datos.loc[fin_train:fin_validacion, :].copy()
    datos_test  = datos.loc[fin_validacion:, :].copy()
    
    print(f"Train: {len(datos_train)} registros")
    print(f"Validación: {len(datos_val)} registros")
    print(f"Test: {len(datos_test)} registros")
    
    # 3. Crear modelo simple
    print("\n3. Creando modelo...")
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
        regressor       = LGBMRegressor(random_state=15926, verbose=-1),
        lags            = 24,  # Últimas 24 horas
        window_features = window_features
    )
    
    # 4. Entrenar modelo
    print("4. Entrenando modelo...")
    forecaster.fit(y=datos.loc[:fin_validacion, 'Demand'])
    print("Modelo entrenado exitosamente")
    
    # 5. Hacer predicciones
    print("\n5. Haciendo predicciones...")
    predicciones = forecaster.predict(steps=24)
    print(f"Predicciones generadas para las próximas 24 horas")
    print(f"Primera predicción: {predicciones.iloc[0]:.2f} MW")
    print(f"Última predicción: {predicciones.iloc[-1]:.2f} MW")
    
    # 6. Evaluar modelo con backtesting
    print("\n6. Evaluando modelo con backtesting...")
    cv = TimeSeriesFold(
        steps              = 24,
        initial_train_size = len(datos.loc[:fin_validacion]),
        refit              = False
    )
    
    metrica, predicciones_backtest = backtesting_forecaster(
        forecaster = forecaster,
        y          = datos['Demand'],
        cv         = cv,
        metric     = 'mean_absolute_error',
        verbose    = False
    )
    
    mae = metrica.iloc[0, 0]
    print(f"Error absoluto medio (MAE): {mae:.2f} MW")
    
    # 7. Visualizar resultados
    print("\n7. Generando visualización...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Datos de entrenamiento
    datos_train['Demand'].plot(ax=ax, label='Train', alpha=0.7, color='blue')
    
    # Datos de validación
    datos_val['Demand'].plot(ax=ax, label='Validation', alpha=0.7, color='orange')
    
    # Datos de test
    datos_test['Demand'].plot(ax=ax, label='Test', alpha=0.7, color='green')
    
    # Predicciones del backtest
    predicciones_backtest['pred'].plot(ax=ax, label='Predicciones', alpha=0.8, color='red')
    
    ax.set_title('Demanda Eléctrica - Predicciones vs Valores Reales')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Demanda (MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 8. Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Modelo: LightGBM con 24 lags")
    print(f"MAE: {mae:.2f} MW")
    print(f"Predicciones generadas para {len(predicciones)} horas")
    print("¡Ejemplo completado exitosamente!")
    
    return forecaster, mae

if __name__ == "__main__":
    forecaster, mae = ejemplo_basico()
