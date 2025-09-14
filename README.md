# Predicción de Demanda Energética con Machine Learning

Este proyecto implementa un sistema completo de forecasting para la demanda eléctrica utilizando modelos de machine learning, basado en el paper de Joaquín Amat Rodrigo y Javier Escobar Ortiz.

## Características

- **Análisis exploratorio** de series temporales con visualizaciones interactivas
- **Modelo baseline** usando valores equivalentes de fechas anteriores
- **Modelo autoregresivo recursivo** con LightGBM
- **Variables exógenas** basadas en calendario, luz solar, festivos y temperatura
- **Optimización de hiperparámetros** con búsqueda bayesiana
- **Selección de predictores** con RFECV
- **Forecasting probabilístico** con intervalos de confianza
- **Explicabilidad del modelo** con SHAP values
- **Forecaster direct multi-step** para predicciones independientes
- **Predicción diaria anticipada** con gap temporal

## Instalación

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd forecasting
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Ejecuta el script principal:

```bash
python energy_demand_forecasting.py
```

## Estructura del Proyecto

```
forecasting/
├── energy_demand_forecasting.py  # Script principal
├── requirements.txt              # Dependencias
└── README.md                    # Este archivo
```

## Datos

El script utiliza el dataset `vic_electricity` de skforecast, que contiene datos de demanda eléctrica de Victoria, Australia, desde 2011-12-31 hasta 2014-12-31 con registros cada 30 minutos.

## Metodología

1. **Carga y procesamiento**: Conversión a datos horarios y división train-val-test
2. **Exploración**: Análisis de estacionalidad y autocorrelación
3. **Baseline**: Modelo de referencia con valores del día anterior
4. **Modelo principal**: LightGBM con lags y variables exógenas
5. **Optimización**: Búsqueda bayesiana de hiperparámetros
6. **Selección**: RFECV para identificar predictores más relevantes
7. **Probabilístico**: Intervalos de confianza con conformal prediction
8. **Explicabilidad**: SHAP values para interpretar predicciones
9. **Multi-step**: Estrategias recursiva y directa
10. **Anticipada**: Predicciones con gap temporal

## Resultados

El pipeline genera métricas de error (MAE) para cada modelo implementado, permitiendo comparar el rendimiento de diferentes enfoques.

## Referencias

- Amat Rodrigo, J., & Escobar Ortiz, J. (2025). Predicción (forecasting) de la demanda energética con machine learning. https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html
- skforecast: https://skforecast.org/

## Licencia

Este proyecto está basado en el trabajo de Joaquín Amat Rodrigo y Javier Escobar Ortiz, disponible bajo licencia CC BY-NC-SA 4.0.
