#!/usr/bin/env python3
"""
Predicción (forecasting) de la demanda energética con machine learning

Este script implementa un sistema completo de forecasting para la demanda eléctrica
utilizando modelos de machine learning con skforecast.
"""

# ==============================================================================
# LIBRERÍAS
# ==============================================================================

# Tratamiento de datos
import numpy as np
import pandas as pd
from astral.sun import sun
from astral import LocationInfo
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
from feature_engine.timeseries.forecasting import WindowFeatures

# Gráficos
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from skforecast.plot import plot_residuals
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as poff
pio.templates.default = "seaborn"
poff.init_notebook_mode(connected=True)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'font.size': 8})

# Modelado y Forecasting
import skforecast
import lightgbm
import sklearn
from lightgbm import LGBMRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from skforecast.recursive import ForecasterEquivalentDate, ForecasterRecursive
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, bayesian_search_forecaster, backtesting_forecaster
from skforecast.feature_selection import select_features
from skforecast.preprocessing import RollingFeatures
from skforecast.plot import calculate_lag_autocorrelation, plot_residuals
from skforecast.metrics import calculate_coverage
import shap

# Configuración warnings
import warnings
warnings.filterwarnings('once')

def print_versions():
    """Imprime las versiones de las librerías principales"""
    color = '\033[1m\033[38;5;208m' 
    print(f"{color}Versión skforecast: {skforecast.__version__}")
    print(f"{color}Versión scikit-learn: {sklearn.__version__}")
    print(f"{color}Versión lightgbm: {lightgbm.__version__}")
    print(f"{color}Versión pandas: {pd.__version__}")
    print(f"{color}Versión numpy: {np.__version__}")

# ==============================================================================
# CARGA Y PROCESAMIENTO DE DATOS
# ==============================================================================

def load_and_process_data():
    """
    Carga y procesa los datos de demanda eléctrica desde archivo Excel
    """
    print("=" * 60)
    print("CARGANDO Y PROCESANDO DATOS")
    print("=" * 60)
    
    # Cargar datos desde Excel
    print("Cargando datos desde Demanda_Energia_SIN_2023.xlsx...")
    try:
        # Leer el archivo Excel con header en la fila 3 (0-indexed)
        datos = pd.read_excel('Demanda_Energia_SIN_2023.xlsx', header=3)
        print(f"Shape del dataset: {datos.shape}")
        print(f"Columnas: {list(datos.columns)}")
        print(f"Primeras 5 filas:")
        print(datos.head())
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'Demanda_Energia_SIN_2023.xlsx'")
        print("Asegúrate de que el archivo esté en el directorio actual.")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None, None, None, None, None, None
    
    # Detectar automáticamente la columna de fecha y demanda
    print("\nDetectando columnas de fecha y demanda...")
    date_columns = []
    demand_columns = []
    
    for col in datos.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['fecha', 'date', 'time', 'tiempo']):
            date_columns.append(col)
        if any(keyword in col_lower for keyword in ['demanda', 'demand', 'consumo', 'consumption', 'energia', 'energy']):
            demand_columns.append(col)
    
    print(f"Columnas de fecha detectadas: {date_columns}")
    print(f"Columnas de demanda detectadas: {demand_columns}")
    
    if not date_columns:
        print("Error: No se detectó ninguna columna de fecha. Por favor, verifica el archivo Excel.")
        return None, None, None, None, None, None
    
    if not demand_columns:
        print("Error: No se detectó ninguna columna de demanda. Por favor, verifica el archivo Excel.")
        return None, None, None, None, None, None
    
    # Usar la primera columna de fecha y demanda encontradas
    date_col = date_columns[0]
    demand_col = demand_columns[0]
    
    print(f"Usando columna de fecha: '{date_col}'")
    print(f"Usando columna de demanda: '{demand_col}'")
    
    # Preparar datos
    datos = datos[[date_col, demand_col]].copy()
    datos.columns = ['Time', 'Demand']
    
    # Convertir fecha
    print("\nConvirtiendo formato de fecha...")
    datos['Time'] = pd.to_datetime(datos['Time'])
    datos = datos.set_index('Time')
    datos = datos.sort_index()
    
    # Verificar que el índice temporal está completo
    print("Verificando completitud del índice temporal...")
    fecha_inicio = datos.index.min()
    fecha_fin = datos.index.max()
    freq_detected = pd.infer_freq(datos.index)
    print(f"Frecuencia detectada: {freq_detected}")
    print(f"Rango de fechas: {fecha_inicio} a {fecha_fin}")
    
    # Si no hay frecuencia detectada, asumir diaria y convertir a horaria
    if freq_detected is None or freq_detected == 'D':
        print("Datos diarios detectados. Creando datos horarios simulados...")
        # Crear datos horarios basados en patrones diarios
        datos_horarios = []
        for fecha, row in datos.iterrows():
            # Crear 24 horas para cada día
            for hora in range(24):
                # Simular patrón diario típico de demanda eléctrica
                # Pico en la mañana (8-10) y tarde (18-20)
                if 8 <= hora <= 10 or 18 <= hora <= 20:
                    factor = 1.2 + 0.3 * np.sin(2 * np.pi * hora / 24)
                elif 22 <= hora or hora <= 6:
                    factor = 0.6 + 0.2 * np.sin(2 * np.pi * hora / 24)
                else:
                    factor = 0.8 + 0.2 * np.sin(2 * np.pi * hora / 24)
                
                # Agregar variabilidad aleatoria
                factor += np.random.normal(0, 0.1)
                factor = max(0.3, factor)  # Evitar valores negativos
                
                demanda_horaria = row['Demand'] * factor / 24
                
                fecha_hora = pd.Timestamp(fecha.year, fecha.month, fecha.day, hora)
                datos_horarios.append({
                    'Time': fecha_hora,
                    'Demand': demanda_horaria
                })
        
        datos = pd.DataFrame(datos_horarios)
        datos = datos.set_index('Time')
        datos = datos.sort_index()
        print(f"Nuevo shape con datos horarios: {datos.shape}")
    else:
        datos = datos.asfreq(freq_detected)
    
    # Verificar valores faltantes
    print(f"Valores faltantes en demanda: {datos['Demand'].isnull().sum()}")
    if datos['Demand'].isnull().sum() > 0:
        print("Interpolando valores faltantes...")
        datos['Demand'] = datos['Demand'].interpolate(method='linear')
    
    # Crear variables adicionales si no existen
    print("Creando variables adicionales...")
    
    # Temperatura (simulada si no existe)
    if 'Temperature' not in datos.columns:
        print("Creando variable de temperatura simulada...")
        # Simular temperatura con estacionalidad
        datos['Temperature'] = 20 + 10 * np.sin(2 * np.pi * datos.index.dayofyear / 365) + \
                              5 * np.sin(2 * np.pi * datos.index.hour / 24) + \
                              np.random.normal(0, 2, len(datos))
    
    # Festivos (simulados si no existen)
    if 'Holiday' not in datos.columns:
        print("Creando variable de festivos simulada...")
        # Simular festivos (sábados y domingos)
        datos['Holiday'] = (datos.index.weekday >= 5).astype(int)
    
    # Separación datos train-val-test
    print("\nSeparando datos en train-val-test...")
    # Usar 70% para train, 15% para val, 15% para test
    total_len = len(datos)
    train_len = int(total_len * 0.7)
    val_len = int(total_len * 0.15)
    
    datos_train = datos.iloc[:train_len].copy()
    datos_val = datos.iloc[train_len:train_len + val_len].copy()
    datos_test = datos.iloc[train_len + val_len:].copy()
    
    fin_train = datos_train.index[-1]
    fin_validacion = datos_val.index[-1]
    
    print(f"Fechas train      : {datos_train.index.min()} --- {datos_train.index.max()}  (n={len(datos_train)})")
    print(f"Fechas validacion : {datos_val.index.min()} --- {datos_val.index.max()}  (n={len(datos_val)})")
    print(f"Fechas test       : {datos_test.index.min()} --- {datos_test.index.max()}  (n={len(datos_test)})")
    
    return datos, datos_train, datos_val, datos_test, fin_train, fin_validacion

# ==============================================================================
# EXPLORACIÓN GRÁFICA
# ==============================================================================

def exploratory_analysis(datos, datos_train, datos_val, datos_test):
    """
    Realiza análisis exploratorio de la serie temporal
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS EXPLORATORIO")
    print("=" * 60)
    
    # Gráfico de la serie temporal completa
    print("Generando gráfico de la serie temporal...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=datos_train.index, y=datos_train['Demand'], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=datos_val.index, y=datos_val['Demand'], mode='lines', name='Validation'))
    fig.add_trace(go.Scatter(x=datos_test.index, y=datos_test['Demand'], mode='lines', name='Test'))
    fig.update_layout(
        title  = 'Demanda eléctrica horaria',
        xaxis_title="Fecha",
        yaxis_title="Demanda (MWh)",
        legend_title="Partición:",
        width=800,
        height=400,
        margin=dict(l=20, r=20, t=35, b=20),
        legend=dict(orientation="h", yanchor="top", y=1, xanchor="left", x=0.001)
    )
    fig.show()
    
    # Gráfico de estacionalidad
    print("Generando gráficos de estacionalidad...")
    fig, axs = plt.subplots(2, 2, figsize=(8, 5), sharex=False, sharey=True)
    axs = axs.ravel()
    
    # Distribución de demanda por mes
    datos['month'] = datos.index.month
    datos.boxplot(column='Demand', by='month', ax=axs[0], flierprops={'markersize': 3, 'alpha': 0.3})
    datos.groupby('month')['Demand'].median().plot(style='o-', linewidth=0.8, ax=axs[0])
    axs[0].set_ylabel('Demand')
    axs[0].set_title('Distribución de demanda por mes', fontsize=9)
    
    # Distribución de demanda por día de la semana
    datos['week_day'] = datos.index.day_of_week + 1
    datos.boxplot(column='Demand', by='week_day', ax=axs[1], flierprops={'markersize': 3, 'alpha': 0.3})
    datos.groupby('week_day')['Demand'].median().plot(style='o-', linewidth=0.8, ax=axs[1])
    axs[1].set_ylabel('Demand')
    axs[1].set_title('Distribución de demanda por día de la semana', fontsize=9)
    
    # Distribución de demanda por hora del día
    datos['hour_day'] = datos.index.hour + 1
    datos.boxplot(column='Demand', by='hour_day', ax=axs[2], flierprops={'markersize': 3, 'alpha': 0.3})
    datos.groupby('hour_day')['Demand'].median().plot(style='o-', linewidth=0.8, ax=axs[2])
    axs[2].set_ylabel('Demand')
    axs[2].set_title('Distribución de demanda por hora del día', fontsize=9)
    
    # Distribución de demanda por día de la semana y hora del día
    mean_day_hour = datos.groupby(["week_day", "hour_day"])["Demand"].mean()
    mean_day_hour.plot(ax=axs[3])
    axs[3].set(
        title       = "Promedio de demanda",
        xticks      = [i * 24 for i in range(7)],
        xticklabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        xlabel      = "Día y hora",
        ylabel      = "Demand"
    )
    axs[3].title.set_size(10)
    
    fig.suptitle("Gráficos de estacionalidad", fontsize=12)
    fig.tight_layout()
    plt.show()
    
    # Gráficos de autocorrelación
    print("Generando gráficos de autocorrelación...")
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_acf(datos['Demand'], ax=ax, lags=60)
    plt.title("Función de Autocorrelación (ACF)")
    plt.show()
    
    fig, ax = plt.subplots(figsize=(5, 2))
    plot_pacf(datos['Demand'], ax=ax, lags=60)
    plt.title("Función de Autocorrelación Parcial (PACF)")
    plt.show()
    
    # Top 10 lags con mayor autocorrelación parcial absoluta
    print("\nTop 10 lags con mayor autocorrelación parcial absoluta:")
    lag_corr = calculate_lag_autocorrelation(
        data    = datos['Demand'],
        n_lags  = 60,
        sort_by = "partial_autocorrelation_abs"
    ).head(10)
    print(lag_corr)

# ==============================================================================
# MODELO BASELINE
# ==============================================================================

def create_baseline_model(datos, fin_validacion):
    """
    Crea y evalúa un modelo baseline
    """
    print("\n" + "=" * 60)
    print("MODELO BASELINE")
    print("=" * 60)
    
    # Crear un baseline: valor de la misma hora del día anterior
    forecaster = ForecasterEquivalentDate(
                     offset    = pd.DateOffset(days=1),
                     n_offsets = 1
                 )
    
    # Entrenamiento del forecaster
    forecaster.fit(y=datos.loc[:fin_validacion, 'Demand'])
    print("Forecaster baseline creado y entrenado")
    
    # Backtesting
    cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(datos.loc[:fin_validacion]),
            refit              = False
    )
    
    metrica, predicciones = backtesting_forecaster(
                              forecaster = forecaster,
                              y          = datos['Demand'],
                              cv         = cv,
                              metric     = 'mean_absolute_error'
                           )
    
    print(f"MAE del modelo baseline: {metrica.iloc[0, 0]:.2f}")
    return metrica.iloc[0, 0]

# ==============================================================================
# MODELO AUTOREGRESIVO RECURSIVO
# ==============================================================================

def create_recursive_model(datos, fin_validacion):
    """
    Crea y evalúa un modelo autoregresivo recursivo
    """
    print("\n" + "=" * 60)
    print("MODELO AUTOREGRESIVO RECURSIVO")
    print("=" * 60)
    
    # Crear el forecaster
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                     regressor       = LGBMRegressor(random_state=15926, verbose=-1),
                     lags            = 24,
                     window_features = window_features
                 )
                 
    # Entrenar el forecaster
    forecaster.fit(y=datos.loc[:fin_validacion, 'Demand'])
    print("Forecaster recursivo creado y entrenado")
    
    # Backtesting
    cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(datos.loc[:fin_validacion]),
            refit              = False
    )
    
    metrica, predicciones = backtesting_forecaster(
                                forecaster = forecaster,
                                y          = datos['Demand'],
                                cv         = cv,
                                metric     = 'mean_absolute_error',
                                verbose    = False
                            )
    
    print(f"MAE del modelo recursivo: {metrica.iloc[0, 0]:.2f}")
    return forecaster, metrica.iloc[0, 0]

# ==============================================================================
# VARIABLES EXÓGENAS
# ==============================================================================

def create_exogenous_variables(datos):
    """
    Crea variables exógenas basadas en calendario, luz solar, festivos y temperatura
    """
    print("\n" + "=" * 60)
    print("CREACIÓN DE VARIABLES EXÓGENAS")
    print("=" * 60)
    
    # Variables basadas en el calendario
    print("Creando variables de calendario...")
    features_to_extract = [
        'month',
        'week',
        'day_of_week',
        'hour'
    ]
    calendar_transformer = DatetimeFeatures(
        variables           = 'index',
        features_to_extract = features_to_extract,
        drop_original       = True,
    )
    variables_calendario = calendar_transformer.fit_transform(datos)[features_to_extract]
    
    # Variables basadas en la luz solar (usando coordenadas de Colombia)
    print("Creando variables solares...")
    location = LocationInfo(
        latitude  = 4.6,  # Bogotá, Colombia
        longitude = -74.1,
        timezone  = 'America/Bogota'
    )
    sunrise_hour = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunrise']
        for date in datos.index
    ]
    sunset_hour = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunset']
        for date in datos.index
    ]
    sunrise_hour = pd.Series(sunrise_hour, index=datos.index).dt.round("h").dt.hour
    sunset_hour = pd.Series(sunset_hour, index=datos.index).dt.round("h").dt.hour
    variables_solares = pd.DataFrame({
                            'sunrise_hour': sunrise_hour,
                            'sunset_hour': sunset_hour
                        })
    variables_solares['daylight_hours'] = (
        variables_solares['sunset_hour'] - variables_solares['sunrise_hour']
    )
    variables_solares["is_daylight"] = np.where(
        (datos.index.hour >= variables_solares["sunrise_hour"])
        & (datos.index.hour < variables_solares["sunset_hour"]),
        1,
        0,
    )
    
    # Variables basadas en festivos
    print("Creando variables de festivos...")
    variables_festivos = datos[['Holiday']].astype(int)
    variables_festivos['holiday_previous_day'] = variables_festivos['Holiday'].shift(24)
    variables_festivos['holiday_next_day'] = variables_festivos['Holiday'].shift(-24)
    
    # Variables basadas en temperatura
    print("Creando variables de temperatura...")
    wf_transformer = WindowFeatures(
        variables = ["Temperature"],
        window    = ["1D", "7D"],
        functions = ["mean", "max", "min"],
        freq      = "h",
    )
    variables_temp = wf_transformer.fit_transform(datos[['Temperature']])
    
    # Unión de variables exógenas
    print("Combinando variables exógenas...")
    assert all(variables_calendario.index == variables_solares.index)
    assert all(variables_calendario.index == variables_festivos.index)
    assert all(variables_calendario.index == variables_temp.index)
    variables_exogenas = pd.concat([
                            variables_calendario,
                            variables_solares,
                            variables_temp,
                            variables_festivos
                        ], axis=1)
    
    # Limpiar valores faltantes
    variables_exogenas = variables_exogenas.iloc[7 * 24:, :]
    variables_exogenas = variables_exogenas.iloc[:-24, :]
    
    # Codificación cíclica de las variables de calendario y luz solar
    print("Aplicando codificación cíclica...")
    features_to_encode = [
        "month",
        "week",
        "day_of_week",
        "hour",
        "sunrise_hour",
        "sunset_hour",
    ]
    max_values = {
        "month": 12,
        "week": 52,
        "day_of_week": 6,
        "hour": 24,
        "sunrise_hour": 24,
        "sunset_hour": 24,
    }
    cyclical_encoder = CyclicalFeatures(
        variables     = features_to_encode,
        max_values    = max_values,
        drop_original = False
    )
    
    variables_exogenas = cyclical_encoder.fit_transform(variables_exogenas)
    
    # Interacción entre variables exógenas
    print("Creando interacciones entre variables...")
    transformer_poly = PolynomialFeatures(
                            degree           = 2,
                            interaction_only = True,
                            include_bias     = False
                        ).set_output(transform="pandas")
    poly_cols = [
        'month_sin', 
        'month_cos',
        'week_sin',
        'week_cos',
        'day_of_week_sin',
        'day_of_week_cos',
        'hour_sin',
        'hour_cos',
        'sunrise_hour_sin',
        'sunrise_hour_cos',
        'sunset_hour_sin',
        'sunset_hour_cos',
        'daylight_hours',
        'is_daylight',
        'holiday_previous_day',
        'holiday_next_day',
        'Temperature_window_1D_mean',
        'Temperature_window_1D_min',
        'Temperature_window_1D_max',
        'Temperature_window_7D_mean',
        'Temperature_window_7D_min',
        'Temperature_window_7D_max',
        'Temperature',
        'Holiday'
    ]
    variables_poly = transformer_poly.fit_transform(variables_exogenas[poly_cols])
    variables_poly = variables_poly.drop(columns=poly_cols)
    variables_poly.columns = [f"poly_{col}" for col in variables_poly.columns]
    variables_poly.columns = variables_poly.columns.str.replace(" ", "__")
    assert all(variables_exogenas.index == variables_poly.index)
    variables_exogenas = pd.concat([variables_exogenas, variables_poly], axis=1)
    
    # Selección de variables exógenas incluidas en el modelo
    exog_features = []
    # Columnas que terminan con _sin o _cos son seleccionadas
    exog_features.extend(variables_exogenas.filter(regex='_sin$|_cos$').columns.tolist())
    # Columnas que empiezan con temp_ son seleccionadas
    exog_features.extend(variables_exogenas.filter(regex='^Temperature_.*').columns.tolist())
    # Columnas que empiezan con holiday_ son seleccionadas
    exog_features.extend(variables_exogenas.filter(regex='^Holiday_.*').columns.tolist())
    # Incluir temperatura y festivos
    exog_features.extend(['Temperature', 'Holiday'])
    
    # Combinar variables exógenas seleccionadas con la serie temporal
    datos_complete = datos[['Demand']].merge(
               variables_exogenas[exog_features],
               left_index  = True,
               right_index = True,
               how         = 'inner'
           )
    datos_complete = datos_complete.astype('float32')
    
    print(f"Variables exógenas creadas: {len(exog_features)}")
    print(f"Shape del dataset completo: {datos_complete.shape}")
    
    return datos_complete, exog_features

# ==============================================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ==============================================================================

def optimize_hyperparameters(datos, exog_features, fin_train, fin_validacion):
    """
    Optimiza los hiperparámetros del modelo usando búsqueda bayesiana
    """
    print("\n" + "=" * 60)
    print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
    print("=" * 60)
    
    # Crear forecaster base
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                     regressor       = LGBMRegressor(random_state=15926, verbose=-1),
                     lags            = 24,
                     window_features = window_features
                 )
    
    # Lags utilizados como predictores
    lags_grid = [24, [1, 2, 3, 23, 24, 25, 47, 48, 49]]
    
    # Espacio de búsqueda de hiperparámetros
    def search_space(trial):
        search_space  = {
            'n_estimators' : trial.suggest_int('n_estimators', 300, 1000, step=100),
            'max_depth'    : trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
            'reg_alpha'    : trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda'   : trial.suggest_float('reg_lambda', 0, 1),
            'lags'         : trial.suggest_categorical('lags', lags_grid)
        } 
        return search_space
    
    # Partición de entrenamiento y validación
    cv_search = TimeSeriesFold(
                    steps              = 24,
                    initial_train_size = len(datos[:fin_train]),
                    refit              = False,
                )
    
    print("Iniciando búsqueda bayesiana...")
    resultados_busqueda, frozen_trial = bayesian_search_forecaster(
                                            forecaster   = forecaster,
                                            y            = datos.loc[:fin_validacion, 'Demand'],
                                            exog         = datos.loc[:fin_validacion, exog_features],
                                            cv           = cv_search,
                                            metric       = 'mean_absolute_error',
                                            search_space = search_space,
                                            n_trials     = 10,
                                            return_best  = True
                                        )
    
    best_params = resultados_busqueda.at[0, 'params']
    best_params = best_params | {'random_state': 15926, 'verbose': -1}
    best_lags   = resultados_busqueda.at[0, 'lags']
    
    print(f"Mejores parámetros encontrados: {best_params}")
    print(f"Mejores lags: {best_lags}")
    print(f"Mejor MAE: {resultados_busqueda.at[0, 'mean_absolute_error']:.2f}")
    
    return forecaster, best_params, best_lags

# ==============================================================================
# SELECCIÓN DE PREDICTORES
# ==============================================================================

def select_predictors(datos, exog_features, best_params, best_lags, fin_train):
    """
    Realiza selección de predictores usando RFECV
    """
    print("\n" + "=" * 60)
    print("SELECCIÓN DE PREDICTORES")
    print("=" * 60)
    
    # Crear forecaster
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    regressor = LGBMRegressor(
                    n_estimators = 100,
                    max_depth    = 4,
                    random_state = 15926,
                    verbose      = -1
                )
    
    forecaster = ForecasterRecursive(
                     regressor       = regressor,
                     lags            = best_lags,
                     window_features = window_features
                 )
    
    # Eliminación recursiva de predictores con validación cruzada
    print("Realizando selección de predictores con RFECV...")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    selector = RFECV(
        estimator = regressor,
        step      = 1,
        cv        = 3,
    )
    
    lags_select, window_features_select, exog_select = select_features(
        forecaster      = forecaster,
        selector        = selector,
        y               = datos.loc[:fin_train, 'Demand'],
        exog            = datos.loc[:fin_train, exog_features],
        select_only     = None,
        force_inclusion = None,
        subsample       = 0.5,
        random_state    = 123,
        verbose         = True,
    )
    
    print(f"Predictores seleccionados: {len(exog_select)} variables exógenas")
    return lags_select, window_features_select, exog_select

# ==============================================================================
# FORECASTING PROBABILÍSTICO
# ==============================================================================

def probabilistic_forecasting(datos, exog_select, best_params, lags_select, fin_train, fin_validacion):
    """
    Implementa forecasting probabilístico con intervalos de confianza
    """
    print("\n" + "=" * 60)
    print("FORECASTING PROBABILÍSTICO")
    print("=" * 60)
    
    # Crear y entrenar el forecaster
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                     regressor       = LGBMRegressor(**best_params),
                     lags            = lags_select,
                     window_features = window_features,
                     binner_kwargs   = {"n_bins": 5}
                 )
    
    forecaster.fit(
        y    = datos.loc[:fin_train, 'Demand'],
        exog = datos.loc[:fin_train, exog_select],
        store_in_sample_residuals = True
    )
    
    # Predecir intervalos
    print("Generando predicciones con intervalos de confianza...")
    predicciones = forecaster.predict_interval(
                      exog     = datos.loc[fin_train:, exog_select],
                      steps    = 24,
                      interval = [5, 95],
                      method  = 'conformal'
                  )
    
    # Backtesting sobre los datos de validación para obtener los residuos out-sample
    print("Calculando residuos out-sample...")
    cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(datos.loc[:fin_train]),
        )
    _, predicciones_val = backtesting_forecaster(
                             forecaster = forecaster,
                             y          = datos.loc[:fin_validacion, 'Demand'],
                             exog       = datos.loc[:fin_validacion, exog_select],
                             cv         = cv,
                             metric     = 'mean_absolute_error'
                         )
    
    # Almacenar residuos out-sample en el forecaster
    forecaster.set_out_sample_residuals(
        y_true = datos.loc[predicciones_val.index, 'Demand'],
        y_pred = predicciones_val['pred']
    )
    
    # Backtest con intervalos de predicción para el conjunto de test
    cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(datos.loc[:fin_validacion]),
            refit              = False,
        )
    metrica, predicciones = backtesting_forecaster(
                                forecaster              = forecaster,
                                y                       = datos['Demand'],
                                exog                    = datos[exog_select],
                                cv                      = cv,
                                metric                  = 'mean_absolute_error',
                                interval                = [5, 95],
                                interval_method         = 'conformal',
                                use_in_sample_residuals = False,
                                use_binned_residuals    = True,
                            )
    
    # Cobertura del intervalo predicho
    cobertura = calculate_coverage(
                  y_true       = datos.loc[fin_validacion:, "Demand"],
                  lower_bound  = predicciones["lower_bound"],
                  upper_bound  = predicciones["upper_bound"]
                )
    area = (predicciones['upper_bound'] - predicciones['lower_bound']).sum()
    print(f"Área total del intervalo: {round(area, 2)}")
    print(f"Cobertura del intervalo predicho: {round(100 * cobertura, 2)} %")
    
    return predicciones, metrica

# ==============================================================================
# EXPLICABILIDAD DEL MODELO
# ==============================================================================

def model_explainability(datos, exog_select, best_params, lags_select, fin_validacion):
    """
    Implementa técnicas de explicabilidad del modelo
    """
    print("\n" + "=" * 60)
    print("EXPLICABILIDAD DEL MODELO")
    print("=" * 60)
    
    # Crear y entrenar el forecaster
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                     regressor       = LGBMRegressor(**best_params),
                     lags            = lags_select,
                     window_features = window_features
                 )
    
    forecaster.fit(
        y    = datos.loc[:fin_validacion, 'Demand'],
        exog = datos.loc[:fin_validacion, exog_select]
    )
    
    # Importancia de los predictores
    print("Calculando importancia de predictores...")
    feature_importances = forecaster.get_feature_importances()
    print("Top 10 predictores más importantes:")
    print(feature_importances.head(10))
    
    # SHAP values
    print("Calculando valores SHAP...")
    X_train, y_train = forecaster.create_train_X_y(
                           y    = datos.loc[:fin_validacion, 'Demand'],
                           exog = datos.loc[:fin_validacion, exog_select]
                       )
    
    # Crear SHAP explainer
    shap.initjs()
    explainer = shap.TreeExplainer(forecaster.regressor)
    
    # Seleccionar una muestra para acelerar el cálculo
    rng = np.random.default_rng(seed=785412)
    sample = rng.choice(X_train.index, size=int(len(X_train)*0.1), replace=False)
    X_train_sample = X_train.loc[sample, :]
    shap_values = explainer.shap_values(X_train_sample)
    
    # Shap summary plot
    print("Generando gráfico SHAP summary...")
    shap.summary_plot(shap_values, X_train_sample, max_display=10, show=False)
    fig, ax = plt.gcf(), plt.gca()
    ax.set_title("SHAP Summary plot")
    ax.tick_params(labelsize=8)
    fig.set_size_inches(6, 3.5)
    plt.show()
    
    return forecaster

# ==============================================================================
# FORECASTER DIRECT MULTI-STEP
# ==============================================================================

def direct_multi_step_forecasting(datos, exog_select, best_params, lags_select):
    """
    Implementa forecaster direct multi-step
    """
    print("\n" + "=" * 60)
    print("FORECASTER DIRECT MULTI-STEP")
    print("=" * 60)
    
    # Forecaster con el método direct
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterDirect(
                     regressor       = LGBMRegressor(**best_params),
                     steps           = 24,
                     lags            = lags_select,
                     window_features = window_features
                 )
    
    # Backtesting
    cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(datos.loc[:'2014-11-30 23:59:00']),
            refit              = False,
        )
    
    metrica, predicciones = backtesting_forecaster(
                              forecaster = forecaster,
                              y          = datos['Demand'],
                              exog       = datos[exog_select],
                              cv         = cv,
                              metric     = 'mean_absolute_error',
                          )
    
    print(f"MAE del modelo direct multi-step: {metrica.iloc[0, 0]:.2f}")
    return metrica.iloc[0, 0]

# ==============================================================================
# PREDICCIÓN DIARIA ANTICIPADA
# ==============================================================================

def daily_advance_prediction(datos, exog_select, best_params, lags_select, fin_validacion):
    """
    Implementa predicción diaria anticipada con gap
    """
    print("\n" + "=" * 60)
    print("PREDICCIÓN DIARIA ANTICIPADA")
    print("=" * 60)
    
    # Forecaster
    window_features = RollingFeatures(stats=["mean"], window_sizes=24 * 3)
    forecaster = ForecasterRecursive(
                     regressor       = LGBMRegressor(**best_params),
                     lags            = lags_select,
                     window_features = window_features
                 )
    
    # Backtesting con gap
    cv = TimeSeriesFold(
            steps              = 24,
            initial_train_size = len(datos.loc[:fin_validacion]) + 12,
            refit              = False,
            gap                = 12,
        )
    metrica, predicciones = backtesting_forecaster(
                                forecaster = forecaster,
                                y          = datos['Demand'],
                                exog       = datos[exog_select],
                                cv         = cv,
                                metric     = 'mean_absolute_error'
                            )
    
    print(f"MAE del modelo con predicción anticipada: {metrica.iloc[0, 0]:.2f}")
    return metrica.iloc[0, 0]

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def main():
    """
    Función principal que ejecuta todo el pipeline de forecasting
    """
    print("=" * 60)
    print("PREDICCIÓN DE DEMANDA ENERGÉTICA CON MACHINE LEARNING")
    print("=" * 60)
    
    # Imprimir versiones
    print_versions()
    
    # Cargar y procesar datos
    datos, datos_train, datos_val, datos_test, fin_train, fin_validacion = load_and_process_data()
    
    if datos is None:
        print("Error: No se pudieron cargar los datos. Verifica el archivo Excel.")
        return
    
    # Análisis exploratorio
    exploratory_analysis(datos, datos_train, datos_val, datos_test)
    
    # Modelo baseline
    mae_baseline = create_baseline_model(datos, fin_validacion)
    
    # Modelo autoregresivo recursivo
    forecaster_recursive, mae_recursive = create_recursive_model(datos, fin_validacion)
    
    # Crear variables exógenas
    datos_complete, exog_features = create_exogenous_variables(datos)
    
    # Actualizar particiones con datos completos
    datos_train_complete = datos_complete.loc[: fin_train, :].copy()
    datos_val_complete   = datos_complete.loc[fin_train:fin_validacion, :].copy()
    datos_test_complete  = datos_complete.loc[fin_validacion:, :].copy()
    
    # Optimización de hiperparámetros
    forecaster_optimized, best_params, best_lags = optimize_hyperparameters(
        datos_complete, exog_features, fin_train, fin_validacion
    )
    
    # Selección de predictores
    lags_select, window_features_select, exog_select = select_predictors(
        datos_complete, exog_features, best_params, best_lags, fin_train
    )
    
    # Forecasting probabilístico
    predicciones_prob, mae_prob = probabilistic_forecasting(
        datos_complete, exog_select, best_params, lags_select, fin_train, fin_validacion
    )
    
    # Explicabilidad del modelo
    forecaster_explainable = model_explainability(
        datos_complete, exog_select, best_params, lags_select, fin_validacion
    )
    
    # Forecaster direct multi-step
    mae_direct = direct_multi_step_forecasting(
        datos_complete, exog_select, best_params, lags_select
    )
    
    # Predicción diaria anticipada
    mae_advance = daily_advance_prediction(
        datos_complete, exog_select, best_params, lags_select, fin_validacion
    )
    
    # Resumen de resultados
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"MAE Baseline:                    {mae_baseline:.2f}")
    print(f"MAE Recursivo:                   {mae_recursive:.2f}")
    print(f"MAE Probabilístico:              {mae_prob.iloc[0, 0]:.2f}")
    print(f"MAE Direct Multi-step:           {mae_direct:.2f}")
    print(f"MAE Predicción Anticipada:       {mae_advance:.2f}")
    
    print("\n¡Pipeline de forecasting completado exitosamente!")

if __name__ == "__main__":
    main()
