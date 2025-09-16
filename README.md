# Predicción de Demanda Energética con Machine Learning

Este proyecto implementa un sistema completo de forecasting para la demanda eléctrica utilizando modelos de machine learning, enfocado en la predicción del día siguiente.

## Características Principales

- **Carga automática de múltiples archivos Excel** desde la carpeta `demanda-energia-sin/`
- **Análisis exploratorio completo** con visualizaciones interactivas y estadísticas descriptivas
- **Predicción del día siguiente** con 24 horas de anticipación
- **Modelo baseline** usando valores equivalentes de fechas anteriores
- **Modelo autoregresivo recursivo** con LightGBM optimizado
- **Variables exógenas** basadas en calendario, luz solar, festivos y temperatura
- **Optimización de hiperparámetros** con búsqueda bayesiana
- **Selección automática de predictores** con RFECV
- **Visualizaciones mejoradas** con gráficos de distribución, box plots por año y zoom temporal
- **Exportación de resultados** en formato CSV para uso posterior

## Archivos del Proyecto

Este proyecto incluye los siguientes archivos:

1. **`energy_demand_forecasting.py`**: Script original que utiliza datos de Victoria, Australia (para demostración)
2. **`energy_demand_forecasting_xlsx.py`**: Script principal para predicción del día siguiente con archivos Excel
3. **`forecasting_demanda_energia.ipynb`**: **NUEVO** - Notebook de forecasting recursivo multi-step con ForecasterRecursive
4. **`forecasting_colab.ipynb`**: Notebook completo para Google Colab (recomendado)
5. **`demanda-energia-sin/`**: Carpeta con archivos Excel de demanda energética por año (2000-2025)
6. **`requirements.txt`**: Dependencias del proyecto

## Instalación

### Instalación Local

1. Clona el repositorio:
```bash
git clone <url-del-repositorio>
cd forecasting
```

2. Crea un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Los archivos Excel de demanda energética ya están incluidos en la carpeta `demanda-energia-sin/`

### Uso Local

**Para forecasting recursivo multi-step (NUEVO - Recomendado):**
```bash
jupyter notebook forecasting_demanda_energia.ipynb
```

**Para predicción del día siguiente (script):**
```bash
python energy_demand_forecasting_xlsx.py
```

**Para usar con datos de demostración (Victoria, Australia):**
```bash
python energy_demand_forecasting.py
```

### Resultados

**Notebook de Forecasting Recursivo (`forecasting_demanda_energia.ipynb`):**
- **Análisis exploratorio completo** con gráficos interactivos (Plotly + Matplotlib)
- **Partición de datos** train/validation/test (70/15/15)
- **Modelo recursivo multi-step** con ForecasterRecursive y LightGBM
- **Gráficos de dispersión** para evaluar real vs predicho
- **Series temporales** con predicciones superpuestas
- **Predicción del día siguiente** con 24 horas de anticipación
- **Archivo CSV** con las predicciones (`prediccion_demanda_YYYYMMDD.csv`)
- **Métricas de rendimiento** (MAE, RMSE, R²)

**Scripts Python:**
- **Análisis exploratorio completo** con gráficos interactivos
- **Predicción horaria** para las 24 horas del día siguiente
- **Archivo CSV** con las predicciones (`prediccion_demanda_YYYYMMDD.csv`)
- **Métricas de rendimiento** del modelo

## Uso en Google Colab (Recomendado)

### Opción 1: Usar el Notebook Completo

1. **Descarga el archivo `forecasting_colab.ipynb`** desde este repositorio
2. **Sube el notebook a Google Colab**: Ve a [colab.research.google.com](https://colab.research.google.com) y selecciona "Subir notebook"
3. **Ejecuta las celdas** en orden - el notebook está completamente autocontenido e incluye:
   - Instalación automática de dependencias
   - Interfaz para subir tu archivo Excel
   - Todo el código necesario integrado
   - Visualizaciones interactivas

### Opción 2: Desde GitHub

1. **Abre Google Colab**: Ve a [colab.research.google.com](https://colab.research.google.com)

2. **Crea un nuevo notebook** y ejecuta las siguientes celdas:

```python
# Instalar dependencias
!pip install skforecast lightgbm feature-engine astral shap plotly

# Clonar el repositorio
!git clone https://github.com/tu-usuario/forecasting.git
%cd forecasting

# Subir tu archivo Excel
from google.colab import files
uploaded = files.upload()

# Renombrar el archivo si es necesario
import os
if 'Demanda_Energia_SIN_2023.xlsx' not in os.listdir():
    for filename in uploaded.keys():
        if filename.endswith('.xlsx'):
            os.rename(filename, 'Demanda_Energia_SIN_2023.xlsx')
            break

# Ejecutar el script para Excel
!python energy_demand_forecasting_xlsx.py
```

### Opción 3: Usar Google Drive

1. **Sube tu archivo Excel** a Google Drive con el nombre `Demanda_Energia_SIN_2023.xlsx`

2. **Monta Google Drive** en Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copia el archivo al directorio de trabajo
!cp "/content/drive/MyDrive/Demanda_Energia_SIN_2023.xlsx" "/content/"

# Clonar el repositorio
!git clone https://github.com/tu-usuario/forecasting.git
%cd forecasting

# Ejecutar el script
!python energy_demand_forecasting_xlsx.py
```

## Datos Incluidos

El proyecto incluye datos históricos de demanda energética del Sistema Interconectado Nacional (SIN) de Colombia:

- **Período**: 2000-2025 (26 años de datos)
- **Frecuencia**: Datos diarios convertidos automáticamente a horarios
- **Archivos**: 25 archivos Excel organizados por año
- **Formato**: Cada archivo contiene columnas de fecha y demanda energética

### Estructura de Datos

Los archivos Excel siguen el formato estándar:
- **Columna de fecha**: Detectada automáticamente por palabras clave
- **Columna de demanda**: Detectada automáticamente por palabras clave  
- **Header**: Los datos comienzan en la fila 4 (header en fila 3)
- **Conversión automática**: Datos diarios se convierten a horarios simulando patrones típicos

### Ejemplo de datos incluidos:

| Año | Archivo | Observaciones |
|-----|---------|---------------|
| 2000 | Demanda_Energia_SIN_2000.xlsx | ~119 días |
| 2001 | Demanda_Energia_SIN_2001.xlsx | ~108 días |
| ... | ... | ... |
| 2023 | Demanda_Energia_SIN_2023.xlsx | ~142 días |
| 2024 | Demanda_Energia_SIN_2024.xlsx | ~115 días |
| 2025 | Demanda_Energia_SIN_2025.xlsx | ~93 días |

## Estructura del Proyecto

```
forecasting/
├── energy_demand_forecasting.py         # Script original (datos Victoria)
├── energy_demand_forecasting_xlsx.py    # Script principal para predicción
├── forecasting_demanda_energia.ipynb   # NUEVO: Notebook forecasting recursivo
├── forecasting_colab.ipynb             # Notebook completo para Colab
├── requirements.txt                     # Dependencias
├── .gitignore                          # Archivos ignorados por git
├── demanda-energia-sin/                # Carpeta con datos históricos
│   ├── Demanda_Energia_SIN_2000.xlsx
│   ├── Demanda_Energia_SIN_2001.xlsx
│   ├── ...
│   ├── Demanda_Energia_SIN_2023.xlsx
│   ├── Demanda_Energia_SIN_2024.xlsx
│   └── Demanda_Energia_SIN_2025.xlsx
├── prediccion_demanda_YYYYMMDD.csv     # Archivo de salida con predicciones
└── README.md                           # Este archivo
```

## Metodología

### Notebook de Forecasting Recursivo (`forecasting_demanda_energia.ipynb`)

**Método**: Forecasting Recursivo Multi-Step con `ForecasterRecursive`

1. **Carga automática**: Lectura de múltiples archivos Excel desde `demanda-energia-sin/`
2. **Análisis exploratorio**: Estadísticas descriptivas, visualizaciones interactivas con Plotly
3. **Procesamiento**: Conversión de datos diarios a horarios con patrones simulados
4. **Partición temporal**: 70% entrenamiento, 15% validación, 15% prueba
5. **Modelo recursivo**: ForecasterRecursive con LightGBM
   - **Lags**: 24 horas (último día completo)
   - **Features de ventana**: Media y desviación estándar de 24h y 7 días
6. **Evaluación**: Métricas MAE, RMSE, R² en validación y prueba
7. **Visualizaciones**: Gráficos de dispersión (real vs predicho) y series temporales
8. **Predicción del día siguiente**: 24 horas de anticipación con visualizaciones
9. **Exportación**: Resultados guardados en CSV para uso posterior

### Scripts Python (Metodología Original)

1. **Carga automática**: Lectura de múltiples archivos Excel desde `demanda-energia-sin/`
2. **Análisis exploratorio**: Estadísticas descriptivas, visualizaciones y detección de patrones
3. **Procesamiento**: Conversión de datos diarios a horarios con patrones simulados
4. **Partición temporal**: 70% entrenamiento, 15% validación, 15% prueba
5. **Modelo baseline**: Valores del día anterior como referencia
6. **Variables exógenas**: Calendario, luz solar, festivos y temperatura
7. **Optimización**: Búsqueda bayesiana de hiperparámetros de LightGBM
8. **Selección de predictores**: RFECV para identificar variables más relevantes
9. **Predicción del día siguiente**: 24 horas de anticipación con visualizaciones
10. **Exportación**: Resultados guardados en CSV para uso posterior

## Características del Forecasting Recursivo

### ¿Qué es el Forecasting Recursivo Multi-Step?

El **forecasting recursivo multi-step** es un método que utiliza las propias predicciones del modelo como valores de entrada para predecir el siguiente valor. Por ejemplo, para predecir las 5 horas siguientes:

1. El modelo predice t+1 usando datos históricos
2. Usa esa predicción para predecir t+2
3. Usa t+2 para predecir t+3, y así sucesivamente

### Ventajas del Método Recursivo

- **Captura dependencias temporales complejas**: El modelo aprende patrones de largo plazo
- **Automatización completa**: La clase `ForecasterRecursive` maneja el proceso recursivo
- **Eficiencia computacional**: Un solo modelo para múltiples pasos
- **Ideal para series temporales**: Especialmente efectivo para demanda energética

### Implementación en el Notebook

```python
# Configuración del modelo recursivo
forecaster = ForecasterRecursive(
    regressor=LGBMRegressor(n_estimators=100, max_depth=6),
    lags=24,  # Usar las últimas 24 horas
    window_features=RollingFeatures(
        stats=["mean", "std"], 
        window_sizes=[24, 24*7]  # 24 horas y 7 días
    )
)
```

## Variables Exógenas Creadas

El script crea automáticamente las siguientes variables:

- **Calendario**: mes, semana, día de la semana, hora
- **Luz solar**: hora de salida/puesta del sol, horas de luz, período diurno (adaptado a Colombia)
- **Festivos**: festivos actuales, anteriores y siguientes
- **Temperatura**: promedios, máximos y mínimos (1 día y 7 días) - simulada si no está disponible
- **Interacciones**: combinaciones polinómicas entre variables

## Resultados

### Notebook de Forecasting Recursivo

El notebook genera:

- **Métricas de evaluación**: MAE, RMSE, R² para validación y prueba
- **Gráficos de dispersión**: Comparación visual entre valores reales y predichos
- **Series temporales**: Visualización de predicciones superpuestas en datos históricos
- **Predicción del día siguiente**: 24 horas con estadísticas detalladas (total, promedio, máximo, mínimo)
- **Archivos de salida**: 
  - `prediccion_demanda_YYYYMMDD.csv` - Predicciones horarias
  - `metricas_modelo.csv` - Métricas de rendimiento

### Scripts Python

El pipeline genera métricas de error (MAE) para cada modelo implementado, permitiendo comparar el rendimiento de diferentes enfoques.

## Solución de Problemas

### Error: "No se encontró el archivo Excel"
- Verifica que el archivo esté en el directorio correcto
- Asegúrate de que el nombre sea exactamente `Demanda_Energia_SIN_2023.xlsx`

### Error: "No se detectó columna de fecha/demanda"
- Verifica que los nombres de las columnas contengan las palabras clave mencionadas
- Revisa que el archivo Excel tenga al menos una columna de fecha y una de demanda
- Asegúrate de que el header esté en la fila 3 (datos desde fila 4)

### Error de memoria en Google Colab
- Usa `Runtime > Restart runtime` si el notebook se queda sin memoria
- Considera usar un subconjunto de datos más pequeño para pruebas

### Problemas con datos diarios
- El script automáticamente convierte datos diarios a horarios simulando patrones típicos
- Si tienes datos horarios reales, asegúrate de que la frecuencia sea correcta

## Referencias

- Amat Rodrigo, J., & Escobar Ortiz, J. (2025). Predicción (forecasting) de la demanda energética con machine learning. https://www.cienciadedatos.net/documentos/py29-forecasting-demanda-energia-electrica-python.html
- skforecast: https://skforecast.org/

## Licencia

Disponible bajo licencia CC BY-NC-SA 4.0.
