# Predicción de Demanda Energética con Machine Learning

Este proyecto implementa un sistema completo de forecasting para la demanda eléctrica utilizando modelos de machine learning.

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

## Archivos del Proyecto

Este proyecto incluye dos versiones del script principal:

1. **`energy_demand_forecasting.py`**: Script original que utiliza datos de Victoria, Australia (para demostración)
2. **`energy_demand_forecasting_xlsx.py`**: Script adaptado para trabajar con archivos Excel personalizados
3. **`forecasting_colab.ipynb`**: Notebook completo para Google Colab (recomendado)

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

4. Coloca tu archivo Excel con los datos de demanda energética en el directorio raíz del proyecto con el nombre `Demanda_Energia_SIN_2023.xlsx`

### Uso Local

**Para usar con datos de Excel personalizados (recomendado):**
```bash
python energy_demand_forecasting_xlsx.py
```

**Para usar con datos de demostración (Victoria, Australia):**
```bash
python energy_demand_forecasting.py
```

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

## Formato del Archivo Excel

El script `energy_demand_forecasting_xlsx.py` espera un archivo Excel con las siguientes características:

- **Columna de fecha**: Debe contener una de estas palabras clave en el nombre: 'fecha', 'date', 'time', 'tiempo'
- **Columna de demanda**: Debe contener una de estas palabras clave en el nombre: 'demanda', 'demand', 'consumo', 'consumption', 'energia', 'energy'
- **Formato de fecha**: Debe ser reconocible por pandas (ej: YYYY-MM-DD, DD/MM/YYYY, etc.)
- **Datos**: La columna de demanda debe contener valores numéricos
- **Header**: Los datos deben comenzar en la fila 4 (header en fila 3)

### Ejemplo de estructura esperada:

| Fecha | Demanda |
|-------|---------|
| 2023-01-01 00:00 | 1500.5 |
| 2023-01-01 01:00 | 1450.2 |
| 2023-01-01 02:00 | 1400.8 |
| ... | ... |

## Estructura del Proyecto

```
forecasting/
├── energy_demand_forecasting.py         # Script original (datos Victoria)
├── energy_demand_forecasting_xlsx.py    # Script para archivos Excel
├── forecasting_colab.ipynb             # Notebook completo para Colab
├── requirements.txt                     # Dependencias
├── .gitignore                          # Archivos ignorados por git
├── Demanda_Energia_SIN_2023.xlsx      # Archivo de datos Excel
└── README.md                           # Este archivo
```

## Metodología

1. **Carga y procesamiento**: Detección automática de columnas y conversión a datos horarios
2. **Exploración**: Análisis de estacionalidad y autocorrelación
3. **Baseline**: Modelo de referencia con valores del día anterior
4. **Modelo principal**: LightGBM con lags y variables exógenas
5. **Optimización**: Búsqueda bayesiana de hiperparámetros
6. **Selección**: RFECV para identificar predictores más relevantes
7. **Probabilístico**: Intervalos de confianza con conformal prediction
8. **Explicabilidad**: SHAP values para interpretar predicciones
9. **Multi-step**: Estrategias recursiva y directa
10. **Anticipada**: Predicciones con gap temporal

## Variables Exógenas Creadas

El script crea automáticamente las siguientes variables:

- **Calendario**: mes, semana, día de la semana, hora
- **Luz solar**: hora de salida/puesta del sol, horas de luz, período diurno (adaptado a Colombia)
- **Festivos**: festivos actuales, anteriores y siguientes
- **Temperatura**: promedios, máximos y mínimos (1 día y 7 días) - simulada si no está disponible
- **Interacciones**: combinaciones polinómicas entre variables

## Resultados

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
