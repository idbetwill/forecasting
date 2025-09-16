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
3. **`forecasting_colab.ipynb`**: Notebook completo para Google Colab (recomendado)
4. **`demanda-energia-sin/`**: Carpeta con archivos Excel de demanda energética por año (2000-2023)
5. **`requirements.txt`**: Dependencias del proyecto

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

**Para predicción del día siguiente (recomendado):**
```bash
python energy_demand_forecasting_xlsx.py
```

**Para usar con datos de demostración (Victoria, Australia):**
```bash
python energy_demand_forecasting.py
```

### Resultados

El script generará:
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

- **Período**: 2000-2023 (24 años de datos)
- **Frecuencia**: Datos diarios convertidos automáticamente a horarios
- **Archivos**: 24 archivos Excel organizados por año
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

## Estructura del Proyecto

```
forecasting/
├── energy_demand_forecasting.py         # Script original (datos Victoria)
├── energy_demand_forecasting_xlsx.py    # Script principal para predicción
├── forecasting_colab.ipynb             # Notebook completo para Colab
├── requirements.txt                     # Dependencias
├── .gitignore                          # Archivos ignorados por git
├── demanda-energia-sin/                # Carpeta con datos históricos
│   ├── Demanda_Energia_SIN_2000.xlsx
│   ├── Demanda_Energia_SIN_2001.xlsx
│   ├── ...
│   └── Demanda_Energia_SIN_2023.xlsx
├── prediccion_demanda_YYYYMMDD.csv     # Archivo de salida con predicciones
└── README.md                           # Este archivo
```

## Metodología

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
