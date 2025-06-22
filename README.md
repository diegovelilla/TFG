# Métodos de Generación de Datos Sintéticos Tabulares Basados en Inteligencia Artificial para Aumentación de Datos

## Descripción
Implementación de una librería de Python que permite la aumentación de datos para cualquier conjunto de datos con formato tabular. Los métodos de generación de los datos sintéticos serán basados en modelos de inteligencia artificial para conseguir resultados de estado del arte. 

## Requisitos previos
Este proyecto necesita del uso de una versión de Python >=3.10, <3.14.

## Instalación
Pasos a seguir para poder usar la librería.

### 1. Crear un entorno virtual (opcional pero recomendado)
```bash
python -m venv venv
```
Activa el entorno virtual:
- En Windows:
  ```bash
  venv\Scripts\activate
  ```
- En macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 2. Actualizar pip a la última versión
```bash
python -m pip install --upgrade pip
```

### 3. Instalar dependencias
Dependiendo de si pretendes usar GPU o no, instala las dependencias necesarias.

```bash
pip install -r TFG/requirements-gpu.txt

-------------------------------

pip install -r TFG/requirements-cpu.txt
```

Suele tardar un poco.

## Uso
Para probar la librería se puede usar el notebook `presentacion.ipynb`.

## Estructura del Proyecto
Explica brevemente la estructura del directorio del proyecto:
```
📂 TFG
├── 📂 CraftingTable
│   ├── 📂 models   
│   │   ├── 📂 tabddpm_utils      # Utilidades para el modelo TabDDPM
│   │   ├── 📂 tabsyn_utils       # Utilidades para el modelo TabSyn
│   │   ├── base.py               # Implementación del modelo base
│   │   ├── ctgan.py              # Implementación del modelo CTGAN
│   │   ├── tabddpm.py            # Implementación del modelo TabDDPM
│   │   ├── tabsyn.py             # Implementación del modelo TabSyn
│   │   └── tvae.py               # Implementación del modelo TVAE
│   │
│   └── utils.py                  # Utilidades para la librería.
│
├── requirements-gpu.txt          # Dependencias para la versión con uso de GPU.
├── requirements-cpu.txt          # Dependencias para la versión sin uso de GPU.
├── README.md                     # Documentación principal.
└── presentacion.ipynb            # Notebook con ejemplos de uso de la librería.