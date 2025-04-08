# Métodos de Generación de Datos Sintéticos Tabulares Basados en Inteligencia Artificial para Aumentación de Datos

## Descripción
Implementación de una librería de Python que permita la aumentación de datos para cualquier conjunto de datos con formato tabular. Los métodos de generación de los datos sintéticos serán basados en modelos de inteligencia artificial para conseguir resultados de estado del arte. Por el momento, la librería lleva el nombre de CraftingTable, ya que sirve para, en otras palabras, crear tablas.

## Requisitos previos
Antes de instalar y ejecutar el proyecto, asegúrate de tener instalado lo siguiente:

- Python 3.10.5

## Instalación
Sigue estos pasos para configurar el entorno y ejecutar el proyecto correctamente.

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
Asegúrate de estar en la raíz del proyecto y ejecuta:
```bash
pip install -r requirements.txt
```

Suele tardar un poco.

## Uso
Para probar la librería se puede usar el `notebook_test.ipynb`.

## Estructura del Proyecto
Explica brevemente la estructura del directorio del proyecto:
```
📂 TFG_v1
├── 📂 CraftingTable
│   ├── CraftingTable.py        # Implementación de la librería.
│   ├── ctgan.py                # Reimplementación clase CTGAN.
│   └── tvae.py                 # Reimplementación clase TVAE.
├── model_ctgan_pretrained.pt   # Modelo preentrenado de CTGAN.
├── model_tvae_pretrained.pt    # Modelo preentrenado de TVAE.
├── requirements.txt            # Dependencias.
├── README.md                   # Documentación principal.
└── notebook_test.ipynb         # Notebook para probar la librería.