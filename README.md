# Análisis de Ventas de Chocolates

## 1. Introducción

Este proyecto realiza un **análisis exploratorio de datos (EDA)** sobre un conjunto de ventas de chocolates. Su objetivo es:

* Comprender patrones de venta por país y producto.
* Analizar la distribución de las ventas y cantidades enviadas.
* Visualizar resultados mediante gráficos estadísticos y comparativos.

El dataset contiene información de ventas de chocolates en varios países durante el año 2022.

---

## 2. Conjunto de Datos

### 2.1 Características

* Dataset: `chocolate_sales.csv`
* Columnas principales:

  * `Sales Person`: Nombre del vendedor
  * `Country`: País de venta
  * `Product`: Tipo de chocolate
  * `Date`: Fecha de venta
  * `Amount`: Monto de la venta (USD)
  * `Boxes Shipped`: Número de cajas enviadas
* Total de registros: 3282

```python
df = pd.read_csv('chocolate_sales.csv')
df.head(5)
```

---

## 3. Librerías

Se utilizan librerías estándar para análisis de datos y visualización:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", palette="rocket")
```

---

## 4. Limpieza de Datos

* Conversión de la columna `Date` a tipo datetime.
* Limpieza de la columna `Amount` (eliminar `$` y `,` y convertir a float).

```python
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df['Amount'] = df['Amount'].str.replace('$','', regex=False).str.replace(',','', regex=False).astype('float')
df.dtypes
```

---

## 5. Análisis Exploratorio

### 5.1 Estadísticas Básicas

```python
precio_promedio = np.mean(df['Amount'])
cajas_mediana = np.median(df['Boxes Shipped'])

print(f"Precio promedio: ${precio_promedio:,.2f}")
print(f"Mediana de cajas enviadas: {cajas_mediana}")
```

### 5.2 Ventas por País y Producto

```python
country_sales = df.groupby('Country')['Amount'].sum().sort_values(ascending=False)
top_products = df.groupby('Product')['Amount'].sum().sort_values(ascending=False)

print(country_sales.map('{:,.2f}'))
print(top_products.map('${:,.2f}'))
```

---

## 6. Visualización

### 6.1 Distribución de Ventas y Ventas por País (Matplotlib)

```python
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.hist(df['Amount'], bins=20, edgecolor='black')
plt.title('Distribución de Ventas')
plt.xlabel('Monto ($)')
plt.ylabel('Frecuencia')

plt.subplot(1, 2, 2)
bars = country_sales.plot(kind='bar', ax=plt.gca())
plt.title('Ventas Totales por País')
plt.xlabel('País')
plt.ylabel('Monto total ($)')
plt.xticks(rotation=45)

for i, v in enumerate(country_sales):
    plt.text(i, v + v*0.01, '{:,.0f}'.format(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()
```

### 6.2 Boxplot y Barras por País (Seaborn)

```python
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df, x='Country', y='Amount', ax=axs[0])
axs[0].set_title('Monto de Ventas por País')
axs[0].tick_params(axis='x', rotation=45)

amount_sum = df.groupby('Country')['Amount'].sum().sort_values(ascending=False)
countplot = sns.barplot(x=amount_sum.values, y=amount_sum.index, ax=axs[1], palette="rocket")
axs[1].set_title('Monto Total de Ventas por País')

for p in countplot.patches:
    width = p.get_width()
    axs[1].text(width + width*0.01, p.get_y() + p.get_height()/2., f'${width:,.0f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
```

### 6.3 Ventas vs Cajas Enviadas

```python
boxes_sum = df.groupby('Country')['Boxes Shipped'].sum().sort_values(ascending=False)

fig, axs = plt.subplots(1, 2, figsize=(15, 5))

sns.boxplot(data=df, x='Country', y='Amount', ax=axs[0])
axs[0].set_title('Monto de Ventas por País')
axs[0].tick_params(axis='x', rotation=45)

countplot = sns.barplot(x=boxes_sum.values, y=boxes_sum.index, ax=axs[1], palette="rocket")
axs[1].set_title('Total de Cajas Enviadas por País')

for p in countplot.patches:
    width = p.get_width()
    axs[1].text(width + width*0.01, p.get_y() + p.get_height()/2., f'{int(width):,}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## 7. Conclusiones

* **Patrones de ventas:** Australia y UK lideran ventas totales; productos destacados: `Smooth Sliky Salty` y `50% Dark Bites`.
* **Distribución de ventas:** La mayoría de ventas están entre $2,500 y $8,500; algunos valores extremos afectan los boxplots.
* **Cajas enviadas:** Existe correlación moderada entre ventas y cajas enviadas.

---

## 8. Mejoras Futuras

* Analizar ventas por **mes o trimestre** para detectar estacionalidad.
* Investigar relación entre **producto y país** más a fondo.
* Aplicar **modelos de regresión** para predicción de ventas futuras.
* Incorporar **visualizaciones interactivas** con Plotly o Dash.

---

## 9. Requisitos

* Python 3.8+
* Librerías:

```bash
pip install pandas numpy matplotlib seaborn
```
