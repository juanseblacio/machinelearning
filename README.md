# 🧠 Proyecto de Machine Learning: Casos Churn en Telecomunicaciones

## 🎯 Introducción

### El Problema
En un entorno altamente competitivo como el de las telecomunicaciones, retener a los clientes se ha convertido en una prioridad estratégica para las empresas.

Los efectos del fenómeno de abandono de clientes (CHURN en inglés) implican:
- Pérdida significativa de ingesos
- Inversiones constantes para captación de nuevos clientes

### La Solución
Usando técnicas de Machine Learning, se identifica patrones y comportamiento que predicen si un cliente puede abandonar o no el servicio. Esto permitirá implementar estrategias más efectivas y focalizadas para retener clientes.

---

## 🔎 Metodología

### Análisis Exploratorio
- Número de filas y columnas: La data cuenta con 7043 observaciones y un total de 50 columnas.
- Variables a eliminar:

| Column                   | Description                                                                 | Tipo      | Transformación                                | Observación                                      |
|--------------------------|-----------------------------------------------------------------------------|-----------|-----------------------------------------------|--------------------------------------------------|
| Gender                   | The customer’s gender: Male, Female                                           | Categórica| 1 Female 0 Male                               |                                                  |
| Age                      | The customer’s current age, in years, at the time the fiscal quarter ended. | Numérica  | StandardScaler                                | Distribución sin Outliers                        |
| Married                  | Indicates if the customer is married: Yes, No                                | Categórica| 1 Yes 0 No                                    |                                                  |
| Number of Dependents     | Indicates the number of dependents that live with the customer.              | Numérica  | Agrupar en 0, 1, 2, 3 y 4 o más              | De 4 en adelante la distribución cambia drásticamente |

### Procesamiento de Datos
- Imputación de valores nulos: [Sí/No, método utilizado].
- Encoding de variables categóricas: [Label Encoding, One-Hot Encoding, etc.]
- Feature Engineering: [Descripción breve de las nuevas variables creadas].

### Entrenamiento y Tuneo de Hiperparámetros
- Modelos probados: [Ej: Random Forest, XGBoost].
- Hiperparámetros optimizados: [Sí/No, método usado: GridSearchCV, RandomizedSearchCV].
- Métrica priorizada: [Ej: Recall, para minimizar falsos negativos].

### Interpretabilidad
- Método de interpretación: [SHAP, Permutation Importance, etc.]
- Principales hallazgos: [Ej: La variable "número de llamadas al servicio técnico" es altamente predictiva.]

---

## 📈 Resultados

### Rendimiento del Modelo

| Métrica | Valor |
|:--------|------:|
| Accuracy | [0.XX] |
| Precision | [0.XX] |
| Recall | [0.XX] |
| Specificity | [0.XX] |
| Negative Predictive Value | [0.XX] |
| F1 Score | [0.XX] |

- Matriz de Confusión:
  ![Matriz de Confusión](ruta/a/tu/imagen.png)

- Matriz en dólares:
  ![Matriz de Confusión Monetaria](ruta/a/tu/imagen.png)

### Importancia de Variables
- Gráfico de SHAP:
  ![SHAP Values](ruta/a/tu/imagen.png)
- Principales variables que impactan en la predicción:
  - [Variable 1]: Aumenta/disminuye la probabilidad.
  - [Variable 2]: Aumenta/disminuye la probabilidad.

---

## 🛠️ Implementación en el Negocio

[Describe cómo tu modelo se podría integrar a la operación: alertas tempranas, segmentación de clientes para retención, etc.]

---

## ⚠️ Limitaciones

- [Ej: No se consideraron variables de comportamiento en tiempo real.]
- [Ej: No se probó con modelos de Deep Learning.]

---

## 📝 Conclusiones y Recomendaciones

### Conclusiones
[¿Lograste tu objetivo? ¿Qué valor aporta el modelo?]

### Recomendaciones
- Para analistas: [Qué pruebas adicionales podrían hacer].
- Para el negocio: [Qué datos deberían recolectar para mejorar el modelo.]

---

## 🔮 Future Work

- [Probar otros algoritmos como LightGBM].
- [Agregar nuevas variables: uso de servicios premium, satisfacción del cliente.]
- [Automatizar el pipeline de scoring.]

---

## 📂 Archivos del Repositorio

- `proyecto.ipynb`: Notebook con el desarrollo completo.
- `slides/`: Capturas de las diapositivas usadas (opcional).
- `requirements.txt`: (opcional) Librerías necesarias para correr el proyecto.

---

## 📚 Tecnologías usadas

- Python
- Pandas
- Numpy
- Scikit-learn
- Matplotlib
- Seaborn
- SHAP

---

## 🚀 Cómo ejecutar el proyecto

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
