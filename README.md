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
- Variables a analizar:

| Column                        | Descripción                                                                                                                       | Tipo        | Transformación                          | Observación                                      |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|-------------|-----------------------------------------|--------------------------------------------------|
| Gender                         | El género del cliente: Masculino, Femenino                                                                                         | Categórica  | 1 Female 0 Male                         |                                                  |
| Age                            | La edad actual del cliente, en años, al final del trimestre fiscal.                                                               | Numérica    | StandardScaler                          | Distribución sin Outliers                        |
| Married                        | Indica si el cliente está casado: Sí, No                                                                                          | Categórica  | 1 Yes 0 No                              |                                                  |
| Number of Dependents           | Indica el número de dependientes que vive con el cliente.                                                                         | Numérica    | Agrupar en 0, 1, 2, 3 y 4 o más        | De 4 en adelante la distribución cambia drásticamente |
| Number of Referrals            | Indica el número de referencias que el cliente ha realizado hasta la fecha.                                                        | Numérica    | RobustScaler                            | Distribución con Outliers                        |
| Tenure in Months               | Indica la cantidad total de meses que el cliente ha estado con la empresa hasta el final del trimestre especificado.               | Numérica    | StandardScaler                          | Distribución sin Outliers                        |
| Offer                          | Identifica la última oferta de marketing que el cliente aceptó, si es aplicable: Ninguna, Oferta A, Oferta B, Oferta C, Oferta D, y Oferta E. | Categórica  | OneHotEncoder                           | Varias categorías sin Orden                      |
| Phone Service                  | Indica si el cliente está suscrito al servicio telefónico en casa con la empresa: Sí, No                                           | Categórica  | 1 Yes 0 No                              |                                                  |
| Multiple Lines                 | Indica si el cliente está suscrito a múltiples líneas telefónicas con la empresa: Sí, No                                            | Categórica  | 1 Yes 0 No                              |                                                  |
| Internet Type                  |                                                                                                                                    | Categórica  | OneHotEncoder                           | Varias categorías sin Orden                      |
| Avg Monthly GB Download        | Indica el volumen promedio de descarga en gigabytes del cliente, calculado hasta el final del trimestre especificado.               | Numérica    | RobustScaler                            | Distribución con Outliers                        |
| Online Security                | Indica si el cliente está suscrito a un servicio adicional de seguridad en línea proporcionado por la empresa: Sí, No              | Categórica  | 1 Yes 0 No                              |                                                  |
| Online Backup                  | Indica si el cliente está suscrito a un servicio adicional de respaldo en línea proporcionado por la empresa: Sí, No                | Categórica  | 1 Yes 0 No                              |                                                  |
| Device Protection Plan         | Indica si el cliente está suscrito a un plan de protección de dispositivos adicional para su equipo de Internet proporcionado por la empresa: Sí, No | Categórica  | 1 Yes 0 No                              |                                                  |
| Premium Tech Support           | Indica si el cliente está suscrito a un plan de soporte técnico adicional de la empresa con tiempos de espera reducidos: Sí, No   | Categórica  | 1 Yes 0 No                              |                                                  |
| Streaming TV                   | Indica si el cliente utiliza su servicio de Internet para transmitir programación televisiva de un proveedor externo: Sí, No       | Categórica  | 1 Yes 0 No                              |                                                  |
| Streaming Movies               | Indica si el cliente utiliza su servicio de Internet para transmitir películas de un proveedor externo: Sí, No                    | Categórica  | 1 Yes 0 No                              |                                                  |
| Streaming Music                | Indica si el cliente utiliza su servicio de Internet para transmitir música de un proveedor externo: Sí, No                       | Categórica  | 1 Yes 0 No                              |                                                  |
| Unlimited Data                 | Indica si el cliente ha pagado una tarifa mensual adicional para tener descargas/cargas de datos ilimitados: Sí, No               | Categórica  | 1 Yes 0 No                              |                                                  |
| Contract                       | Indica el tipo de contrato actual del cliente: Mes a Mes, Un Año, Dos Años.                                                        | Categórica  | OneHotEncoder                           | Varias categorías sin Orden                      |
| Payment Method                 | Indica cómo el cliente paga su factura: Retiro Bancario, Tarjeta de Crédito, Cheque por Correo                                  | Categórica  | OneHotEncoder                           | Varias categorías sin Orden                      |
| Monthly Charge                 | Indica el cargo mensual total actual del cliente por todos sus servicios con la empresa.                                          | Numérica    | StandardScaler                          | Distribución sin Outliers                        |
| Total Charges                  | Indica el total de cargos del cliente, calculado hasta el final del trimestre especificado.                                         | Numérica    | StandardScaler                          | Distribución sin Outliers                        |
| Total Refunds                  | Indica el total de reembolsos del cliente, calculado hasta el final del trimestre especificado.                                      | Numérica    | RobustScaler                            | Distribución con Outliers                        |
| Total Extra Data Charges       | Indica el total de cargos del cliente por descargas de datos extra por encima de los especificados en su plan, hasta el final del trimestre especificado. | Numérica    | RobustScaler                            | Distribución con Outliers                        |
| Total Long Distance Charges    | Indica el total de cargos del cliente por llamadas de larga distancia por encima de los especificados en su plan, hasta el final del trimestre especificado. | Numérica    | RobustScaler                            | Distribución con Outliers                        |
| Total Revenue                  | Se suma el ingreso de todos los servicios                                                                                           | Numérica    | RobustScaler                            | Distribución con Outliers                        |
| CLTV                           | Valor del Cliente Durante Toda su Vida. Un CLTV predicho se calcula usando fórmulas corporativas y datos existentes. Cuanto más alto sea el valor, más valioso es el cliente. | Numérica    | StandardScaler                          | Distribución sin Outliers                        |
| Churn Value                    | 1 = el cliente dejó la empresa este trimestre. 0 = el cliente permaneció con la empresa. Relacionado directamente con la etiqueta de Churn. | Variable Respuesta |                                         |                                                  |


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
