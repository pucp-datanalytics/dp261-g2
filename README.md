# Justificación económica de las métricas de evaluación

Para este proyecto se propone utilizar el **recall como métrica principal**, el **F1-score como métrica secundaria** y el **AUC-ROC como tercera métrica de evaluación** para la clase de cancelación (`is_canceled = 1`).

La elección de estas métricas se basa en el impacto económico de los errores del modelo. En el negocio hotelero, el error más costoso es no detectar una reserva que realmente se cancelará. Este caso corresponde a un **falso negativo**: la reserva sí se cancela, pero el modelo predice que no se cancelará.

Cuando ocurre un falso negativo, el hotel pierde la oportunidad de aplicar acciones preventivas como confirmaciones anticipadas, políticas de depósito, campañas de retención, descuentos dirigidos u overbooking controlado.

El **F1-score** se utiliza como segunda métrica porque permite controlar el equilibrio entre detectar cancelaciones reales y no generar demasiadas falsas alertas.

El **AUC-ROC** se incorpora como tercera métrica porque permite validar si el modelo separa adecuadamente las reservas canceladas y no canceladas antes de ajustar el umbral de decisión.

---

## Información base

### Variable objetivo: `is_canceled`

| Indicador | Valor |
|---|---:|
| Count | 119,390 |
| Mean | 0.37 |
| Std | 0.48 |
| Min | 0.00 |
| 25% | 0.00 |
| 50% | 0.00 |
| 75% | 1.00 |
| Max | 1.00 |

Como `is_canceled` es una variable binaria, su media representa la proporción de reservas canceladas. Por lo tanto, aproximadamente el **37% de las reservas fueron canceladas**.

Esto confirma que las cancelaciones representan una proporción relevante del negocio y justifican el desarrollo de un modelo predictivo orientado a anticiparlas.

### Variable económica: `ADR`

La variable `ADR` representa el ingreso promedio diario por habitación ocupada.

| Indicador | Valor |
|---|---:|
| Count | 119,390 |
| Mean | 101.83 |
| Std | 50.54 |
| Min | -6.38 |
| 25% | 69.29 |
| 50% | 94.58 |
| 75% | 126.00 |
| Max | 5,400.00 |

El **ADR promedio** es **101.83**, por lo que una cancelación real no detectada puede representar una pérdida potencial cercana a **101.83 por noche**.

---

## Supuestos económicos usados

Para hacer el análisis más interpretable, se utiliza un escenario de **100,000 reservas evaluadas**.

| Concepto | Valor |
|---|---:|
| Reservas evaluadas | 100,000 |
| Proporción de cancelaciones | 37% |
| Reservas canceladas estimadas | 37,000 |
| Reservas no canceladas estimadas | 63,000 |
| ADR promedio | 101.83 |
| Costo estimado por falsa alerta | 5.00 |

El costo de una falsa alerta se estima en **5.00 unidades monetarias** por reserva intervenida innecesariamente. Este supuesto representa costos operativos asociados a acciones como comunicación adicional, revisión administrativa, seguimiento comercial breve y registro de la alerta.

| Acción operativa innecesaria | Costo estimado |
|---|---:|
| Envío de comunicación adicional | 0.50 |
| Tiempo administrativo de revisión | 1.00 |
| Seguimiento comercial o llamada breve | 2.00 |
| Registro y monitoreo de la alerta | 0.50 |
| Posible incentivo menor o gestión adicional | 1.00 |
| **Costo total estimado por falsa alerta** | **5.00** |

Las fórmulas usadas para estimar el impacto económico son:

```text
Pérdida por falsos negativos = Falsos negativos × ADR promedio
```

```text
Costo por falsas alertas = Falsos positivos × costo promedio de intervención
```

Estos valores son supuestos simplificados, pero permiten comparar órdenes de magnitud. El punto central es que una falsa alerta genera un costo operativo, mientras que una cancelación real no detectada puede representar pérdida de ingreso asociada al ADR.

---

## Recall como métrica principal

El **recall** mide qué porcentaje de cancelaciones reales logra detectar el modelo. Se prioriza porque el error más costoso para el hotel es el **falso negativo**: una reserva que sí se cancela, pero el modelo no la detecta.

Con base en el supuesto de 100,000 reservas evaluadas y una tasa de cancelación de 37%, se estiman **37,000 reservas canceladas reales**.

| Escenario | Recall | Cancelaciones detectadas | Cancelaciones no detectadas | Pérdida potencial por noche |
|---|---:|---:|---:|---:|
| Bajo desempeño | 60% | 22,200 | 14,800 | 1,507,084 |
| Alto desempeño | 90% | 33,300 | 3,700 | 376,771 |

Pasar de un recall de **60% a 90%** permitiría detectar **11,100 cancelaciones adicionales** y reducir la pérdida potencial por noche en aproximadamente **1,130,313**.

| Mejora económica | Valor aproximado |
|---|---:|
| Cancelaciones adicionales detectadas | 11,100 |
| Reducción de cancelaciones no detectadas | 11,100 |
| Reducción de pérdida potencial | 1,130,313 |

Por este impacto económico, el **recall** se define como la métrica principal del proyecto.

---

## F1-score como métrica secundaria

El **F1-score** combina recall y precision. Se usa como métrica secundaria porque permite balancear la detección de cancelaciones con el control de falsas alertas.

Un modelo podría aumentar el recall marcando muchas reservas como riesgosas, pero si eso genera demasiadas falsas alertas, también se incrementan los costos operativos. Por eso, el F1-score ayuda a controlar el equilibrio general del modelo.

Para simplificar el análisis, se comparan dos escenarios donde precision y recall tienen el mismo valor.

| Escenario | Precision | Recall | F1-score | Cancelaciones no detectadas | Falsas alertas estimadas | Impacto económico total |
|---|---:|---:|---:|---:|---:|---:|
| Bajo desempeño | 60% | 60% | 60% | 14,800 | 14,800 | 1,581,084 |
| Alto desempeño | 90% | 90% | 90% | 3,700 | 3,700 | 395,271 |

El impacto económico total se calcula como:

```text
Impacto económico total =
Pérdida por cancelaciones no detectadas + Costo por falsas alertas
```

Pasar de un F1-score de **60% a 90%** reduciría el impacto económico total estimado en aproximadamente **1,185,813**.

| Mejora económica | Valor aproximado |
|---|---:|
| Reducción de cancelaciones no detectadas | 11,100 |
| Reducción de falsas alertas estimadas | 11,100 |
| Reducción total estimada | 1,185,813 |

El **F1-score** no reemplaza al recall, pero ayuda a evitar que el modelo mejore la detección generando demasiadas alertas innecesarias.

---

## AUC-ROC como tercera métrica

El **AUC-ROC** evalúa la capacidad general del modelo para separar reservas canceladas y no canceladas en distintos umbrales de clasificación.

A diferencia del recall y el F1-score, que dependen de un umbral específico, el AUC-ROC permite validar si el modelo tiene buena capacidad de discriminación antes de definir el umbral operativo final.

| Métrica | Qué evalúa | Relación con el negocio |
|---|---|---|
| Recall | Cancelaciones reales detectadas con un umbral definido | Reduce falsos negativos |
| F1-score | Balance entre recall y precision con un umbral definido | Controla detección y falsas alertas |
| AUC-ROC | Separación general entre clases en distintos umbrales | Valida si el modelo discrimina bien antes de ajustar umbrales |

El AUC-ROC se ubica como tercera métrica porque es útil para validar la calidad general del modelo, pero no mide directamente el costo económico de los falsos negativos. Un modelo puede tener buen AUC-ROC, pero si se elige un umbral inadecuado, podría tener un recall insuficiente para el objetivo económico del proyecto.

Por eso, el AUC-ROC se considera una métrica importante de validación general, pero se ubica detrás del recall y del F1-score en la priorización del proyecto.

---

## Precision como métrica de apoyo

La **precision** mide qué proporción de las reservas predichas como canceladas realmente se cancelan. Una precision baja genera falsas alertas y costos operativos.

Estas falsas alertas no implican necesariamente pérdida de ADR, pero sí generan costos porque el hotel podría intervenir reservas que realmente no se iban a cancelar.

Para analizar el impacto de precision, se mantiene un supuesto de **recall constante de 80%**. Esto permite aislar el efecto de precision sobre los falsos positivos.

Con 37,000 cancelaciones reales:

```text
Cancelaciones detectadas = 37,000 × 80% = 29,600
```

| Escenario | Precision | Cancelaciones detectadas | Reservas marcadas como riesgo | Falsas alertas | Costo operativo total |
|---|---:|---:|---:|---:|---:|
| Baja precision | 60% | 29,600 | 49,333 | 19,733 | 98,665 |
| Alta precision | 90% | 29,600 | 32,889 | 3,289 | 16,445 |

Pasar de una precision de **60% a 90%** reduciría las falsas alertas en aproximadamente **16,444 casos** y disminuiría el costo operativo estimado en **82,220**.

| Mejora económica | Valor aproximado |
|---|---:|
| Reducción de falsas alertas | 16,444 |
| Reducción de costo operativo | 82,220 |

La precision es importante para la eficiencia operativa. Sin embargo, bajo estos supuestos, su impacto económico estimado es menor que el del recall.

Mientras el recall puede reducir pérdidas potenciales por más de **1.13 millones**, la precision reduce costos operativos por aproximadamente **82 mil**. Por eso, precision se considera una métrica de apoyo, pero no se elige como métrica principal ni secundaria.

---

## Accuracy como métrica complementaria

La **accuracy** mide el porcentaje total de predicciones correctas. Sin embargo, no se considera una métrica prioritaria para este problema porque no distingue entre tipos de error.

En este proyecto, no todos los errores tienen el mismo costo económico. Un falso negativo puede implicar una pérdida de ingreso asociada al ADR, mientras que un falso positivo suele representar un costo operativo menor.

Por esta razón, la accuracy puede utilizarse como referencia general, pero no debe ser la métrica principal de evaluación.

---

## Resumen de priorización de métricas

| Prioridad | Métrica | Rol | Qué controla | Justificación económica |
|---:|---|---|---|---|
| 1 | Recall | Principal | Falsos negativos | Reduce pérdidas por cancelaciones reales no detectadas |
| 2 | F1-score | Secundaria | Balance entre falsos negativos y falsos positivos | Controla el equilibrio entre pérdida económica e intervención innecesaria |
| 3 | AUC-ROC | Tercera métrica | Capacidad general de separar canceladas y no canceladas | Valida si el modelo discrimina bien antes de ajustar umbrales |
| 4 | Precision | Apoyo | Falsos positivos | Reduce costos por intervenciones innecesarias |
| 5 | Accuracy | Complementaria | Acierto general | No distingue si el error tiene alto o bajo costo económico |

---

## Criterios de éxito del modelo

Los criterios de éxito propuestos para el modelo son:

```text
- Alcanzar un recall ≥ 80% en la clase de cancelación.
- Alcanzar un F1-score ≥ 75% en la clase de cancelación.
- Lograr un AUC-ROC ≥ 0.75.
- Mantener una precision razonable para evitar exceso de falsas alertas.
- Superar el desempeño de un modelo base o una regla simple de negocio.
```

---

## Conclusión

El **recall** se elige como métrica principal porque el mayor costo económico es no detectar reservas que realmente se cancelarán. Con una tasa de cancelación aproximada de **37%** y un ADR promedio de **101.83**, cada falso negativo puede representar una pérdida económica relevante.

El **F1-score** se usa como métrica secundaria porque controla el balance entre detección de cancelaciones y falsas alertas.

El **AUC-ROC** se usa como tercera métrica porque permite validar si el modelo separa adecuadamente las reservas canceladas y no canceladas antes de ajustar el umbral de decisión.

La **precision** se mantiene como métrica de apoyo porque reduce costos operativos, pero bajo los supuestos planteados su impacto económico estimado es menor que el de reducir falsos negativos.

La **accuracy** se considera complementaria porque resume el acierto general, pero no refleja adecuadamente el impacto económico diferenciado de cada tipo de error.
