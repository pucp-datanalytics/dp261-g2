# Plan de Monitoreo de Drift

## Objetivo

Detectar cambios en la distribución de variables de entrada respecto al baseline definido durante el Sprint 2.

---

# Variables Monitoreadas

* ADR
* Lead Time
* Previous Cancellations
* Booking Changes
* Market Segment
* Deposit Type

---

# Estrategia

Se registrará periódicamente la distribución estadística de las variables de entrada y se comparará contra el dataset de entrenamiento original.

---

# Indicadores Monitoreados

* Media
* Desviación estándar
* Percentiles
* Distribución categórica

---

# Indicadores de Drift

Se considerará posible drift cuando:

* la media cambie más de 20%
* aumente significativamente la varianza
* aparezcan nuevas categorías
* cambie sustancialmente la proporción de clases

---

# Acciones ante Drift

* revisar calidad de datos
* monitorear desempeño reciente
* reevaluar métricas del modelo
* considerar reentrenamiento

---

# Objetivo Operacional

Mantener estabilidad predictiva y reducir degradación del modelo en producción.
