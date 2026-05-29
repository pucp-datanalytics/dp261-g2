# Métricas Monitoreadas

## Objetivo

Definir métricas clave para monitorear desempeño, estabilidad y comportamiento operativo de la API de predicción de cancelaciones hoteleras.

---

# Métricas de Infraestructura

## CPUUtilization

Monitorea el consumo de CPU de la instancia EC2.

### Objetivo

Detectar sobrecarga o uso anómalo del servidor.

---

## NetworkIn

Cantidad de tráfico entrante hacia la instancia.

### Objetivo

Monitorear volumen de requests y actividad de usuarios.

---

## NetworkOut

Cantidad de tráfico saliente desde la instancia.

### Objetivo

Supervisar respuestas enviadas por la API.

---

## StatusCheckFailed

Estado de salud de la instancia EC2.

### Objetivo

Detectar fallos críticos de infraestructura.

---

# Métricas de Aplicación

## Latencia p50/p95/p99

Tiempo de respuesta de requests hacia la API.

### Objetivo

Monitorear rendimiento y detectar degradación de desempeño.

* p50: latencia típica
* p95: requests lentas
* p99: casos extremos

---

## Throughput

Cantidad de requests procesadas por unidad de tiempo.

### Objetivo

Medir capacidad y carga operativa del sistema.

---

## Error Rate 5xx

Porcentaje de requests que retornan errores del servidor.

### Fórmula

Error Rate 5xx = Requests 5xx / Requests Totales × 100

### Objetivo

Detectar fallos operativos y degradación de estabilidad.

---

# Métricas de Negocio

## Revenue en Riesgo

Monto estimado asociado a reservas clasificadas como alto riesgo.

---

## Ahorro Estimado

Revenue potencialmente protegido mediante acciones preventivas.

### Fórmula

Ahorro Estimado = Revenue Perdido × Tasa de Recuperación
