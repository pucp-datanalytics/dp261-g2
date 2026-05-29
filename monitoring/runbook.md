# Runbook Sprint 6 — Predicción de Cancelaciones Hoteleras

# Objetivo

Documento operativo para monitorear, diagnosticar y recuperar la API de predicción desplegada en AWS.

---

# Servicios Cubiertos

* API FastAPI

  * `/health`
  * `/docs`
  * `/predict`

* Contenedor Docker desplegado en AWS EC2

* Dashboard Streamlit conectado a la API

* AWS CloudWatch

  * métricas
  * dashboard
  * alarmas

---

# Métricas Objetivo

* `/health` responde HTTP 200
* CPU EC2 menor a 80% en condiciones normales
* Error rate 5xx menor a 1%
* API accesible desde dashboard
* Contenedor Docker en estado healthy

---

# Validación Rápida

## Verificar salud API

```bash
curl -i http://<EC2-IP>:8000/health
```

---

## Verificar documentación Swagger

```bash
http://<EC2-IP>:8000/docs
```

---

## Verificar contenedor Docker

```bash
sudo docker ps
```

---

# Incidente: API no responde

## Diagnóstico

Verificar si el contenedor sigue activo:

```bash
sudo docker ps
```

Revisar logs:

```bash
sudo docker logs hotel-api --tail 100
```

Verificar métricas CloudWatch:

* CPU
* Network
* StatusCheckFailed

---

## Acción

Reiniciar contenedor:

```bash
sudo docker restart hotel-api
```

---

# Incidente: Error 5xx elevado

## Posibles causas

* error interno FastAPI
* modelo no cargado
* payload inválido
* falta de memoria

---

## Diagnóstico

Revisar logs:

```bash
sudo docker logs hotel-api
```

Verificar requests desde `/docs`.

---

## Acción

* validar estructura de entrada
* revisar modelo cargado
* reiniciar contenedor si aplica

---

# Incidente: CPU elevada

## Diagnóstico

Revisar métrica:

```text
CPUUtilization
```

en CloudWatch.

---

## Acción

* reiniciar contenedor
* reducir tráfico
* revisar requests concurrentes

---

# Rollback

## Procedimiento

Detener contenedor actual:

```bash
sudo docker stop hotel-api
```

Eliminar contenedor:

```bash
sudo docker rm hotel-api
```

Levantar versión previa estable:

```bash
sudo docker run -d -p 8000:8000 --name hotel-api <imagen_previa>
```

---

# Monitoreo de Drift

## Objetivo

Comparar distribución de variables actuales respecto al baseline Sprint 2.

---

## Variables monitoreadas

* ADR
* Lead Time
* Previous Cancellations
* Booking Changes
* Deposit Type

---

## Acción ante drift

* revisar desempeño reciente
* evaluar necesidad de reentrenamiento
* validar calidad de datos

---

# Herramientas Utilizadas

* FastAPI
* Docker
* AWS EC2
* AWS CloudWatch
* Streamlit

---

# Responsables

* Backend/API Team
* ML Team
* Monitoring Team
