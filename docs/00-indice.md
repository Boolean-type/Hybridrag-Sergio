# Documentación de RAG-KG Prototype

Punto de entrada a la documentación del proyecto. Cada documento es
independiente; este índice indica qué leer según lo que necesites.

## Mapa de documentos

| Documento | Para qué | Léelo si... |
|---|---|---|
| [`../README.md`](../README.md) | Visión general, instalación, configuración y comandos | Es tu primer contacto con el proyecto |
| [`01-arquitectura.md`](01-arquitectura.md) | El pipeline de extremo a extremo, los módulos y el modelo de datos del grafo | Quieres entender cómo encaja todo y por dónde fluye un documento o una pregunta |
| [`02-configuracion-por-dominio.md`](02-configuracion-por-dominio.md) | Cómo se define un dominio con los YAML de `configs/` sin tocar Python | Vas a adaptar el sistema a otro tipo de documento o a afinar `offers` |
| [`03-evaluacion.md`](03-evaluacion.md) | Cómo funciona el scoring (determinista + juez) y cómo se lee un run | Vas a evaluar el sistema o interpretar un informe de `data/eval_runs/` |
| [`04-decisiones-y-limitaciones.md`](04-decisiones-y-limitaciones.md) | Por qué está hecho así y qué no hace (todavía) | Quieres el contexto honesto antes de extender o desplegar |
| [`CHANGELOG.md`](CHANGELOG.md) | Evolución del dominio `offers` | Necesitas saber qué cambió y por qué |
| [`PHASES.md`](PHASES.md) | Roadmap por fases y notas de arquitectura | Quieres el plan de implementación y su estado |

## Recorrido recomendado para alguien nuevo

1. **README** (raíz) — monta el entorno y lanza el end-to-end con la oferta de
   muestra.
2. **01-arquitectura** — entiende el flujo ingesta → grafo → recuperación →
   generación → evaluación.
3. **02-configuracion-por-dominio** — mira cómo el comportamiento se define en
   YAML; abre en paralelo `configs/domains/offers/`.
4. **03-evaluacion** — lanza `make eval-quick` y lee tu primer informe.
5. **04-decisiones-y-limitaciones** — para saber qué dar por sólido y qué tomar
   con cautela.

