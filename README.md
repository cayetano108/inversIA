---
title: InversIA
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 📈 inversIA – Asistente de Educación Financiera basado en RAG

> **Nota:** este repositorio recoge el código del proyecto con fines de consulta y documentación del trabajo realizado. No es funcional de forma autónoma: requiere el corpus documental y el índice vectorial FAISS, que no se incluyen aquí. La versión desplegada está disponible en HuggingFace Spaces.

Asistente conversacional de educación financiera para particulares, desarrollado como Trabajo de Fin de Máster del **Máster Universitario en Ciencia de Datos** de la **Universitat de València** (curso 2025–2026).

> **Autor:** Cayetano Fernando Romero Monteagudo  
> **Tutor:** Arturo Sirvent Fresneda  
> **Título del TFM:** *Diseño y Evaluación de Asistente Financiero basado en Retrieval Augmented Generation (RAG)*

---

## ¿Qué es inversIA?

inversIA es un asistente conversacional que combina un grafo de estados construido con **LangGraph** y un pipeline **RAG** (Retrieval-Augmented Generation) para responder preguntas de educación financiera en español. A diferencia de un LLM de propósito general, el sistema fundamenta sus respuestas en un corpus curado y verificado, filtra los fragmentos recuperados mediante un juez de relevancia y valida la respuesta generada antes de devolverla al usuario.

El sistema está pensado para cubrir conceptos como fondos indexados, ETFs, planes de pensiones, interés compuesto, diversificación, fiscalidad básica española (IRPF) y errores comunes del inversor particular.

> ⚠️ **inversIA tiene fines exclusivamente educativos.** Las respuestas no constituyen asesoramiento financiero personalizado en el sentido del artículo 4 de la Directiva MiFID II. Para decisiones de inversión reales, consulta a un profesional autorizado por la CNMV.

---

## Arquitectura

El pipeline se implementa como un grafo de estados de 8 nodos sobre LangGraph:

```
Pregunta → Expansión → Enrutador ──► Recuperación FAISS
                          │                 │
                          │           Juez de relevancia
                          │                 │
                          │           Enriquecimiento (yfinance)
                          │                 │
                          │           Generación
                          │                 │
                          └──► Redirección  Validación doble
                                            │
                                        Respuesta
```

| Nodo | Función |
|------|---------|
| `augment` | Enriquece la pregunta con contexto financiero y el historial reciente |
| `route_question` | Clasifica la pregunta como dentro/fuera del dominio |
| `retrieve` | Recupera los *k=6* fragmentos más relevantes del índice FAISS |
| `grade_documents` | Filtra los fragmentos no relevantes para la pregunta |
| `enrich` | Incorpora datos de mercado en tiempo real vía yfinance (*tool calling*) |
| `generate` | Genera la respuesta usando el contexto recuperado |
| `check` | Verifica ausencia de alucinaciones y adecuación de la respuesta |
| `redirect` | Redirige preguntas fuera del dominio con una respuesta alternativa |

---

## Corpus y vectorstore

El corpus está compuesto por libros de referencia sobre inversión a largo plazo y gestión pasiva (Bogle, Bernstein, Graham, Housel, Paramés, Larimore et al.) junto con 17 guías divulgativas de la CNMV sobre fondos, ETFs, fiscalidad, protección del inversor y fraudes financieros.

Los documentos se segmentan en fragmentos de 500 tokens (solapamiento 50) y se vectorizan con `text-embedding-3-large` de OpenAI, produciendo un índice FAISS de **2.496 vectores**. Cada fragmento incluye un resumen generado con DeepSeek en el campo de metadatos para mejorar la precisión del retrieval.

---

## Stack tecnológico

| Componente | Librería |
|-----------|----------|
| Grafo de estados | `langgraph ≥ 0.4` |
| Integración LLM | `langchain ≥ 0.3`, `langchain-deepseek`, `langchain-openai` |
| Vectorstore | `faiss-cpu ≥ 1.7.4` |
| Embeddings | `text-embedding-3-large` (OpenAI) |
| Modelo generador principal | `deepseek-chat` (DeepSeek) |
| Datos de mercado | `yfinance ≥ 0.2.40` |
| Interfaz conversacional | `chainlit ≥ 2.0` |

---

## Evaluación

El sistema fue evaluado con métricas RAGAS sobre 28 preguntas estratificadas (simples, multiconcepto, comparativas, adversariales y fuera de alcance), comparando 9 configuraciones: 3 variantes del pipeline (RAG-FULL, RAG-LITE, NO-RAG) × 3 modelos generadores (GPT-3.5-turbo, DeepSeek Chat, GPT-4o).

**Resultados destacados (configuraciones RAG-FULL):**

| Modelo | Faithfulness | Answer Relevancy | Coste/consulta |
|--------|-------------|-----------------|----------------|
| GPT-3.5-turbo | 0,803 | 0,693 | $0,001773 |
| DeepSeek Chat | 0,893 | 0,741 | $0,000960 |
| GPT-4o | 0,905 | 0,794 | $0,010021 |

DeepSeek ofrece una fidelidad prácticamente equivalente a GPT-4o a menos de una décima parte del coste, posicionándose como la opción más equilibrada para un despliegue productivo. El nodo de evaluación de documentos concentra cerca del 50% de la latencia total en todas las configuraciones RAG-FULL.

---

## Variables de entorno requeridas

El sistema utiliza las siguientes claves API, configuradas como *secrets* en HuggingFace:

- `DEEPSEEK_API_KEY` — modelos Deepseek
- `OPENAI_API_KEY_EMBEDDING` — para modelos de openAI y embedding (`text-embedding-3-large`)

---

## Privacidad

El historial de conversación se mantiene únicamente en memoria durante la sesión activa y se descarta al cerrarla. No se almacenan datos personales entre sesiones, simplificando el cumplimiento del RGPD.

