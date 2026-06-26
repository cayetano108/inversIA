#!/usr/bin/env python3
"""
eval_configs.py — Evaluación sistemática de inversIA

Configuraciones evaluadas (§11.3):
  C1  RAG-FULL  GPT-3.5-turbo   Grafo completo con todos los nodos activos
  C2  RAG-LITE  GPT-3.5-turbo   Sin expansión de pregunta ni juez de documentos
  C3  NO-RAG    GPT-3.5-turbo   LLM directo sin recuperación
  C4  RAG-FULL  DeepSeek Chat   Mismo grafo, modelo distinto
  C5  RAG-LITE  DeepSeek Chat   Ablación reducida con DeepSeek
  C6  NO-RAG    DeepSeek Chat   LLM DeepSeek directo sin recuperación
  C7  RAG-FULL  GPT-4o          Grafo completo con GPT-4o
  C8  RAG-LITE  GPT-4o          Ablación reducida con GPT-4o
  C9  NO-RAG    GPT-4o          LLM GPT-4o directo sin recuperación

Métricas (§11.4):
  - faithfulness + answer_relevancy  (RAGAS)
  - latencia total: p50, p90 y media
  - coste aproximado por consulta (estimación por tokens)
  - tasa de rechazo correcto en preguntas fuera de dominio
  - desglose de latencia por nodo (solo configuraciones RAG-FULL)

Protocolo (§11.5):
  - temperature=0 en todos los modelos generadores
  - N_RUNS corridas por pregunta (por defecto 1)
  - Registro en JSONL + checkpoint para reanudar ejecuciones interrumpidas
  - Los jueces internos (relevancia, alucinaciones, adecuación) usan siempre
    DeepSeek para garantizar consistencia entre configuraciones

Variables de entorno requeridas (.env o secrets de HuggingFace):
  DEEPSEEK_API_KEY         → modelo DeepSeek Chat (generador y jueces internos)
  OPENAI_API_KEY_EMBEDDING → GPT-3.5-turbo, GPT-4o y embeddings FAISS

Uso:
  python eval_configs.py                                    # 1 run por pregunta
  N_RUNS=3 python eval_configs.py                          # 3 runs (más estable)
  EVAL_DIR=outputs/eval_20260427 python eval_configs.py    # reanuda checkpoint
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from statistics import mean

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_inversIA")

# ── Parámetros globales ───────────────────────────────────────────────────────

N_RUNS = int(os.getenv("N_RUNS", "1"))

_eval_dir_env = os.getenv("EVAL_DIR")
if _eval_dir_env:
    OUT_DIR = Path(_eval_dir_env)
else:
    OUT_DIR = Path("outputs") / f"eval_inversIA_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = OUT_DIR / "checkpoint.json"
RUNS_LOG_PATH   = OUT_DIR / "runs.jsonl"

# Tarifas estimadas (USD / 1 000 tokens, 2026-Q1)
COST_PER_1K: dict[str, dict[str, float]] = {
    "openai":   {"input": 0.0005,  "output": 0.0010},   # gpt-3.5-turbo
    "deepseek": {"input": 0.00014, "output": 0.00028},   # deepseek-chat (cache miss)
    "gpt-4o":   {"input": 0.0025,  "output": 0.0100},   # gpt-4o
}

# ── Modelos LLM ───────────────────────────────────────────────────────────────
# Se instancian aquí para no depender de variables globales de app_workflow,
# lo que también hace el script más fácil de ejecutar de forma independiente.

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
_OPENAI_KEY   = os.getenv("OPENAI_API_KEY_EMBEDDING")

llm_deepseek = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=_DEEPSEEK_KEY,
)

llm_openai = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=_OPENAI_KEY,
)

llm_gpt4o = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    openai_api_key=_OPENAI_KEY,
)

_LLM_MAP: dict = {
    "openai":   llm_openai,
    "deepseek": llm_deepseek,
    "gpt-4o":   llm_gpt4o,
}

# ── Imports de app_workflow (carga retriever y jueces; puede tardar ~20 s) ────
log.info("Cargando app_workflow (vectorstore + jueces)…")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

try:
    from app_workflow import (
        retriever,
        generate_augment_query,
        retrieval_grader_relevance,   # juez de relevancia — DeepSeek fijo
        hallucination_grader,          # juez de alucinaciones — DeepSeek fijo
        answer_grader,                 # juez de adecuación — DeepSeek fijo
    )
    log.info("app_workflow cargado correctamente.")
except Exception as exc:
    log.error("Error cargando app_workflow: %s", exc)
    raise SystemExit(1)


# ── 1. Helpers: tokens, coste y rechazo ──────────────────────────────────────

def approx_tokens(text: str) -> int:
    """Estimación rápida: ~4 chars/token (sin tokenizador externo)."""
    return max(1, len(text) // 4)


def estimate_cost(model: str, tok_in: int, tok_out: int) -> float:
    """Coste aproximado en USD para una llamada al modelo dado."""
    rates = COST_PER_1K.get(model, {"input": 0.0, "output": 0.0})
    return (tok_in * rates["input"] + tok_out * rates["output"]) / 1000


def estimate_llm_call_cost(
    model: str, prompt_text: str, output_text: str, node: str
) -> dict:
    """
    Devuelve un dict con el desglose de tokens y coste de una llamada LLM.
    La estimación se basa en caracteres (~4 chars/token), no en el contador
    real de la API, pero es homogénea entre configuraciones.
    """
    tok_in  = approx_tokens(prompt_text)
    tok_out = approx_tokens(output_text)
    return {
        "node":      node,
        "model":     model,
        "tokens_in": tok_in,
        "tokens_out": tok_out,
        "cost_usd":  round(estimate_cost(model, tok_in, tok_out), 8),
    }


# Frases que indican que el sistema rechazó una pregunta fuera de dominio
_REFUSAL_MARKERS = [
    "no puedo responder",
    "no está relacionad",
    "fuera de mi especialidad",
    "no es mi área",
    "no es un tema de inversión",
    "no se ajusta al propósito",
    "fuera de alcance",
    "no forma parte",
    "pregúntame sobre inversión",
    "mi propósito es",
    "especialidad es",
    "no está en mi",
]


def is_refusal(text: str) -> bool:
    """Detecta si la respuesta generada rechaza la pregunta por estar fuera de dominio."""
    t = text.lower()
    return any(marker in t for marker in _REFUSAL_MARKERS)


def _content_to_text(obj) -> str:
    """Convierte salidas de LangChain a texto para estimar tokens."""
    if obj is None:
        return ""
    if hasattr(obj, "content"):
        return str(obj.content)
    if hasattr(obj, "binary_score"):
        return str(obj.binary_score)
    return str(obj)


def _timed(fn, *args, **kwargs):
    """Ejecuta fn(*args, **kwargs) y devuelve (resultado, segundos_transcurridos)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round(time.perf_counter() - t0, 3)


# ── 2. Dataset de evaluación — 28 preguntas estratificadas (§11.2) ────────────

QUESTIONS: list[dict] = [
    # ── Factuales simples (10) ────────────────────────────────────────────────
    {
        "id": "Q01", "cat": "factual_simple",
        "question": "¿Qué es el interés compuesto?",
        "expected_claims": ["reinvertir intereses", "crecimiento exponencial", "efecto bola de nieve"],
        "expected_refusal": False,
    },
    {
        "id": "Q02", "cat": "factual_simple",
        "question": "¿Qué es un fondo indexado?",
        "expected_claims": ["replica un índice", "gestión pasiva", "bajos costes de gestión"],
        "expected_refusal": False,
    },
    {
        "id": "Q03", "cat": "factual_simple",
        "question": "¿Qué es el DCA (Dollar Cost Averaging)?",
        "expected_claims": ["aportaciones periódicas", "reduce el riesgo de timing", "precio medio de compra"],
        "expected_refusal": False,
    },
    {
        "id": "Q04", "cat": "factual_simple",
        "question": "¿Qué es un ETF?",
        "expected_claims": ["cotiza en bolsa", "fondo de inversión", "diversificación"],
        "expected_refusal": False,
    },
    {
        "id": "Q05", "cat": "factual_simple",
        "question": "¿Qué es la diversificación de cartera?",
        "expected_claims": ["repartir el riesgo", "distintos activos", "correlación baja"],
        "expected_refusal": False,
    },
    {
        "id": "Q06", "cat": "factual_simple",
        "question": "¿Qué significa que un fondo sea de acumulación?",
        "expected_claims": ["reinvierte los dividendos", "no reparte dividendos", "ventaja fiscal"],
        "expected_refusal": False,
    },
    {
        "id": "Q07", "cat": "factual_simple",
        "question": "¿Qué es el rebalanceo de cartera?",
        "expected_claims": ["restaurar proporciones objetivo", "vender lo que subió", "comprar lo que bajó"],
        "expected_refusal": False,
    },
    {
        "id": "Q08", "cat": "factual_simple",
        "question": "¿Qué es el índice S&P 500?",
        "expected_claims": ["500 grandes empresas", "mercado americano", "referencia global"],
        "expected_refusal": False,
    },
    {
        "id": "Q09", "cat": "factual_simple",
        "question": "¿Qué es el horizonte temporal en inversión?",
        "expected_claims": ["tiempo previsto de inversión", "mayor plazo mayor riesgo asumible", "largo plazo"],
        "expected_refusal": False,
    },
    {
        "id": "Q10", "cat": "factual_simple",
        "question": "¿Qué es la inflación y cómo afecta al ahorro?",
        "expected_claims": ["pérdida de poder adquisitivo", "erosiona el dinero parado", "necesidad de invertir"],
        "expected_refusal": False,
    },
    # ── Multiconcepto — requieren conectar ≥2 conceptos (6) ──────────────────
    {
        "id": "Q11", "cat": "multi_concept",
        "question": "¿Qué ventaja fiscal tiene un ETF de acumulación frente a uno de distribución en España?",
        "expected_claims": ["no tributa al reinvertir dividendos", "difiere el pago de impuestos", "IRPF"],
        "expected_refusal": False,
    },
    {
        "id": "Q12", "cat": "multi_concept",
        "question": "¿Cómo afecta la inflación al rendimiento real de una inversión en renta fija?",
        "expected_claims": ["rentabilidad real = nominal menos inflación", "puede ser negativa", "erosiona el cupón"],
        "expected_refusal": False,
    },
    {
        "id": "Q13", "cat": "multi_concept",
        "question": "¿Cuál es el impacto de los costes de gestión (TER) en la rentabilidad a largo plazo de un fondo?",
        "expected_claims": ["se descuentan anualmente", "efecto del interés compuesto sobre los costes", "reducen rentabilidad"],
        "expected_refusal": False,
    },
    {
        "id": "Q14", "cat": "multi_concept",
        "question": "¿Por qué el DCA puede reducir el riesgo en mercados volátiles y cómo se relaciona con la diversificación temporal?",
        "expected_claims": ["compra más participaciones cuando el mercado cae", "precio medio inferior", "diversificación en el tiempo"],
        "expected_refusal": False,
    },
    {
        "id": "Q15", "cat": "multi_concept",
        "question": "¿Qué ocurre fiscalmente cuando se hace un traspaso entre fondos de inversión en España?",
        "expected_claims": ["no tributa al traspasar", "difiere la plusvalía", "solo tributa al reembolsar"],
        "expected_refusal": False,
    },
    {
        "id": "Q16", "cat": "multi_concept",
        "question": "¿Cómo influye la subida del tipo de interés del BCE en el precio de los bonos del Estado?",
        "expected_claims": ["relación inversa precio-tipo", "precio de los bonos cae", "afecta a la renta fija existente"],
        "expected_refusal": False,
    },
    # ── Comparativas (5) ──────────────────────────────────────────────────────
    {
        "id": "Q17", "cat": "comparative",
        "question": "¿En qué se diferencia un fondo indexado de un ETF?",
        "expected_claims": ["ETF cotiza en bolsa en tiempo real", "fondo se suscribe al valor liquidativo", "diferencias de liquidez y costes"],
        "expected_refusal": False,
    },
    {
        "id": "Q18", "cat": "comparative",
        "question": "¿Cuál es la diferencia entre renta fija y renta variable?",
        "expected_claims": ["fija tiene rendimiento predeterminado", "variable depende del mercado", "diferente perfil de riesgo"],
        "expected_refusal": False,
    },
    {
        "id": "Q19", "cat": "comparative",
        "question": "¿Qué diferencia hay entre un fondo de gestión activa y uno de gestión pasiva?",
        "expected_claims": ["activa busca batir al índice", "pasiva replica el índice", "costes mayores en gestión activa"],
        "expected_refusal": False,
    },
    {
        "id": "Q20", "cat": "comparative",
        "question": "¿En qué se parecen y diferencian la inversión en acciones individuales y en fondos indexados?",
        "expected_claims": ["ambos son renta variable", "fondos ofrecen más diversificación", "acciones individuales mayor riesgo concentrado"],
        "expected_refusal": False,
    },
    {
        "id": "Q21", "cat": "comparative",
        "question": "¿Cuál es la diferencia entre diversificación geográfica y diversificación por tipo de activo?",
        "expected_claims": ["geográfica reparte por países o regiones", "por activo mezcla acciones bonos inmuebles etc", "son complementarias"],
        "expected_refusal": False,
    },
    # ── Fuera de alcance — deben ser rechazadas (4) ───────────────────────────
    {
        "id": "Q22", "cat": "out_of_scope",
        "question": "¿Qué temperatura hará mañana en Madrid?",
        "expected_claims": [],
        "expected_refusal": True,
    },
    {
        "id": "Q23", "cat": "out_of_scope",
        "question": "¿Cuál es la mejor receta de paella valenciana?",
        "expected_claims": [],
        "expected_refusal": True,
    },
    {
        "id": "Q24", "cat": "out_of_scope",
        "question": "¿Cuántos planetas tiene el sistema solar?",
        "expected_claims": [],
        "expected_refusal": True,
    },
    {
        "id": "Q25", "cat": "out_of_scope",
        "question": "¿Quién ganó el último Mundial de fútbol?",
        "expected_claims": [],
        "expected_refusal": True,
    },
    # ── Adversariales — premisa falsa que el sistema debe corregir (3) ────────
    {
        "id": "Q26", "cat": "adversarial",
        "question": "¿Es cierto que los ETFs en España tributan siempre al 19% independientemente de la ganancia?",
        "expected_claims": ["falso", "tramos del IRPF", "el tipo depende del importe de la ganancia"],
        "expected_refusal": False,
    },
    {
        "id": "Q27", "cat": "adversarial",
        "question": "Los fondos indexados siempre pierden dinero a largo plazo, ¿verdad?",
        "expected_claims": ["incorrecto", "históricamente rentables a largo plazo", "bolsa sube a largo plazo"],
        "expected_refusal": False,
    },
    {
        "id": "Q28", "cat": "adversarial",
        "question": "¿No es verdad que el DCA garantiza ganancias en cualquier mercado?",
        "expected_claims": ["no garantiza ganancias", "reduce el riesgo pero no lo elimina", "pérdidas posibles"],
        "expected_refusal": False,
    },
]

# ── 3. Configuraciones (§11.3) ─────────────────────────────────────────────────

CONFIGS: list[dict] = [
    {"id": "C1", "name": "RAG-FULL", "model": "openai",   "type": "rag_full"},
    {"id": "C2", "name": "RAG-LITE", "model": "openai",   "type": "rag_lite"},
    {"id": "C3", "name": "NO-RAG",   "model": "openai",   "type": "no_rag"},
    {"id": "C4", "name": "RAG-FULL", "model": "deepseek", "type": "rag_full"},
    {"id": "C5", "name": "RAG-LITE", "model": "deepseek", "type": "rag_lite"},
    {"id": "C6", "name": "NO-RAG",   "model": "deepseek", "type": "no_rag"},
    {"id": "C7", "name": "RAG-FULL", "model": "gpt-4o",   "type": "rag_full"},
    {"id": "C8", "name": "RAG-LITE", "model": "gpt-4o",   "type": "rag_lite"},
    {"id": "C9", "name": "NO-RAG",   "model": "gpt-4o",   "type": "no_rag"},
]


def get_llm(model_name: str):
    llm = _LLM_MAP.get(model_name)
    if llm is None:
        raise ValueError(f"Modelo no reconocido: '{model_name}'. "
                         f"Opciones válidas: {list(_LLM_MAP)}")
    return llm


# ── 4. Prompt templates ───────────────────────────────────────────────────────

_GEN_PROMPT = ChatPromptTemplate.from_template(
    "Eres un asistente experto en inversión a largo plazo y educación financiera.\n"
    "Usa únicamente el contexto proporcionado para responder. "
    "No menciones las fuentes ni tu condición de asistente.\n\n"
    "Pregunta: {question}\n"
    "Contexto: {context}\n"
    "Respuesta:"
)

_NO_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en inversión a largo plazo y educación financiera en español."),
    ("human", "{question}"),
])


def _gen_prompt_text(question: str, context: str) -> str:
    """Texto aproximado del prompt de generación (para estimar tokens)."""
    return (
        "Eres un asistente experto en inversión a largo plazo y educación financiera.\n"
        "Usa únicamente el contexto proporcionado para responder. "
        "No menciones las fuentes ni tu condición de asistente.\n\n"
        f"Pregunta: {question}\n"
        f"Contexto: {context}\n"
        "Respuesta:"
    )


def _no_rag_prompt_text(question: str) -> str:
    """Texto aproximado del prompt NO-RAG (para estimar tokens)."""
    return (
        "Eres un experto en inversión a largo plazo y educación financiera en español.\n\n"
        f"{question}"
    )


# ── 5. Runners ────────────────────────────────────────────────────────────────

def run_rag_full(question: str, llm, model_name: str) -> dict:
    """
    Pipeline completo (RAG-FULL):
      expansión → recuperación → juez de documentos → generación
      → comprobador de alucinaciones → comprobador de adecuación.

    Los jueces internos usan siempre DeepSeek para consistencia entre configs.
    El coste devuelto es la suma de todas las llamadas LLM del flujo.
    """
    timings: dict        = {}
    cost_breakdown: list = []
    t_total = time.perf_counter()

    # 1. Expansión de pregunta
    aug_msg, timings["augment_sec"] = _timed(generate_augment_query, question, llm)
    aug_content = _content_to_text(aug_msg)
    aug_text    = f"{question}\n\n{aug_content.strip()}"
    cost_breakdown.append(estimate_llm_call_cost(
        model_name, prompt_text=question, output_text=aug_content, node="augment"
    ))

    # 2. Recuperación FAISS (sin coste LLM)
    docs, timings["retrieve_sec"] = _timed(retriever.invoke, aug_text)

    # 3. Juez de relevancia (DeepSeek fijo, una llamada por documento)
    def _grade_docs():
        relevant, local_costs = [], []
        for i, doc in enumerate(docs, start=1):
            score = retrieval_grader_relevance.invoke(
                {"question": question, "document": doc.page_content}
            )
            local_costs.append(estimate_llm_call_cost(
                "deepseek",
                prompt_text=f"Pregunta: {question}\n\nDocumento: {doc.page_content}",
                output_text=_content_to_text(score),
                node=f"grade_documents_{i}",
            ))
            if score.binary_score.lower() in ["sí", "si", "yes"]:
                relevant.append(doc)
        return relevant, local_costs

    grade_result, timings["grade_docs_sec"] = _timed(_grade_docs)
    filtered, grade_costs = grade_result
    cost_breakdown.extend(grade_costs)
    effective = filtered if filtered else docs[:3]

    # 4. Generación
    context    = "\n\n".join(d.page_content for d in effective)
    chain      = _GEN_PROMPT | llm | StrOutputParser()
    generation, timings["generate_sec"] = _timed(
        chain.invoke, {"question": question, "context": context}
    )
    cost_breakdown.append(estimate_llm_call_cost(
        model_name,
        prompt_text=_gen_prompt_text(question, context),
        output_text=generation,
        node="generate",
    ))

    # 5. Comprobador de alucinaciones (DeepSeek fijo)
    hall, timings["hallucination_sec"] = _timed(
        hallucination_grader.invoke,
        {"documents": effective, "generation": generation},
    )
    hall_prompt = (
        "Documentos:\n" + "\n\n".join(d.page_content for d in effective)
        + f"\n\nGeneración:\n{generation}"
    )
    cost_breakdown.append(estimate_llm_call_cost(
        "deepseek",
        prompt_text=hall_prompt,
        output_text=_content_to_text(hall),
        node="hallucination_check",
    ))

    # 6. Comprobador de adecuación (DeepSeek fijo)
    ans_q, timings["answer_quality_sec"] = _timed(
        answer_grader.invoke,
        {"question": question, "generation": generation},
    )
    cost_breakdown.append(estimate_llm_call_cost(
        "deepseek",
        prompt_text=f"Pregunta: {question}\n\nGeneración:\n{generation}",
        output_text=_content_to_text(ans_q),
        node="answer_quality_check",
    ))

    total_cost   = sum(c["cost_usd"]   for c in cost_breakdown)
    total_tok_in = sum(c["tokens_in"]  for c in cost_breakdown)
    total_tok_out= sum(c["tokens_out"] for c in cost_breakdown)

    return {
        "generation":             generation,
        "contexts":               [d.page_content for d in effective],
        "latency_sec":            round(time.perf_counter() - t_total, 3),
        "node_timings":           timings,
        "n_docs_retrieved":       len(docs),
        "n_docs_after_grading":   len(filtered),
        "hallucination_check":    hall.binary_score,
        "answer_quality_check":   ans_q.binary_score,
        "cost_breakdown":         cost_breakdown,
        "tokens_in_total_approx": total_tok_in,
        "tokens_out_total_approx":total_tok_out,
        "cost_usd_total_approx":  round(total_cost, 6),
    }


def run_rag_lite(question: str, llm, model_name: str) -> dict:
    """
    Pipeline reducido (RAG-LITE): recuperación directa → generación.
    Sin expansión de pregunta ni juez de documentos.
    """
    t_total = time.perf_counter()
    docs      = retriever.invoke(question)
    context   = "\n\n".join(d.page_content for d in docs)
    chain     = _GEN_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({"question": question, "context": context})

    cost_breakdown = [estimate_llm_call_cost(
        model_name,
        prompt_text=_gen_prompt_text(question, context),
        output_text=generation,
        node="generate",
    )]
    total_cost    = sum(c["cost_usd"]   for c in cost_breakdown)
    total_tok_in  = sum(c["tokens_in"]  for c in cost_breakdown)
    total_tok_out = sum(c["tokens_out"] for c in cost_breakdown)

    return {
        "generation":             generation,
        "contexts":               [d.page_content for d in docs],
        "latency_sec":            round(time.perf_counter() - t_total, 3),
        "node_timings":           {},
        "n_docs_retrieved":       len(docs),
        "n_docs_after_grading":   len(docs),
        "hallucination_check":    None,
        "answer_quality_check":   None,
        "cost_breakdown":         cost_breakdown,
        "tokens_in_total_approx": total_tok_in,
        "tokens_out_total_approx":total_tok_out,
        "cost_usd_total_approx":  round(total_cost, 6),
    }


def run_no_rag(question: str, llm, model_name: str) -> dict:
    """
    LLM directo (NO-RAG): llamada al modelo sin recuperación de contexto.
    Sirve como baseline para cuantificar el aporte del pipeline RAG.
    """
    t_total   = time.perf_counter()
    chain     = _NO_RAG_PROMPT | llm | StrOutputParser()
    generation = chain.invoke({"question": question})

    cost_breakdown = [estimate_llm_call_cost(
        model_name,
        prompt_text=_no_rag_prompt_text(question),
        output_text=generation,
        node="generate_no_rag",
    )]
    total_cost    = sum(c["cost_usd"]   for c in cost_breakdown)
    total_tok_in  = sum(c["tokens_in"]  for c in cost_breakdown)
    total_tok_out = sum(c["tokens_out"] for c in cost_breakdown)

    return {
        "generation":             generation,
        "contexts":               [],
        "latency_sec":            round(time.perf_counter() - t_total, 3),
        "node_timings":           {},
        "n_docs_retrieved":       0,
        "n_docs_after_grading":   0,
        "hallucination_check":    None,
        "answer_quality_check":   None,
        "cost_breakdown":         cost_breakdown,
        "tokens_in_total_approx": total_tok_in,
        "tokens_out_total_approx":total_tok_out,
        "cost_usd_total_approx":  round(total_cost, 6),
    }


RUNNERS: dict = {
    "rag_full": run_rag_full,
    "rag_lite": run_rag_lite,
    "no_rag":   run_no_rag,
}


# ── 6. Checkpoint ─────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save_checkpoint(cp: dict) -> None:
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as fh:
        json.dump(cp, fh, ensure_ascii=False)


def append_run_log(record: dict) -> None:
    with open(RUNS_LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# ── 7. Bucle de evaluación ────────────────────────────────────────────────────

def run_evaluation() -> list[dict]:
    checkpoint = load_checkpoint()
    all_runs   = list(checkpoint.values())
    total      = len(QUESTIONS) * len(CONFIGS) * N_RUNS
    done       = len(all_runs)

    log.info(
        "Llamadas previstas: %d (%d preguntas × %d configs × %d runs) | "
        "Ya completadas: %d",
        total, len(QUESTIONS), len(CONFIGS), N_RUNS, done,
    )

    for cfg in CONFIGS:
        llm    = get_llm(cfg["model"])
        runner = RUNNERS[cfg["type"]]

        for q in QUESTIONS:
            for run_i in range(1, N_RUNS + 1):
                key = f"{cfg['id']}_{q['id']}_r{run_i}"
                if key in checkpoint:
                    log.debug("[SKIP] %s", key)
                    continue

                log.info("[RUN ] %s | %s | %s", key, cfg["model"], q["question"][:60])
                try:
                    result = runner(q["question"], llm, cfg["model"])

                    cost_breakdown = result.get("cost_breakdown", [])
                    tok_in   = result.get("tokens_in_total_approx",
                                          sum(c.get("tokens_in",  0) for c in cost_breakdown))
                    tok_out  = result.get("tokens_out_total_approx",
                                          sum(c.get("tokens_out", 0) for c in cost_breakdown))
                    total_cost = result.get("cost_usd_total_approx",
                                            sum(c.get("cost_usd", 0.0) for c in cost_breakdown))
                    refusal  = is_refusal(result["generation"])

                    record = {
                        "run_key":          key,
                        "config_id":        cfg["id"],
                        "config_name":      cfg["name"],
                        "config_type":      cfg["type"],
                        "model":            cfg["model"],
                        "question_id":      q["id"],
                        "category":         q["cat"],
                        "question":         q["question"],
                        "run":              run_i,
                        **result,
                        "tokens_in_approx":  tok_in,
                        "tokens_out_approx": tok_out,
                        "cost_usd_approx":   round(total_cost, 6),
                        "cost_breakdown":    cost_breakdown,
                        "is_refusal":        refusal,
                        "expected_refusal":  q["expected_refusal"],
                        "refusal_correct":   refusal == q["expected_refusal"],
                        "error":             None,
                    }
                except Exception as exc:
                    log.error("[ERR ] %s: %s", key, exc)
                    record = {
                        "run_key":           key,
                        "config_id":         cfg["id"],
                        "config_name":       cfg["name"],
                        "config_type":       cfg["type"],
                        "model":             cfg["model"],
                        "question_id":       q["id"],
                        "category":          q["cat"],
                        "question":          q["question"],
                        "run":               run_i,
                        "generation":        "",
                        "contexts":          [],
                        "latency_sec":       None,
                        "node_timings":      {},
                        "n_docs_retrieved":  0,
                        "n_docs_after_grading": 0,
                        "hallucination_check":  None,
                        "answer_quality_check": None,
                        "tokens_in_approx":  0,
                        "tokens_out_approx": 0,
                        "cost_usd_approx":   0.0,
                        "cost_breakdown":    [],
                        "is_refusal":        False,
                        "expected_refusal":  q["expected_refusal"],
                        "refusal_correct":   False,
                        "error":             str(exc),
                    }

                checkpoint[key] = record
                append_run_log(record)
                save_checkpoint(checkpoint)
                all_runs.append(record)
                done += 1

                log.info(
                    "       lat=%.2fs  cost=$%.5f  [%d/%d]",
                    record["latency_sec"] or 0.0,
                    record["cost_usd_approx"],
                    done, total,
                )

    return all_runs


# ── 8. RAGAS ──────────────────────────────────────────────────────────────────

def run_ragas(all_runs: list[dict]) -> dict:
    """
    Calcula faithfulness y answer_relevancy con RAGAS para cada configuración.
    Usa únicamente el run=1 de cada pregunta para evitar triplicar el coste.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        log.warning("ragas o datasets no instalados; omitiendo métricas RAGAS.")
        return {}

    ragas_out: dict = {}

    by_cfg: dict[str, list] = {}
    for r in all_runs:
        if r.get("error") or r["run"] != 1:
            continue
        by_cfg.setdefault(r["config_id"], []).append(r)

    for cid, rows in by_cfg.items():
        log.info("RAGAS → %s (%d preguntas)…", cid, len(rows))
        try:
            has_ctx = any(r["contexts"] for r in rows)
            ds = Dataset.from_dict({
                "question": [r["question"]   for r in rows],
                "answer":   [r["generation"] for r in rows],
                "contexts": [r["contexts"] if r["contexts"] else [""] for r in rows],
            })
            metrics = [answer_relevancy]
            if has_ctx:
                metrics.append(faithfulness)

            res = ragas_evaluate(ds, metrics=metrics)
            df  = res.to_pandas()

            entry: dict = {
                "n_questions":           len(rows),
                "answer_relevancy_mean": round(float(df["answer_relevancy"].mean()), 4),
                "answer_relevancy_std":  round(float(df["answer_relevancy"].std()),  4),
                "per_question":          df.to_dict("records"),
            }
            if "faithfulness" in df.columns:
                entry["faithfulness_mean"] = round(float(df["faithfulness"].mean()), 4)
                entry["faithfulness_std"]  = round(float(df["faithfulness"].std()),  4)

            ragas_out[cid] = entry

        except Exception as exc:
            log.error("RAGAS error %s: %s", cid, exc)
            ragas_out[cid] = {"error": str(exc)}

    return ragas_out


# ── 9. Tabla resumen ──────────────────────────────────────────────────────────

def build_summary(all_runs: list[dict], ragas_out: dict) -> pd.DataFrame:
    """Construye la tabla principal de resultados agregados por configuración."""
    rows = []
    for cfg in CONFIGS:
        cid  = cfg["id"]
        good = [r for r in all_runs if r["config_id"] == cid and not r.get("error")]
        if not good:
            continue

        lats      = sorted(r["latency_sec"] for r in good if r["latency_sec"])
        costs     = [r["cost_usd_approx"] for r in good]
        refusal_q = [r for r in good if r["expected_refusal"]]

        p50 = lats[len(lats) // 2]       if lats else None
        p90 = lats[int(len(lats) * 0.9)] if lats else None

        row: dict = {
            "config_id":          cid,
            "config_name":        cfg["name"],
            "config_type":        cfg["type"],
            "model":              cfg["model"],
            "n_runs":             len(good),
            "latency_p50_s":      round(p50,  2) if p50  else None,
            "latency_p90_s":      round(p90,  2) if p90  else None,
            "latency_mean_s":     round(mean(lats), 2) if lats else None,
            "total_cost_usd":     round(sum(costs), 4),
            "cost_per_query_usd": round(mean(costs), 6) if costs else None,
            "refusal_accuracy":   (
                round(sum(r["refusal_correct"] for r in refusal_q) / len(refusal_q), 3)
                if refusal_q else None
            ),
            "faithfulness_mean":      None,
            "faithfulness_std":       None,
            "answer_relevancy_mean":  None,
            "answer_relevancy_std":   None,
        }

        rr = ragas_out.get(cid, {})
        if "error" not in rr:
            row["faithfulness_mean"]     = rr.get("faithfulness_mean")
            row["faithfulness_std"]      = rr.get("faithfulness_std")
            row["answer_relevancy_mean"] = rr.get("answer_relevancy_mean")
            row["answer_relevancy_std"]  = rr.get("answer_relevancy_std")

        rows.append(row)

    return pd.DataFrame(rows)


# ── 10. Tabla de latencia por nodo (RAG-FULL) ─────────────────────────────────

def build_node_timings(all_runs: list[dict]) -> pd.DataFrame | None:
    """Devuelve la latencia media por nodo para las configuraciones RAG-FULL."""
    rag_full = [
        r for r in all_runs
        if r["config_type"] == "rag_full"
        and r.get("node_timings")
        and not r.get("error")
    ]
    if not rag_full:
        return None

    node_keys = [
        "augment_sec", "retrieve_sec", "grade_docs_sec",
        "generate_sec", "hallucination_sec", "answer_quality_sec",
    ]
    records = []
    for r in rag_full:
        nt    = r.get("node_timings", {})
        entry = {"config_id": r["config_id"], "model": r["model"],
                 "question_id": r["question_id"], "run": r["run"]}
        for k in node_keys:
            entry[k] = nt.get(k)
        records.append(entry)

    df = pd.DataFrame(records)
    return (
        df.groupby(["config_id", "model"])[[k for k in node_keys]]
        .mean()
        .round(3)
        .reset_index()
    )


# ── 11. Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 65)
    log.info("inversIA — Evaluación de configuraciones RAG")
    log.info("Output: %s", OUT_DIR)
    log.info(
        "Plan: %d preguntas × %d configs × %d runs = %d llamadas LLM",
        len(QUESTIONS), len(CONFIGS), N_RUNS,
        len(QUESTIONS) * len(CONFIGS) * N_RUNS,
    )
    log.info("=" * 65)

    # Guardar dataset de preguntas para referencia
    with open(OUT_DIR / "questions.json", "w", encoding="utf-8") as fh:
        json.dump(QUESTIONS, fh, ensure_ascii=False, indent=2)
    log.info("Dataset de preguntas guardado en questions.json")

    # Bucle principal
    all_runs = run_evaluation()

    # Métricas RAGAS
    log.info("Calculando métricas RAGAS…")
    ragas_out = run_ragas(all_runs)

    # Tablas de resultados
    summary_df    = build_summary(all_runs, ragas_out)
    node_time_df  = build_node_timings(all_runs)

    # Guardar resultados
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    with open(OUT_DIR / "ragas_detail.json", "w", encoding="utf-8") as fh:
        json.dump(ragas_out, fh, ensure_ascii=False, indent=2)
    if node_time_df is not None:
        node_time_df.to_csv(OUT_DIR / "node_timings.csv", index=False)

    # Mostrar tabla resumen
    log.info(
        "\n\n── TABLA RESUMEN ────────────────────────────────────────────\n%s\n",
        summary_df.to_string(index=False),
    )
    if node_time_df is not None:
        log.info(
            "\n── LATENCIA POR NODO (RAG-FULL) ─────────────────────────────\n%s\n",
            node_time_df.to_string(index=False),
        )

    total_runs   = len([r for r in all_runs if not r.get("error")])
    total_errors = len([r for r in all_runs if r.get("error")])
    total_cost   = sum(r["cost_usd_approx"] for r in all_runs)

    log.info("─" * 65)
    log.info(
        "Completados: %d | Errores: %d | Coste total aprox: $%.4f",
        total_runs, total_errors, total_cost,
    )
    log.info("Ficheros en: %s", OUT_DIR)
    log.info("  · summary.csv       → tabla principal de resultados")
    log.info("  · ragas_detail.json → métricas RAGAS por config y pregunta")
    log.info("  · runs.jsonl        → log completo de ejecución")
    log.info("  · node_timings.csv  → latencia por nodo (RAG-FULL)")
    log.info("  · questions.json    → dataset de evaluación")
    log.info("✅ Evaluación completada.")
