# main_chainlit.py
import asyncio
from dotenv import load_dotenv

import chainlit as cl
from langchain.schema import Document
from app_workflow import app as langgraph_app

import sys
import io
import time
import contextlib
import base64


import html
from langchain.schema import Document


# ───────────── Captura de prints desde el hilo del workflow ─────────────
class _QueueWriter(io.TextIOBase):
    """Escribe en una asyncio.Queue línea a línea, pensado para usarse en un thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        self.loop = loop
        self.queue = queue
        self._buf = ""

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        # Emite por líneas para no saturar actualizaciones
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            # Entrega al loop principal desde el thread
            asyncio.run_coroutine_threadsafe(self.queue.put(line), self.loop)
        return len(s)

    def flush(self):
        if self._buf:
            asyncio.run_coroutine_threadsafe(self.queue.put(self._buf), self.loop)
            self._buf = ""


load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Config: único LLM disponible
# Cambia el valor a "openai" si quieres alternar rápidamente.
DEFAULT_LLM = "deepseek"

# Nombre del bot (aparece como autor de cada mensaje en la UI)
BOT_NAME = "InversIA"

# ───────────── Plantilla Mermaid del grafo LangGraph ─────────────
# Refleja la topología real de app_workflow.py, con paleta pastel por clase.
# `fallback` (antes websearch) termina en `end_node` y NO vuelve a `generate`:
# así está compilado el grafo en LangGraph (`web_search()` genera su propia
# respuesta alternativa).
MERMAID_TEMPLATE = """flowchart TB
    start(["Inicio"]) --> augment["Ampliación de consulta"]
    augment --> route{"Enrutado de consulta"}
    route -- Consulta financiera --> retrieve["Recuperación vectorial<br>FAISS"]
    retrieve --> grade["Evaluación de documentos"]
    grade -- Documentos relevantes --> enrich["Enriquecimiento<br>financiero"]
    enrich --> generate["Generación de respuesta"]
    generate --> check{"Verificación<br>de respuesta"}
    check -- Respuesta útil --> end_node(["Respuesta"])
    route -- Fuera de dominio --> fallback["Redirección temática<br>con LLM"]
    grade -- Sin contexto relevante --> fallback
    check -- No útil --> fallback
    check -- No soportada --> generate
    fallback --> end_node

     start:::startEnd
     augment:::process
     route:::decision
     retrieve:::retrieval
     grade:::retrieval
     enrich:::generation
     generate:::generation
     check:::decision
     fallback:::fallbackStyle
     end_node:::startEnd
    classDef startEnd fill:#f8fafc,stroke:#64748b,stroke-width:1.5px,color:#0f172a
    classDef process fill:#e0f2fe,stroke:#0369a1,stroke-width:1.5px,color:#0f172a
    classDef retrieval fill:#dcfce7,stroke:#15803d,stroke-width:1.5px,color:#0f172a
    classDef generation fill:#ede9fe,stroke:#7c3aed,stroke-width:1.5px,color:#0f172a
    classDef decision fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#0f172a
    classDef fallbackStyle fill:#fee2e2,stroke:#b91c1c,stroke-width:1.5px,color:#0f172a"""


# Índice de cada arista tal y como aparece en la plantilla (0-based).
# Lo usa `linkStyle <i>` de Mermaid para pintar aristas visitadas en rojo.
EDGE_INDEX: dict[tuple[str, str], int] = {
    ("start", "augment"): 0,
    ("augment", "route"): 1,
    ("route", "retrieve"): 2,
    ("retrieve", "grade"): 3,
    ("grade", "enrich"): 4,
    ("enrich", "generate"): 5,
    ("generate", "check"): 6,
    ("check", "end_node"): 7,
    ("route", "fallback"): 8,
    ("grade", "fallback"): 9,
    ("check", "fallback"): 10,
    ("check", "generate"): 11,
    ("fallback", "end_node"): 12,
}


def _build_mermaid(visited_ids: list[str]) -> str:
    """
    Devuelve markdown con la imagen del grafo renderizado (vía mermaid.ink).

    Resalta:
      - Los NODOS visitados con `style <id> fill:#ff6b6b…`
      - Las ARISTAS recorridas con `linkStyle <i> stroke:#ff6b6b…`
        reconstruyendo el camino start → visitados → finish y mapeando
        cada par consecutivo a su índice de arista en EDGE_INDEX.
    """
    # Nodos únicos visitados (orden preservado)
    seen_nodes: list[str] = []
    for nid in visited_ids:
        if nid and nid not in seen_nodes:
            seen_nodes.append(nid)

    node_styles = "\n".join(
        f"    style {nid} fill:#ff6b6b,stroke:#c92a2a,color:#fff,stroke-width:2px"
        for nid in seen_nodes
    )

    # Aristas visitadas: camino completo con entry y end
    path = ["start"] + [nid for nid in visited_ids] + ["end_node"]
    edge_indices: list[int] = []
    for a, b in zip(path, path[1:]):
        idx = EDGE_INDEX.get((a, b))
        if idx is not None and idx not in edge_indices:
            edge_indices.append(idx)
    edge_styles = ""
    if edge_indices:
        joined = ",".join(str(i) for i in edge_indices)
        edge_styles = (
            f"    linkStyle {joined} " "stroke:#ff6b6b,stroke-width:3px,color:#ff6b6b"
        )

    body = MERMAID_TEMPLATE
    extras = [x for x in (node_styles, edge_styles) if x]
    if extras:
        body = body + "\n" + "\n".join(extras)

    # mermaid.ink: base64 url-safe del código + scale=2 para nitidez
    b64 = base64.urlsafe_b64encode(body.encode("utf-8")).decode("ascii")

    # También añadimos fit para que el escalado sea correcto
    img_url = f"https://mermaid.ink/img/{b64}?width=1500&scale=1"

    # Retornamos la imagen con el link para que sea pinchable
    return f"[![Grafo del workflow]({img_url})]({img_url})"


# ──────────────────────────────────────────────────────────────────────────────


# ───────────── Helpers de UI / formato ─────────────
def _format_sources_md(documents: list[Document]) -> str:
    """Legacy: markdown con <details> (no renderiza bien en Chainlit)."""
    if not documents:
        return ""
    items = []
    for i, d in enumerate(documents, start=1):
        snippet = (
            (d.metadata or {}).get("summary") if isinstance(d.metadata, dict) else None
        )
        if not snippet:
            snippet = (d.page_content or "").strip()
        snippet = snippet.replace("\n", " ")[400:]
        items.append(f"- **Documento {i}**: {snippet}…")
    return (
        "<details>\n"
        "<summary>📄 Documentos consultados</summary>\n\n"
        + "\n".join(items)
        + "\n\n</details>"
    )


import re

# Verbos amigables por herramienta yfinance para el desplegable.
_TOOL_VERBS = {
    "get_quote": "Cotización de",
    "get_fundamentals": "Fundamentales de",
    "get_fund_info": "Información del fondo/ETF",
    "get_historical": "Histórico de",
    "get_news": "Noticias de",
}


def _extract_yf_long_name(first_line: str) -> str:
    """Saca el nombre largo ('T. Rowe Price Global Technology I', etc.) de la
    primera línea devuelta por las tools de yfinance. Devuelve "" si no hay."""
    # Cotización de NAME (TICKER):  |  Fundamentales de NAME:
    # Información del fondo/ETF NAME:  |  Últimas noticias para NAME:
    m = re.search(r"(?:de|ETF|para)\s+(.+?)\s*(?:\(|:)", first_line)
    if not m:
        return ""
    return m.group(1).strip()


def _format_sources_clean(documents: list[Document]) -> str:
    """
    Formato markdown del panel "Documentos consultados": dos secciones separadas.

    1. Documentos del corpus RAG (resumen ≤220 chars).
    2. Búsquedas en yfinance (resumen conciso ≤150 chars por entrada).
       Los resultados de `search_ticker` se ocultan: son plumbing interno.
    """
    if not documents:
        return ""

    # Pre-pass: construir ticker → nombre largo desde tools que devuelven longName.
    ticker_to_name: dict[str, str] = {}
    for d in documents:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        if meta.get("source") != "yfinance":
            continue
        tool = meta.get("tool") or ""
        if tool not in ("get_quote", "get_fundamentals", "get_fund_info"):
            continue
        args = meta.get("args") or {}
        ticker = args.get("ticker")
        if not ticker:
            continue
        body = (d.page_content or "").strip()
        if body.startswith("[yfinance"):
            body = body.split("\n", 1)[1] if "\n" in body else body
        first_line = body.split("\n", 1)[0]
        name = _extract_yf_long_name(first_line)
        if name and name.upper() != str(ticker).upper():
            ticker_to_name.setdefault(ticker, name)

    rag_items: list[str] = []
    yf_items: list[str] = []
    rag_n = 0

    for d in documents:
        meta = d.metadata if isinstance(d.metadata, dict) else {}
        source = meta.get("source")

        if source == "yfinance":
            tool = meta.get("tool") or ""
            args = meta.get("args") or {}

            if tool == "search_ticker":
                # Mostrar qué término se buscó y los candidatos encontrados.
                query = args.get("query") or ""
                body = (d.page_content or "").strip()
                if body.startswith("[yfinance"):
                    body = body.split("\n", 1)[1] if "\n" in body else body
                # Extraer tickers candidatos de las líneas "- SYM  (nombre, exch)"
                candidates = [
                    line.lstrip("- ").split()[0]
                    for line in body.splitlines()
                    if line.strip().startswith("-")
                ]
                candidates_str = (
                    ", ".join(candidates[:4]) if candidates else "sin resultados"
                )
                yf_items.append(
                    f"- 🔍 **Búsqueda «{query}»** → tickers encontrados: `{candidates_str}`"
                )
                continue

            ticker = args.get("ticker") or ""
            nice_name = ticker_to_name.get(ticker, "")

            verb = _TOOL_VERBS.get(tool, tool.replace("_", " ").capitalize())
            if nice_name and ticker:
                title = f"{verb} {nice_name} ({ticker})"
            elif ticker:
                title = f"{verb} {ticker}"
            else:
                title = verb
            if tool == "get_historical" and args.get("period"):
                title += f" — {args['period']}"

            body = (d.page_content or "").strip()
            if body.startswith("[yfinance"):
                body = body.split("\n", 1)[1] if "\n" in body else body
            snippet = body.replace("\n", " · ")
            if len(snippet) > 150:
                snippet = snippet[:150].rstrip() + "…"

            yf_items.append(f"- **{title}**: {snippet}")
        else:
            rag_n += 1
            snippet = meta.get("summary") or (d.page_content or "").strip()
            snippet = snippet.replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220].rstrip() + "…"
            rag_items.append(f"- **Documento {rag_n}**: {snippet}")

    parts: list[str] = []
    if rag_items:
        parts.append("**📚 Documentos del corpus**")
        parts.extend(rag_items)
    if yf_items:
        if parts:
            parts.append("")
        parts.append("**📊 Datos consultados en yfinance**")
        parts.extend(yf_items)

    return "\n".join(parts)


# --------------


# helper para acordeón
def _format_sources_html(documents: list[Document]) -> str:
    if not documents:
        return ""
    items = []
    for i, d in enumerate(documents, start=1):
        snippet = (
            (d.metadata or {}).get("summary") if isinstance(d.metadata, dict) else None
        )
        if not snippet:
            snippet = (d.page_content or "").strip()
        snippet = snippet.replace("\n", " ")[:220]
        items.append(f"<li><strong>Documento {i}</strong>: {snippet}…</li>")
    return (
        "<details style='margin-top:12px'>"
        "<summary><b>📄 Documentos consultados</b></summary>"
        "<div style='font-size:13px;color:#888;margin-top:8px'>"
        "<ul style='margin-left:18px'>" + "".join(items) + "</ul>"
        "</div>"
        "</details>"
    )


def _format_full_context_html(documents: list[Document], char_limit: int = 1500) -> str:
    """
    Acordeón con el texto fuente utilizado. Cada documento va en su propio <details>.
    Usa <pre> para mantener formato, y recorta a char_limit por documento para rendimiento.
    """
    if not documents:
        return ""
    blocks = []
    for i, d in enumerate(documents, start=1):
        # meta opcionales para mostrar
        title = None
        if isinstance(d.metadata, dict):
            title = (
                d.metadata.get("title")
                or d.metadata.get("source")
                or d.metadata.get("path")
            )
        title_txt = f" — {title}" if title else ""
        raw = d.page_content or ""
        truncated = raw[:char_limit]
        ellipsis = "… (recortado)" if len(raw) > char_limit else ""
        # escapar HTML para que no rompa el DOM
        escaped = html.escape(truncated)
        blocks.append(
            "<details style='margin:8px 0'>"
            f"<summary><b>Documento {i}{html.escape(title_txt)}</b></summary>"
            "<div style='margin-top:6px'>"
            f"<pre style='white-space:pre-wrap;font-size:13px;color:#666;background:#f7f7f7;padding:10px;border-radius:8px;'>{escaped}{ellipsis}</pre>"
            "</div>"
            "</details>"
        )
    return (
        "<details style='margin-top:12px'>"
        "<summary><b>🗂️ Contexto completo utilizado (texto fuente)</b></summary>"
        "<div style='margin-top:8px'>" + "".join(blocks) + "</div>"
        "</details>"
    )


# --------------


def _pairs_to_alternating_roles(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Convierte [(user, assistant), ...] a [("user", u1), ("assistant", a1), ...]
    y limita a los últimos 6 turnos (12 mensajes) para alinear con tu workflow.
    """
    alt: list[tuple[str, str]] = []
    for u, a in pairs[-6:]:
        alt.append(("user", u))
        alt.append(("assistant", a))
    return alt


# Compatibilidad con distintas versiones de Chainlit para actualizar mensajes
async def _safe_update_msg(msg: cl.Message, text: str):
    """
    Intenta msg.update(content=...) (APIs nuevas) y, si falla, usa
    msg.content = ...; await msg.update() (APIs antiguas).
    """
    try:
        await msg.update(content=text)
    except TypeError:
        msg.content = text
        await msg.update()


# ───────────── Tracker de nodos con contador en vivo ─────────────
class NodeTracker:
    """
    Gestiona la visualización del grafo LangGraph con UN solo mensaje en vivo.

    - Durante la ejecución, se muestra solo el nodo actual con contador.
    - Cada nodo finalizado se almacena internamente con su duración.
    - Al llamar `finalize()`, se elimina el mensaje en vivo y se devuelve la
      historia completa en markdown, lista para adjuntar como elemento lateral
      (desplegable) al mensaje final.
    """

    def __init__(self):
        self.current_msg: cl.Message | None = None
        self.current_id: str = ""  # ID canónico del nodo (augment, retrieve, ...)
        self.label_active: str = ""
        self.label_done: str = ""
        self.start_time: float = 0.0
        self.timer_task: asyncio.Task | None = None
        # history: lista de (node_id, label_done, elapsed_seconds)
        self.history: list[tuple[str, str, int]] = []
        self.overall_start: float | None = None  # inicio del primer nodo

    async def start_node(self, node_id: str, label_active: str, label_done: str):
        # Dedupe: si el nodo activo ya es este mismo, ignorar
        if self.label_active == label_active and self.current_msg is not None:
            return

        # Congelar la métrica del nodo anterior en `history` y parar su timer
        await self._archive_previous()

        # Marcar inicio global en el primer nodo (para el total)
        if self.overall_start is None:
            self.overall_start = time.time()

        # Arrancar el nuevo nodo
        self.current_id = node_id
        self.label_active = label_active
        self.label_done = label_done
        self.start_time = time.time()

        live_text = f"{label_active}… (0s)"
        if self.current_msg is None:
            # Primer nodo: crear el mensaje vivo
            self.current_msg = cl.Message(content=live_text, author=BOT_NAME)
            await self.current_msg.send()
        else:
            # Nodos siguientes: reutilizar el mismo mensaje
            await _safe_update_msg(self.current_msg, live_text)

        self.timer_task = asyncio.create_task(self._timer_loop())

    async def _timer_loop(self):
        try:
            while True:
                await asyncio.sleep(1)
                elapsed = int(time.time() - self.start_time)
                if self.current_msg is not None and self.label_active:
                    await _safe_update_msg(
                        self.current_msg,
                        f"{self.label_active}… ({elapsed}s)",
                    )
        except asyncio.CancelledError:
            pass

    async def _archive_previous(self):
        """Parar el timer y añadir la duración del nodo actual al histórico."""
        if self.timer_task and not self.timer_task.done():
            self.timer_task.cancel()
            try:
                await self.timer_task
            except asyncio.CancelledError:
                pass
        if self.label_active:
            elapsed = int(time.time() - self.start_time)
            self.history.append((self.current_id, self.label_done, elapsed))
        self.label_active = ""
        self.current_id = ""
        self.timer_task = None

    async def finalize(self) -> str:
        """
        Cierra el tracker: archiva el último nodo, elimina el mensaje en vivo
        y devuelve un bloque markdown con:
          - diagrama Mermaid del grafo con los nodos visitados resaltados
          - lista con tiempos por nodo
          - tiempo total
        """
        await self._archive_previous()
        if self.current_msg is not None:
            try:
                await self.current_msg.remove()
            except Exception:
                # Fallback: vaciar el contenido si remove() no está disponible
                await _safe_update_msg(self.current_msg, "")
            self.current_msg = None
        if not self.history:
            return ""

        # 1) Diagrama Mermaid con los nodos visitados resaltados
        visited_ids = [nid for (nid, _, _) in self.history]
        mermaid_md = _build_mermaid(visited_ids)

        # 2) Lista cronológica con tiempos
        bullet_lines = [
            f"- _{done} durante {elapsed}s_" for (_, done, elapsed) in self.history
        ]

        # 3) Tiempo total
        total_line = ""
        if self.overall_start is not None:
            total = int(time.time() - self.overall_start)
            total_line = f"**⏱️ Tiempo total de respuesta: {total}s**"

        parts = [
            "### 🗺️ Recorrido por el grafo",
            mermaid_md,
            "",
            "### ⏱️ Tiempos por nodo",
            *bullet_lines,
        ]
        if total_line:
            parts += ["", total_line]
        return "\n".join(parts)


# ───────────── Ciclo de vida de la app ─────────────

# Disclaimer regulatorio mostrado como panel lateral en el mensaje de bienvenida.
# El título debe aparecer textualmente en el `content` del mensaje para que
# Chainlit lo convierta en un enlace clicable al panel.
DISCLAIMER_TITLE = "⚠️ Aviso: finalidad educativa, no asesoramiento financiero"

DISCLAIMER_BODY = """
**Esto no es asesoramiento financiero personalizado.**

InversIA es un proyecto académico con fines exclusivamente educativos. Las
respuestas que ofrece **no constituyen asesoramiento de inversión, recomendación
personalizada ni servicio de planificación financiera** en el sentido del
artículo 4 de la Directiva MiFID II. La prestación de servicios de
asesoramiento individualizado en materia de instrumentos financieros está
reservada en España a entidades autorizadas por la CNMV.

**Limitaciones que conviene tener presentes**

- El corpus sobre el que opera el asistente tiene un sesgo deliberado hacia la
  filosofía de inversión pasiva a largo plazo. No es neutral entre distintas
  escuelas de inversión.
- Aunque el sistema reduce las alucinaciones mediante anclaje en fuentes
  verificables, no las elimina por completo. Verifica siempre la información
  fiscal, normativa o de producto antes de tomar cualquier decisión.
- Para decisiones de inversión reales consulta a un profesional autorizado.

**Datos personales**

El sistema no almacena el historial entre sesiones; la conversación se descarta
al cerrar la ventana del navegador. No existe tratamiento de datos personales
en el sentido del artículo 4.1 del RGPD más allá de la sesión activa.
"""

if DEFAULT_LLM == "deepseek":
    llm_displayed_name = "DeepSeek Chat"
else:
    llm_displayed_name = "ChatGPT-3.5"

@cl.on_chat_start
async def on_chat_start():
    # Historial persistente por sesión: lista de pares (user, assistant)
    cl.user_session.set("history_pairs", [])

    await cl.Message(
        author=BOT_NAME,
        content=(
            "👋 **Bienvenido a InversIA**\n\n"
            f"Este asistente usa **{llm_displayed_name}** y un flujo RAG con "
            "validaciones.\n"
            "_Nota:_ la primera respuesta puede tardar unos segundos por la "
            "carga del vectorstore."
            "\n\n"
            "**AVISO**: InversIA no es una plataforma de asesoramiento financiero personalizado, "
            "es un proyecto con fines exclusivamente académicos."
        ),
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    # Recupera historial guardado en la sesión
    history_pairs: list[tuple[str, str]] = cl.user_session.get("history_pairs") or []
    chat_history_pairs = _pairs_to_alternating_roles(history_pairs)

    # Tracker de nodos (un mensaje por nodo con contador en vivo)
    tracker = NodeTracker()

    # Estado inicial para el grafo
    initial_state = {
        "question": message.content,
        "augmented_question": "",
        "generation": "",
        "web_search": "No",
        "documents": [],
        "step_count": 0,
        "source_type": "",
        "chat_history": chat_history_pairs,
        "modelo_llm": DEFAULT_LLM,
    }

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    final_result = None
    sentinel = object()

    # Worker: ejecuta workflow y redirige prints
    def _worker():
        nonlocal final_result
        writer = _QueueWriter(loop, queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                final_result = langgraph_app.invoke(initial_state)
        finally:
            writer.flush()
            asyncio.run_coroutine_threadsafe(queue.put(sentinel), loop)

    # Lanza el worker en un hilo
    fut = loop.run_in_executor(None, _worker)

    # Consumir los prints para actualizar el estado del nodo en vivo
    trace_md: str = ""
    try:
        while True:
            line = await queue.get()
            if line is sentinel:
                break

            if "AUMENTANDO LA PREGUNTA" in line:
                await tracker.start_node(
                    "augment", "🔍 Aumentando pregunta", "🔍 Aumentó pregunta"
                )
            elif "RUTEANDO LA PREGUNTA" in line:
                await tracker.start_node(
                    "route", "🧭 Dirigiendo la pregunta", "🧭 Dirigió pregunta"
                )
            elif "RECUPERANDO DOCUMENTOS" in line:
                await tracker.start_node(
                    "retrieve",
                    "📚 Extrayendo fuentes de información",
                    "📚 Extrajo fuentes de información",
                )
            elif "EVALUANDO Y ORDENANDO" in line:
                await tracker.start_node(
                    "grade",
                    "🧪 Evaluando fuentes de información",
                    "🧪 Evaluó fuentes de información",
                )
            elif "ENRIQUECIENDO CON DATOS DE MERCADO" in line:
                await tracker.start_node(
                    "enrich",
                    "📊 Recogiendo información del mercado",
                    "📊 Recogió información del mercado",
                )
            elif (
                "PREGUNTA FUERA DE ALCANCE" in line
                or "BÚSQUEDA WEB" in line
                or "BUSQUEDA WEB" in line
            ):
                await tracker.start_node(
                    "fallback",
                    "❌ Pregunta no relacionada con el propósito, generando respuesta alternativa",
                    "❌ Generó respuesta alternativa",
                )
            elif "GENERANDO RESPUESTA" in line:
                await tracker.start_node(
                    "generate", "✍️ Generando respuesta", "✍️ Generó respuesta"
                )
            elif "COMPROBANDO" in line:
                await tracker.start_node(
                    "check", "🧠 Comprobando respuesta", "🧠 Comprobó respuesta"
                )

        # Espera a que termine el worker
        await fut
    finally:
        # Cerrar el tracker siempre (éxito o error) y recoger la trazabilidad
        trace_md = await tracker.finalize()

    # Construir respuesta final
    result = final_result or {}

    response = (result.get("generation") or "").strip()
    documents = result.get("documents") or []

    # Preparar elementos adjuntos como paneles laterales desplegables.
    # IMPORTANTE: para display="side", el `name` del elemento debe aparecer
    # literalmente dentro del contenido del mensaje para que Chainlit lo
    # convierta en un enlace clicable.
    elements = []
    refs: list[str] = []

    if documents:
        sources_md = _format_sources_clean(documents)
        docs_name = "📄 Documentos consultados"
        elements.append(cl.Text(name=docs_name, content=sources_md, display="side"))
        refs.append(docs_name)

    if trace_md:
        trace_name = "🕑 Trazabilidad del grafo"
        elements.append(cl.Text(name=trace_name, content=trace_md, display="side"))
        refs.append(trace_name)

    # Disclaimer regulatorio: chip presente en todas las respuestas para que el
    # usuario pueda revisarlo en cualquier momento de la conversación.
    elements.append(
        cl.Text(name=DISCLAIMER_TITLE, content=DISCLAIMER_BODY, display="side")
    )
    refs.append(DISCLAIMER_TITLE)

    final_content = response
    if refs:
        final_content = f"{response}\n\n---\n" + " · ".join(refs)

    # Enviar la respuesta principal con los elementos adjuntos
    await cl.Message(
        content=final_content,
        elements=elements,
        author=BOT_NAME,
    ).send()

    # Persistir el historial para los próximos turnos.
    # Guardamos solo la respuesta "pura" (sin el bloque de refs) para que el
    # historial que recibe el workflow no se contamine con texto de UI.
    if response:
        history_pairs.append((message.content, response))
        cl.user_session.set("history_pairs", history_pairs)