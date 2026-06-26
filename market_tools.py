"""
Herramientas de mercado basadas en yfinance.

Se usan desde el nodo `market_data_enrichment` de `app_workflow.py`:
el LLM, con `bind_tools(MARKET_TOOLS)`, decide si hace falta invocarlas
y con qué argumentos. El resultado se añade a `state["documents"]` como
`Document(metadata={"source": "yfinance", ...})` para que entre como
contexto adicional al paso de generación.
"""

from __future__ import annotations

from typing import Optional

import yfinance as yf
from langchain_core.tools import tool


# ───────────────────────── helpers internos ─────────────────────────


def _safe_info(ticker: str) -> dict:
    """`yf.Ticker(...).info` puede fallar con símbolos inválidos o rate limits."""
    try:
        return yf.Ticker(ticker).info or {}
    except Exception:
        return {}


def _fmt_num(val, pct: bool = False) -> str:
    if val is None:
        return "n/d"
    try:
        if pct:
            return f"{float(val) * 100:.2f}%" if abs(float(val)) < 1 else f"{float(val):.2f}%"
        if isinstance(val, (int, float)):
            if abs(val) >= 1_000_000_000:
                return f"{val/1_000_000_000:.2f}B"
            if abs(val) >= 1_000_000:
                return f"{val/1_000_000:.2f}M"
            return f"{val:,.2f}"
    except Exception:
        pass
    return str(val)


# ───────────────────────────── tools ─────────────────────────────


@tool
def get_quote(ticker: str) -> str:
    """
    Devuelve la cotización actual de un instrumento (acción, ETF, fondo cotizado,
    índice o divisa). El parámetro `ticker` debe ser el símbolo bursátil, por ejemplo
    "AAPL" (Apple), "VOO" (Vanguard S&P 500 ETF), "^GSPC" (S&P 500) o "EURUSD=X".
    Incluye precio actual, cambio diario, rango del día y volumen.
    """
    info = _safe_info(ticker)
    if not info:
        return f"No se han podido obtener datos para «{ticker}»."
    nombre = info.get("longName") or info.get("shortName") or ticker
    precio = info.get("regularMarketPrice") or info.get("currentPrice")
    moneda = info.get("currency", "")
    cambio_pct = info.get("regularMarketChangePercent")
    volumen = info.get("regularMarketVolume")
    maximo = info.get("dayHigh")
    minimo = info.get("dayLow")
    apertura = info.get("regularMarketOpen")

    lines = [f"Cotización de {nombre} ({ticker}):"]
    if precio is not None:
        lines.append(f"- Precio actual: {precio} {moneda}")
    if apertura is not None:
        lines.append(f"- Apertura: {apertura} {moneda}")
    if cambio_pct is not None:
        lines.append(f"- Cambio diario: {float(cambio_pct):+.2f}%")
    if maximo is not None and minimo is not None:
        lines.append(f"- Rango del día: {minimo} — {maximo} {moneda}")
    if volumen is not None:
        lines.append(f"- Volumen: {volumen:,}")
    return "\n".join(lines)


@tool
def get_fundamentals(ticker: str) -> str:
    """
    Devuelve los fundamentales de una acción: PER, PER forward, BPA, capitalización
    bursátil, sector, industria, rentabilidad por dividendo, beta y margen de beneficio.
    Útil para análisis de valoración de acciones.
    """
    info = _safe_info(ticker)
    if not info:
        return f"No se han podido obtener fundamentales para «{ticker}»."
    lines = [f"Fundamentales de {info.get('longName') or ticker}:"]
    mapping: list[tuple[str, str, bool]] = [
        ("PER (trailing)", "trailingPE", False),
        ("PER (forward)", "forwardPE", False),
        ("BPA (EPS)", "trailingEps", False),
        ("Capitalización", "marketCap", False),
        ("Sector", "sector", False),
        ("Industria", "industry", False),
        ("Rentabilidad por dividendo", "dividendYield", True),
        ("Beta", "beta", False),
        ("Margen de beneficio", "profitMargins", True),
        ("País", "country", False),
    ]
    for label, key, is_pct in mapping:
        val = info.get(key)
        if val is not None:
            lines.append(f"- {label}: {_fmt_num(val, pct=is_pct)}")
    return "\n".join(lines)


@tool
def get_fund_info(ticker: str) -> str:
    """
    Devuelve información de un ETF o fondo cotizado: categoría, activos totales,
    TER anual, yield, NAV, familia del fondo y top holdings si están disponibles.
    Usa esta tool cuando el usuario pregunta por un ETF o fondo, no una acción.
    """
    t = yf.Ticker(ticker)
    info = _safe_info(ticker)
    if not info:
        return f"No se ha podido recuperar información del fondo/ETF «{ticker}»."
    lines = [f"Información del fondo/ETF {info.get('longName') or ticker}:"]
    mapping: list[tuple[str, str, bool]] = [
        ("Categoría", "category", False),
        ("Tipo", "quoteType", False),
        ("Activos totales", "totalAssets", False),
        ("TER anual", "annualReportExpenseRatio", True),
        ("Yield", "yield", True),
        ("NAV", "navPrice", False),
        ("Familia de fondos", "fundFamily", False),
    ]
    for label, key, is_pct in mapping:
        val = info.get(key)
        if val is not None:
            lines.append(f"- {label}: {_fmt_num(val, pct=is_pct)}")
    try:
        funds_data = getattr(t, "funds_data", None)
        if funds_data is not None:
            holdings = getattr(funds_data, "top_holdings", None)
            if holdings is not None and len(holdings) > 0:
                lines.append("Top holdings:")
                for idx, row in holdings.head(10).iterrows():
                    pct = row.get("Holding Percent") if hasattr(row, "get") else None
                    lines.append(f"  · {idx}: {_fmt_num(pct, pct=True)}")
    except Exception:
        pass
    return "\n".join(lines)


@tool
def get_historical(ticker: str, period: str = "1mo") -> str:
    """
    Devuelve un resumen de precios históricos para `ticker` en el periodo indicado.
    Valores válidos para `period`: "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max".
    Útil para responder a preguntas sobre rentabilidad pasada o evolución reciente.
    """
    try:
        hist = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        return f"Error obteniendo histórico: {e}"
    if hist is None or hist.empty:
        return f"No hay datos históricos para «{ticker}» en periodo «{period}»."
    precio_ini = float(hist["Close"].iloc[0])
    precio_fin = float(hist["Close"].iloc[-1])
    variacion = (precio_fin - precio_ini) / precio_ini * 100 if precio_ini else 0.0
    return (
        f"Precios históricos de {ticker} ({period}):\n"
        f"- Inicio: {precio_ini:.2f}\n"
        f"- Fin: {precio_fin:.2f}\n"
        f"- Variación: {variacion:+.2f}%\n"
        f"- Máximo: {float(hist['Close'].max()):.2f}\n"
        f"- Mínimo: {float(hist['Close'].min()):.2f}\n"
        f"- Sesiones: {len(hist)}"
    )


@tool
def get_news(ticker: str, limit: int = 5) -> str:
    """
    Devuelve las últimas noticias relacionadas con un ticker.
    """
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as e:
        return f"Error obteniendo noticias: {e}"
    news = news[: max(1, int(limit))]
    if not news:
        return f"No hay noticias recientes para «{ticker}»."
    out = [f"Últimas noticias para {ticker}:"]
    for n in news:
        # yfinance ha cambiado varias veces el esquema de news; soportamos los dos.
        content = n.get("content") if isinstance(n, dict) and "content" in n else n
        title = (content or {}).get("title", "(sin título)")
        provider = (
            (content or {}).get("provider", {}).get("displayName")
            or (content or {}).get("publisher", "")
        )
        url = (
            (content or {}).get("canonicalUrl", {}).get("url")
            or (content or {}).get("link", "")
        )
        out.append(f"- {title} [{provider}]  {url}")
    return "\n".join(out)


@tool
def search_ticker(query: str) -> str:
    """
    Busca tickers candidatos a partir de un nombre aproximado.
    Ej: "Apple" → AAPL, "Vanguard S&P 500" → VOO / VUSA.L.
    Usa esta tool antes que otras si el usuario menciona una empresa o fondo por
    nombre y no por ticker.
    """
    try:
        from yfinance import Search  # type: ignore

        results = Search(query, max_results=5).quotes
    except Exception as e:
        return f"Error buscando ticker para «{query}»: {e}"
    if not results:
        return f"No se han encontrado tickers para «{query}»."
    out = [f"Candidatos para «{query}»:"]
    for r in results:
        sym = r.get("symbol")
        name = r.get("shortname") or r.get("longname") or ""
        exch = r.get("exchange", "")
        out.append(f"- {sym}  ({name}, {exch})")
    return "\n".join(out)


MARKET_TOOLS = [
    get_quote,
    get_fundamentals,
    get_fund_info,
    get_historical,
    get_news,
    search_ticker,
]

TOOLS_BY_NAME = {t.name: t for t in MARKET_TOOLS}
