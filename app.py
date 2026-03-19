"""
Servidor web para o Agente de Pesquisa Acadêmica.
Usa Groq (Llama 3.3 70B) para geração de texto - gratuito.
"""

import os
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests as http_requests
from flask import Flask, render_template, request, Response, stream_with_context, jsonify

from agent import search_academic_papers, search_openalex, search_arxiv, format_citation, SYSTEM_PROMPT

app = Flask(__name__)


# ─────────────────────────────────────────────
# Rota principal
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def _deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicatas por DOI (prioritário) ou título normalizado."""
    seen_doi: set = set()
    seen_title: set = set()
    result = []
    for p in papers:
        doi = (p.get("doi") or "").strip()
        title_key = (p.get("title") or "").strip().lower()[:60]
        if doi and doi in seen_doi:
            continue
        if title_key and title_key in seen_title:
            continue
        if doi:
            seen_doi.add(doi)
        if title_key:
            seen_title.add(title_key)
        result.append(p)
    return result


def _parallel_search(query_pt: str, query_en: str) -> list[dict]:
    """
    Busca em paralelo em 3 fontes acadêmicas confiáveis:
      - Semantic Scholar  (query PT + EN)
      - OpenAlex          (grandes universidades mundiais, query EN)
      - arXiv / Cornell   (query EN)
    Retorna lista única deduplicada e ordenada por citações.
    """
    buckets: dict = {"ss_pt": [], "ss_en": [], "openalex": [], "arxiv": []}

    tasks = {
        "ss_pt":    lambda: search_academic_papers(query=query_pt, max_results=5),
        "ss_en":    lambda: search_academic_papers(query=query_en, max_results=5),
        "openalex": lambda: search_openalex(query=query_en, max_results=5),
        "arxiv":    lambda: search_arxiv(query=query_en, max_results=4),
    }

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): key for key, fn in tasks.items()}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                res = fut.result()
                if res and (not isinstance(res[0], dict) or "error" not in res[0]):
                    buckets[key] = res
            except Exception:
                pass

    combined = _deduplicate(
        buckets["ss_pt"] + buckets["ss_en"] + buckets["openalex"] + buckets["arxiv"]
    )
    combined.sort(key=lambda x: x.get("citations", 0), reverse=True)
    return combined[:5]


def run_pipeline(user_message: str, citation_style: str = "ABNT") -> list[dict]:
    """
    Pipeline otimizado para Vercel (< 10s):
      1. Busca Semantic Scholar em PT e EN em paralelo (~1s)
      2. Formata citações em Python (instantâneo)
      3. Groq Llama 3.3 70B gera o resumo (~1-2s, 1 chamada)
    """
    events: list[dict] = []

    def emit(event_type: str, data: dict):
        events.append({"type": event_type, "data": data})

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        emit("error", {"message": "GROQ_API_KEY não configurada no servidor."})
        return events

    emit("status", {"text": "Buscando em Semantic Scholar, OpenAlex e arXiv..."})

    # ── Passo 1: Busca paralela (3 fontes acadêmicas) ─────────────
    query_en = user_message + " research"
    emit("tool", {"name": "search", "label": f'Semantic Scholar: "{user_message}"'})
    emit("tool", {"name": "search", "label": f'OpenAlex (universidades): "{query_en}"'})
    emit("tool", {"name": "search", "label": f'arXiv / Cornell University: "{query_en}"'})
    sources = _parallel_search(user_message, query_en)

    # ── Passo 2: Citações em Python ───────────────────────────────
    citations = []
    if sources:
        emit("status", {"text": f"Formatando {len(sources)} citações ({citation_style})..."})
        for s in sources:
            origin = {
                "openalex": "OpenAlex",
                "arxiv": "arXiv/Cornell",
            }.get(s.get("source_type", ""), "Semantic Scholar")
            emit("tool", {
                "name": "citation",
                "label": f"[{origin}] {s.get('title', '')[:45]}…",
            })
            citations.append(format_citation(s, style=citation_style))

    # ── Passo 3: Groq gera o resumo ──────────────────────────────
    emit("status", {"text": "Gerando análise com Groq Llama 3.3 70B..."})

    citations_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(citations))

    if sources:
        # Payload slim: apenas campos essenciais + abstract truncado a 180 chars
        slim = [
            {
                "titulo": s.get("title", ""),
                "autores": ", ".join(s.get("authors", [])[:3]),
                "ano": s.get("year", ""),
                "fonte": s.get("journal", ""),
                "origem": {"openalex": "OpenAlex", "arxiv": "arXiv/Cornell"}.get(
                    s.get("source_type", ""), "Semantic Scholar"
                ),
                "resumo": (s.get("abstract") or "")[:180],
            }
            for s in sources
        ]
        sources_block = (
            f"Fontes ({len(slim)}):\n"
            + json.dumps(slim, ensure_ascii=False)
            + f"\n\nCitações ({citation_style}):\n{citations_text}"
        )
        instruction = (
            "Responda em português brasileiro:\n"
            "1. Introdução sobre o tema (2 frases)\n"
            "2. Cada fonte: **título**, origem, autores, ano, resumo e relevância\n"
            "3. Seção Referências com as citações acima (copie exatamente)"
        )
    else:
        sources_block = "Nenhuma fonte encontrada nas bases acadêmicas."
        instruction = (
            "Responda em português brasileiro:\n"
            "1. Explicação geral sobre o tema\n"
            "2. Informe que não foram encontradas fontes indexadas\n"
            "3. Sugira termos de busca alternativos"
        )

    user_prompt = f'Pesquisa: "{user_message}"\n\n{sources_block}\n\n{instruction}'

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        resp = http_requests.post(url, json=payload, headers=headers, timeout=20)
        if resp.status_code != 200:
            body = resp.json()
            err = body.get("error", {}).get("message", resp.text)
            emit("error", {"message": f"Erro Groq ({resp.status_code}): {err}"})
            return events
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        emit("result", {"text": text})

    except Exception as e:
        emit("error", {"message": f"Erro Groq: {str(e)}"})

    return events


def _parse_style(query: str) -> tuple[str, str]:
    for style in ("ABNT", "APA", "MLA", "STEAM"):
        if query.upper().endswith(f" {style}"):
            return query[: -(len(style) + 1)].strip(), style
    return query, "ABNT"


# ─────────────────────────────────────────────
# Endpoint SSE (local)
# ─────────────────────────────────────────────

@app.route("/pesquisar", methods=["POST"])
def pesquisar():
    data = request.get_json(silent=True) or {}
    raw_query = (data.get("query") or "").strip()
    if not raw_query:
        return {"error": "Pesquisa vazia."}, 400

    user_message, style = _parse_style(raw_query)
    q: queue.Queue = queue.Queue()

    def worker():
        evts = run_pipeline(user_message, style)
        for e in evts:
            payload = json.dumps(e["data"], ensure_ascii=False)
            q.put(f"event: {e['type']}\ndata: {payload}\n\n")
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = q.get()
            if item is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield item

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─────────────────────────────────────────────
# Endpoint síncrono (Vercel)
# ─────────────────────────────────────────────

@app.route("/pesquisar-sync", methods=["POST"])
def pesquisar_sync():
    data = request.get_json(silent=True) or {}
    raw_query = (data.get("query") or "").strip()
    if not raw_query:
        return jsonify({"error": "Pesquisa vazia."}), 400

    user_message, style = _parse_style(raw_query)
    events = run_pipeline(user_message, style)
    return jsonify({"events": events})


# ─────────────────────────────────────────────
# Início local
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Servidor iniciado em http://localhost:{port}\n")
    app.run(debug=False, port=port, threaded=True)
