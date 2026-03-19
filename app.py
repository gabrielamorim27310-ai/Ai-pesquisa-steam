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

from agent import search_academic_papers, format_citation, SYSTEM_PROMPT

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
    """Busca em paralelo em português e inglês e combina os resultados."""
    results_pt: list = []
    results_en: list = []

    def search_pt():
        return search_academic_papers(query=query_pt, max_results=5)

    def search_en():
        return search_academic_papers(query=query_en, max_results=5)

    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_pt = executor.submit(search_pt)
        fut_en = executor.submit(search_en)
        for fut in as_completed([fut_pt, fut_en]):
            res = fut.result()
            if res and "error" not in res[0]:
                if fut is fut_pt:
                    results_pt = res
                else:
                    results_en = res

    combined = _deduplicate(results_pt + results_en)
    # Ordena por citações decrescentes e limita a 6
    combined.sort(key=lambda x: x.get("citations", 0), reverse=True)
    return combined[:6]


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

    emit("status", {"text": "Buscando artigos em português e inglês..."})

    # ── Passo 1: Busca paralela PT + EN ───────────────────────────
    query_en = user_message + " research"
    emit("tool", {"name": "search", "label": f'Buscando (PT): "{user_message}"'})
    emit("tool", {"name": "search", "label": f'Buscando (EN): "{query_en}"'})
    papers = _parallel_search(user_message, query_en)

    # ── Passo 2: Citações em Python ───────────────────────────────
    if papers:
        emit("status", {"text": f"Formatando {len(papers)} citações ({citation_style})..."})
        citations = []
        for p in papers:
            emit("tool", {
                "name": "citation",
                "label": f"Formatando {citation_style}: {p.get('title', '')[:50]}…",
            })
            citations.append(format_citation(p, style=citation_style))
    else:
        citations = []

    # ── Passo 3: Groq gera o resumo ──────────────────────────────
    emit("status", {"text": "Gerando análise com Groq Llama 3.3 70B..."})

    papers_json = json.dumps(papers, ensure_ascii=False, indent=2) if papers else "[]"
    citations_text = "\n\n".join(f"{i+1}. {c}" for i, c in enumerate(citations))

    if papers:
        sources_block = f"""Artigos encontrados (fontes confiáveis — Semantic Scholar):
{papers_json}

Citações já formatadas ({citation_style}):
{citations_text}"""
        instruction = (
            "Escreva uma resposta estruturada em português brasileiro com:\n"
            "1. Breve introdução sobre o tema (2-3 frases)\n"
            "2. Para cada artigo: título em negrito, autores, ano, resumo curto e relevância\n"
            "3. Seção \"Referências\" com as citações acima (copie exatamente)"
        )
    else:
        sources_block = "Nenhum artigo foi encontrado nas bases de dados para este tema."
        instruction = (
            "Mesmo sem artigos nas bases, escreva em português brasileiro:\n"
            "1. Uma explicação geral sobre o tema com base no seu conhecimento\n"
            "2. Indique que não foram encontrados artigos indexados para este tema específico\n"
            "3. Sugira termos de busca alternativos que o usuário pode tentar"
        )

    user_prompt = f"""O usuário pesquisou: "{user_message}"

{sources_block}

{instruction}"""

    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4096,
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
