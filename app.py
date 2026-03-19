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

from agent import search_academic_papers, search_wikipedia, format_citation, SYSTEM_PROMPT

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


def _parallel_search(query_pt: str, query_en: str) -> tuple[list[dict], list[dict]]:
    """
    Busca em paralelo em 4 fontes confiáveis:
      - Semantic Scholar (PT query + EN query)
      - Wikipedia PT
      - Wikipedia EN
    Retorna (papers, web_sources) já deduplicados.
    """
    buckets: dict = {"papers_pt": [], "papers_en": [], "wiki_pt": [], "wiki_en": []}

    tasks = {
        "papers_pt": lambda: search_academic_papers(query=query_pt, max_results=5),
        "papers_en": lambda: search_academic_papers(query=query_en, max_results=5),
        "wiki_pt":   lambda: search_wikipedia(query=query_pt, lang="pt", max_results=3),
        "wiki_en":   lambda: search_wikipedia(query=query_en, lang="en", max_results=3),
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

    papers = _deduplicate(buckets["papers_pt"] + buckets["papers_en"])
    papers.sort(key=lambda x: x.get("citations", 0), reverse=True)
    papers = papers[:6]

    web = _deduplicate(buckets["wiki_pt"] + buckets["wiki_en"])
    web = web[:4]

    return papers, web


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

    emit("status", {"text": "Buscando em fontes confiáveis (artigos + Wikipedia)..."})

    # ── Passo 1: Busca paralela (Semantic Scholar + Wikipedia PT/EN) ──
    query_en = user_message + " research"
    emit("tool", {"name": "search", "label": f'Semantic Scholar (PT): "{user_message}"'})
    emit("tool", {"name": "search", "label": f'Semantic Scholar (EN): "{query_en}"'})
    emit("tool", {"name": "search", "label": f'Wikipedia PT + EN: "{user_message}"'})
    papers, web_sources = _parallel_search(user_message, query_en)

    all_sources = papers + web_sources

    # ── Passo 2: Citações em Python ───────────────────────────────
    citations = []
    if all_sources:
        emit("status", {"text": f"Formatando {len(all_sources)} citações ({citation_style})..."})
        for s in all_sources:
            emit("tool", {
                "name": "citation",
                "label": f"Formatando {citation_style}: {s.get('title', '')[:50]}…",
            })
            citations.append(format_citation(s, style=citation_style))

    # ── Passo 3: Groq gera o resumo ──────────────────────────────
    emit("status", {"text": "Gerando análise com Groq Llama 3.3 70B..."})

    citations_text = "\n\n".join(f"{i+1}. {c}" for i, c in enumerate(citations))

    if all_sources:
        papers_block = (
            f"Artigos científicos (Semantic Scholar):\n{json.dumps(papers, ensure_ascii=False, indent=2)}"
            if papers else "Nenhum artigo científico encontrado."
        )
        web_block = (
            f"Fontes enciclopédicas (Wikipedia PT/EN):\n{json.dumps(web_sources, ensure_ascii=False, indent=2)}"
            if web_sources else "Nenhuma fonte Wikipedia encontrada."
        )
        sources_block = f"{papers_block}\n\n{web_block}\n\nCitações já formatadas ({citation_style}):\n{citations_text}"
        instruction = (
            "Escreva uma resposta estruturada em português brasileiro com:\n"
            "1. Breve introdução sobre o tema (2-3 frases)\n"
            "2. Para cada fonte: título em negrito, origem (artigo/Wikipedia), "
            "autores/instituição, ano, resumo curto e relevância\n"
            "3. Seção \"Referências\" com as citações acima (copie exatamente)"
        )
    else:
        sources_block = "Nenhuma fonte foi encontrada (Semantic Scholar nem Wikipedia)."
        instruction = (
            "Mesmo sem fontes externas, escreva em português brasileiro:\n"
            "1. Uma explicação geral sobre o tema com base no seu conhecimento\n"
            "2. Indique que não foram encontradas fontes indexadas para este tema\n"
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
