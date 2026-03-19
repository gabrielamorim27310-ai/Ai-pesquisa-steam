"""
Servidor web para o Agente de Pesquisa Acadêmica.

- Local:  usa SSE (Server-Sent Events) para streaming em tempo real.
- Vercel: usa endpoint síncrono /pesquisar-sync (serverless não suporta SSE).
  O frontend detecta automaticamente o ambiente.
"""

import os
import json
import queue
import threading
import anthropic
from flask import Flask, render_template, request, Response, stream_with_context, jsonify

from agent import execute_tool, TOOLS, SYSTEM_PROMPT

app = Flask(__name__)


# ─────────────────────────────────────────────
# Rota principal
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# Lógica central do agente (compartilhada)
# ─────────────────────────────────────────────

def run_agent_collecting_events(user_message: str) -> list[dict]:
    """
    Executa o agente e coleta todos os eventos numa lista.
    Usado pelo endpoint síncrono e pelo SSE.
    """
    events: list[dict] = []

    def emit(event_type: str, data: dict):
        events.append({"type": event_type, "data": data})

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        emit("error", {"message": "ANTHROPIC_API_KEY não configurada no servidor."})
        return events

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    emit("status", {"text": "Analisando sua pesquisa..."})

    try:
        while True:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        emit("result", {"text": block.text})
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        if tool_name == "search_academic_papers":
                            emit("tool", {
                                "name": "search",
                                "label": f"Buscando: \"{tool_input.get('query', '')}\"",
                            })
                        elif tool_name == "format_citation":
                            title = tool_input.get("article", {}).get("title", "")
                            style = tool_input.get("style", "ABNT")
                            emit("tool", {
                                "name": "citation",
                                "label": f"Formatando citação {style}: {title[:50]}…",
                            })

                        result = execute_tool(tool_name, tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                messages.append({"role": "user", "content": tool_results})

            else:
                for block in response.content:
                    if block.type == "text":
                        emit("result", {"text": block.text})
                break

    except anthropic.AuthenticationError:
        emit("error", {"message": "Chave de API inválida. Verifique ANTHROPIC_API_KEY."})
    except anthropic.RateLimitError:
        emit("error", {"message": "Limite de requisições atingido. Aguarde alguns segundos."})
    except anthropic.APIConnectionError as e:
        emit("error", {"message": f"Não foi possível conectar à API da Anthropic. Tente novamente. ({str(e)})"})
    except anthropic.APIStatusError as e:
        emit("error", {"message": f"Erro da API ({e.status_code}): {e.message}"})
    except Exception as e:
        emit("error", {"message": f"Erro interno: {type(e).__name__}: {str(e)}"})

    return events


# ─────────────────────────────────────────────
# Endpoint SSE — para uso local (streaming real)
# ─────────────────────────────────────────────

@app.route("/pesquisar", methods=["POST"])
def pesquisar():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("query") or "").strip()
    if not user_message:
        return {"error": "Pesquisa vazia."}, 400

    q: queue.Queue = queue.Queue()

    def worker():
        evts = run_agent_collecting_events(user_message)
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
# Endpoint síncrono — para Vercel (retorna JSON)
# ─────────────────────────────────────────────

@app.route("/pesquisar-sync", methods=["POST"])
def pesquisar_sync():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("query") or "").strip()
    if not user_message:
        return jsonify({"error": "Pesquisa vazia."}), 400

    events = run_agent_collecting_events(user_message)
    return jsonify({"events": events})


# ─────────────────────────────────────────────
# Início local
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Servidor iniciado em http://localhost:{port}\n")
    app.run(debug=False, port=port, threaded=True)
