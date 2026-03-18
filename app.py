"""
Servidor web para o Agente de Pesquisa Acadêmica.
Usa Server-Sent Events (SSE) para transmitir o progresso em tempo real.
"""

import os
import json
import queue
import threading
import anthropic
from flask import Flask, render_template, request, Response, stream_with_context

from agent import (
    search_academic_papers,
    format_citation,
    execute_tool,
    TOOLS,
    SYSTEM_PROMPT,
)

app = Flask(__name__)


# ─────────────────────────────────────────────
# Rota principal
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ─────────────────────────────────────────────
# Endpoint de pesquisa com streaming SSE
# ─────────────────────────────────────────────

def run_agent_stream(user_message: str, event_queue: queue.Queue):
    """Executa o agente e coloca eventos SSE na fila."""

    def send(event_type: str, data: str | dict):
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False)
        event_queue.put(f"event: {event_type}\ndata: {data}\n\n")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        send("error", {"message": "ANTHROPIC_API_KEY não configurada no servidor."})
        event_queue.put(None)
        return

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    send("status", {"text": "Analisando sua pesquisa..."})

    try:
        while True:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=8192,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            # Resposta final — envia o texto completo
            if response.stop_reason == "end_turn":
                for block in response.content:
                    if block.type == "text":
                        send("result", {"text": block.text})
                break

            # Chamadas de ferramenta
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        # Notifica o frontend qual ferramenta está sendo usada
                        if tool_name == "search_academic_papers":
                            send("tool", {
                                "name": "search",
                                "label": f"Buscando: \"{tool_input.get('query', '')}\"",
                            })
                        elif tool_name == "format_citation":
                            title = tool_input.get("article", {}).get("title", "")
                            style = tool_input.get("style", "ABNT")
                            send("tool", {
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
                # Stop reason inesperado
                for block in response.content:
                    if block.type == "text":
                        send("result", {"text": block.text})
                break

    except anthropic.AuthenticationError:
        send("error", {"message": "Chave de API inválida. Verifique ANTHROPIC_API_KEY."})
    except anthropic.RateLimitError:
        send("error", {"message": "Limite de requisições atingido. Aguarde alguns segundos."})
    except Exception as e:
        send("error", {"message": f"Erro interno: {str(e)}"})
    finally:
        event_queue.put(None)  # Sinal de fim


@app.route("/pesquisar", methods=["POST"])
def pesquisar():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("query") or "").strip()

    if not user_message:
        return {"error": "Pesquisa vazia."}, 400

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=run_agent_stream, args=(user_message, q), daemon=True)
    t.start()

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
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ─────────────────────────────────────────────
# Início
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Servidor iniciado em http://localhost:{port}\n")
    app.run(debug=False, port=port, threaded=True)
