import os
import re
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# Vercel tem filesystem read-only, usar /tmp
_TMP = Path("/tmp")
app.config["UPLOAD_FOLDER"] = _TMP / "uploads"
app.config["RESUMOS_FOLDER"] = _TMP / "resumos"

AUDIO_EXTS = {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".webm", ".flac"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

app.config["RESUMOS_FOLDER"].mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)


# ── Extração de conteúdo de URL ───────────────────────────────────────────────
def _youtube_id(url: str):
    """Extrai o ID do vídeo de uma URL do YouTube."""
    patterns = [
        r"(?:v=|youtu\.be/|/embed/|/v/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def extrair_conteudo_url(url: str) -> tuple[str, str]:
    """
    Retorna (tipo, conteudo):
      tipo: 'youtube' | 'pagina'
      conteudo: texto extraído (truncado)
    """
    import requests

    vid_id = _youtube_id(url)
    if vid_id:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            partes = YouTubeTranscriptApi.get_transcript(vid_id, languages=["pt", "pt-BR", "en"])
            texto = " ".join(p["text"] for p in partes)
            return "youtube", texto[:3000]
        except Exception as e:
            return "youtube", f"[Não foi possível obter legenda: {e}]"

    # Página web genérica
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        # Remove tags HTML
        texto = re.sub(r"<[^>]+>", " ", resp.text)
        texto = re.sub(r"\s+", " ", texto).strip()
        return "pagina", texto[:3000]
    except Exception as e:
        return "pagina", f"[Erro ao acessar o link: {e}]"


# ── Transcrição de áudio via Groq Whisper ─────────────────────────────────────
def transcrever_audio(caminho: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(caminho, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=(Path(caminho).name, f),
            model="whisper-large-v3",
            response_format="text",
            language="pt",
        )
    return resp if isinstance(resp, str) else resp.text


# ── Descrição de imagem via Groq Vision ───────────────────────────────────────
def descrever_imagem(caminho: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(caminho, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(caminho).suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Você é um assistente escolar. Descreva de forma detalhada "
                            "o que está nesta imagem de aula: textos, diagramas, fórmulas, "
                            "quadro-negro, slides ou qualquer conteúdo educacional visível."
                        ),
                    },
                ],
            }
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


# ── Geração do resumo HTML via Groq ───────────────────────────────────────────
def gerar_resumo(transcricao: str, descricoes_imagens: list[str], titulo: str,
                  conteudo_url: str = "", tipo_url: str = "") -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    partes = []
    if transcricao:
        partes.append(f"TRANSCRIÇÃO DO ÁUDIO:\n{transcricao[:1500]}")
    for i, desc in enumerate(descricoes_imagens, 1):
        partes.append(f"IMAGEM {i}:\n{desc[:400]}")
    if conteudo_url:
        label = "TRANSCRIÇÃO DO VÍDEO (YouTube)" if tipo_url == "youtube" else "CONTEÚDO DA PÁGINA"
        partes.append(f"{label}:\n{conteudo_url[:1500]}")

    conteudo = "\n\n".join(partes) if partes else "Nenhum conteúdo enviado."

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente escolar especializado em criar resumos de aulas. "
                    "Organize o conteúdo de forma clara e didática em português brasileiro."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Crie um resumo organizado da aula '{titulo}' com base neste conteúdo:\n\n"
                    f"{conteudo}\n\n"
                    "Estruture o resumo com:\n"
                    "1. Título da aula\n"
                    "2. Tópicos principais (com subtópicos se necessário)\n"
                    "3. Conceitos-chave em destaque\n"
                    "4. Conclusão / O que aprender\n"
                    "Responda APENAS com o conteúdo do resumo em texto simples organizado."
                ),
            },
        ],
        max_tokens=900,
    )
    return resp.choices[0].message.content.strip()


# ── Converter resumo em HTML ───────────────────────────────────────────────────
def resumo_para_html(resumo_texto: str, titulo: str, data_hora: str,
                     n_imagens: int, tem_audio: bool, tipo_url: str = "") -> str:
    linhas = resumo_texto.split("\n")
    blocos = []
    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        if linha.startswith("# ") or linha.isupper() and len(linha) < 80:
            blocos.append(f"<h2>{linha.lstrip('# ')}</h2>")
        elif linha.startswith("## "):
            blocos.append(f"<h3>{linha.lstrip('# ')}</h3>")
        elif linha.startswith("- ") or linha.startswith("• "):
            blocos.append(f"<li>{linha[2:]}</li>")
        elif linha[0].isdigit() and linha[1] in ".):":
            blocos.append(f"<li>{linha[2:].strip()}</li>")
        else:
            blocos.append(f"<p>{linha}</p>")

    corpo = "\n    ".join(blocos)
    fontes = []
    if tem_audio:
        fontes.append('<span class="badge audio">🎵 Áudio</span>')
    if n_imagens:
        fontes.append(f'<span class="badge imagem">🖼️ {n_imagens} imagem(ns)</span>')
    if tipo_url == "youtube":
        fontes.append('<span class="badge youtube">▶️ YouTube</span>')
    elif tipo_url == "pagina":
        fontes.append('<span class="badge link">🔗 Link</span>')

    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{titulo} — Resumo de Aula</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #f0f4f8;
      color: #1a202c;
      padding: 2rem 1rem;
    }}
    .card {{
      max-width: 820px;
      margin: 0 auto;
      background: #fff;
      border-radius: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,.1);
      overflow: hidden;
    }}
    .header {{
      background: linear-gradient(135deg, #4f46e5, #7c3aed);
      color: #fff;
      padding: 2rem 2.5rem;
    }}
    .header h1 {{ font-size: 1.8rem; margin-bottom: .5rem; }}
    .meta {{ font-size: .85rem; opacity: .85; margin-top: .4rem; }}
    .badges {{ display: flex; gap: .5rem; margin-top: 1rem; flex-wrap: wrap; }}
    .badge {{
      font-size: .78rem;
      padding: .25rem .7rem;
      border-radius: 999px;
      font-weight: 600;
    }}
    .badge.audio {{ background: #fef3c7; color: #92400e; }}
    .badge.imagem {{ background: #dbeafe; color: #1e40af; }}
    .badge.youtube {{ background: #fee2e2; color: #991b1b; }}
    .badge.link {{ background: #d1fae5; color: #065f46; }}
    .body {{ padding: 2rem 2.5rem; line-height: 1.75; }}
    h2 {{
      font-size: 1.2rem;
      color: #4f46e5;
      margin: 1.8rem 0 .6rem;
      border-left: 4px solid #4f46e5;
      padding-left: .75rem;
    }}
    h3 {{
      font-size: 1.05rem;
      color: #374151;
      margin: 1.2rem 0 .4rem;
    }}
    p {{ margin: .5rem 0; color: #374151; }}
    li {{
      margin: .4rem 0 .4rem 1.5rem;
      color: #374151;
    }}
    .footer {{
      text-align: center;
      padding: 1rem;
      font-size: .78rem;
      color: #9ca3af;
      border-top: 1px solid #f3f4f6;
    }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>📚 {titulo}</h1>
      <div class="meta">Resumo gerado em {data_hora}</div>
      <div class="badges">{' '.join(fontes)}</div>
    </div>
    <div class="body">
    {corpo}
    </div>
    <div class="footer">Gerado automaticamente pelo Bot de Resumo de Aulas</div>
  </div>
</body>
</html>"""


# ── Rotas Flask ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    resumos = []
    pasta = app.config["RESUMOS_FOLDER"]
    for f in sorted(pasta.glob("*.json"), reverse=True):
        try:
            info = json.loads(f.read_text(encoding="utf-8"))
            resumos.append(info)
        except Exception:
            pass
    return render_template("index.html", resumos=resumos)


@app.route("/upload", methods=["POST"])
def upload():
    titulo = request.form.get("titulo", "Aula sem título").strip() or "Aula sem título"
    link = request.form.get("link", "").strip()
    arquivos = request.files.getlist("arquivos")

    sem_arquivos = not arquivos or all(f.filename == "" for f in arquivos)
    if sem_arquivos and not link:
        return jsonify({"erro": "Envie pelo menos um arquivo ou um link."}), 400

    audios, imagens = [], []
    pasta_upload = app.config["UPLOAD_FOLDER"]

    for arq in arquivos:
        if not arq.filename:
            continue
        nome = secure_filename(arq.filename)
        destino = pasta_upload / nome
        arq.save(str(destino))
        ext = Path(nome).suffix.lower()
        if ext in AUDIO_EXTS:
            audios.append(str(destino))
        elif ext in IMAGE_EXTS:
            imagens.append(str(destino))

    # Processar áudio
    transcricao = ""
    if audios:
        try:
            transcricao = transcrever_audio(audios[0])
        except Exception as e:
            transcricao = f"[Erro na transcrição: {e}]"

    # Processar imagens
    descricoes = []
    for img in imagens[:4]:
        try:
            descricoes.append(descrever_imagem(img))
        except Exception as e:
            descricoes.append(f"[Erro ao descrever imagem: {e}]")

    # Processar link
    conteudo_url, tipo_url = "", ""
    if link:
        try:
            tipo_url, conteudo_url = extrair_conteudo_url(link)
        except Exception as e:
            conteudo_url = f"[Erro ao processar link: {e}]"
            tipo_url = "pagina"

    # Gerar resumo
    try:
        resumo_texto = gerar_resumo(transcricao, descricoes, titulo, conteudo_url, tipo_url)
    except Exception as e:
        resumo_texto = f"Erro ao gerar resumo: {e}"

    # Salvar HTML
    agora = datetime.now()
    slug = agora.strftime("%Y%m%d_%H%M%S")
    data_hora = agora.strftime("%d/%m/%Y às %H:%M")
    html_content = resumo_para_html(resumo_texto, titulo, data_hora, len(imagens), bool(audios), tipo_url)

    pasta_resumos = app.config["RESUMOS_FOLDER"]
    html_path = pasta_resumos / f"resumo_{slug}.html"
    html_path.write_text(html_content, encoding="utf-8")

    meta = {
        "titulo": titulo,
        "slug": slug,
        "data_hora": data_hora,
        "n_imagens": len(imagens),
        "tem_audio": bool(audios),
        "tipo_url": tipo_url,
        "arquivo": f"resumo_{slug}.html",
    }
    (pasta_resumos / f"resumo_{slug}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    return redirect(url_for("ver_resumo", slug=slug))


@app.route("/resumo/<slug>")
def ver_resumo(slug: str):
    pasta = app.config["RESUMOS_FOLDER"]
    html_path = pasta / f"resumo_{slug}.html"
    if not html_path.exists():
        return "Resumo não encontrado.", 404
    return html_path.read_text(encoding="utf-8")


@app.route("/resumos/<path:filename>")
def static_resumos(filename):
    return send_from_directory(app.config["RESUMOS_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
