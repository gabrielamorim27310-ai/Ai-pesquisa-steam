"""
Agente de Pesquisa Acadêmica — Semantic Scholar + Claude Opus 4.6

Busca artigos científicos e retorna citações formatadas (ABNT, APA ou MLA).
Usa a Semantic Scholar API (gratuita, sem necessidade de chave).
"""

import os
import json
import time
import requests
import anthropic

# ─────────────────────────────────────────────
# Funções de busca e formatação
# ─────────────────────────────────────────────

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEMANTIC_SCHOLAR_FIELDS = (
    "title,authors,year,abstract,citationCount,"
    "externalIds,venue,publicationVenue,url,openAccessPdf"
)


def search_academic_papers(query: str, max_results: int = 5, year_from: int = None) -> list[dict]:
    """
    Busca artigos científicos via Semantic Scholar API.
    Retorna lista de dicts com metadados padronizados.
    """
    params = {
        "query": query,
        "limit": min(max_results, 10),
        "fields": SEMANTIC_SCHOLAR_FIELDS,
    }
    if year_from:
        params["year"] = f"{year_from}-"

    headers = {"User-Agent": "AcademicResearchAgent/1.0"}

    for attempt in range(3):
        try:
            resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                return [{"error": f"Erro de conexão: {str(e)}"}]
            time.sleep(2 ** attempt)
        except ValueError:
            return [{"error": "Resposta inválida da API"}]
    else:
        return [{"error": "Limite de requisições atingido. Tente novamente em alguns segundos."}]

    papers = []
    for item in data.get("data", []):
        doi = (item.get("externalIds") or {}).get("DOI", "")
        pdf_url = ""
        if item.get("openAccessPdf"):
            pdf_url = item["openAccessPdf"].get("url", "")

        venue = ""
        if item.get("publicationVenue"):
            venue = item["publicationVenue"].get("name", "")
        if not venue:
            venue = item.get("venue", "")

        paper = {
            "title": item.get("title", "Título não disponível"),
            "authors": [a.get("name", "") for a in (item.get("authors") or [])],
            "year": item.get("year") or "s.d.",
            "journal": venue,
            "abstract": item.get("abstract", ""),
            "url": item.get("url", "") or pdf_url,
            "citations": item.get("citationCount", 0),
            "doi": doi,
        }
        papers.append(paper)

    return papers if papers else [{"error": "Nenhum artigo encontrado para esta busca."}]


def format_citation(article: dict, style: str = "ABNT") -> str:
    """Formata um artigo no estilo ABNT, APA ou MLA."""
    authors: list = article.get("authors", [])
    title: str = article.get("title", "")
    year = str(article.get("year", "s.d."))
    journal: str = article.get("journal", "")
    doi: str = article.get("doi", "")
    url: str = article.get("url", "")

    style = style.upper().strip()

    def _split_name(full_name: str) -> tuple[str, str]:
        """Retorna (sobrenome, iniciais)."""
        parts = full_name.strip().split()
        if not parts:
            return ("Desconhecido", "")
        last = parts[-1]
        initials = ". ".join(p[0].upper() for p in parts[:-1]) + "." if len(parts) > 1 else ""
        return last, initials

    # ── ABNT ──────────────────────────────────────────
    if style == "ABNT":
        if authors:
            last, ini = _split_name(authors[0])
            author_str = f"{last.upper()}, {ini}" if ini else last.upper()
            if len(authors) > 1:
                author_str += " et al."
        else:
            author_str = "AUTOR DESCONHECIDO"

        cit = f"{author_str} {title}."
        if journal:
            cit += f" **{journal}**,"
        cit += f" {year}."
        if doi:
            cit += f" DOI: {doi}."
        elif url:
            cit += f" Disponível em: <{url}>. Acesso em: {_today()}."
        return cit

    # ── APA (7ª ed.) ──────────────────────────────────
    elif style == "APA":
        apa_list = []
        for a in authors[:20]:
            last, ini = _split_name(a)
            apa_list.append(f"{last}, {ini}" if ini else last)
        if len(authors) > 20:
            apa_list = apa_list[:19] + ["... " + apa_list[-1]]

        if len(apa_list) > 1:
            author_str = ", ".join(apa_list[:-1]) + f", & {apa_list[-1]}"
        else:
            author_str = apa_list[0] if apa_list else "Autor desconhecido"

        cit = f"{author_str} ({year}). {title}."
        if journal:
            cit += f" *{journal}*."
        if doi:
            cit += f" https://doi.org/{doi}"
        elif url:
            cit += f" {url}"
        return cit

    # ── MLA (9ª ed.) ──────────────────────────────────
    elif style == "MLA":
        if authors:
            last, ini = _split_name(authors[0])
            first_parts = [p for p in authors[0].split()[:-1]]
            first_name = " ".join(first_parts)
            author_str = f"{last}, {first_name}" if first_name else last
            if len(authors) > 1:
                author_str += ", et al."
        else:
            author_str = "Autor desconhecido"

        cit = f'{author_str}. "{title}."'
        if journal:
            cit += f" *{journal}*,"
        cit += f" {year}."
        if doi:
            cit += f" DOI: {doi}."
        elif url:
            cit += f" {url}."
        return cit

    return f"{title} ({year})"


def _today() -> str:
    from datetime import date
    d = date.today()
    return f"{d.day:02d} {_month_pt(d.month)} {d.year}"


def _month_pt(m: int) -> str:
    months = ["jan.", "fev.", "mar.", "abr.", "maio", "jun.",
              "jul.", "ago.", "set.", "out.", "nov.", "dez."]
    return months[m - 1]


# ─────────────────────────────────────────────
# Definição das ferramentas para a API Claude
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_academic_papers",
        "description": (
            "Busca artigos científicos na Semantic Scholar (base com +200 milhões de artigos). "
            "Retorna título, autores, ano, resumo, periódico, URL, DOI e número de citações. "
            "Use esta ferramenta sempre que o usuário pedir artigos, pesquisas ou referências científicas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Termos de busca em inglês ou português. "
                        "Use termos específicos para melhores resultados. "
                        "Exemplo: 'deep learning medical image segmentation'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Quantidade de artigos a retornar (1–10, padrão: 5)",
                    "default": 5,
                },
                "year_from": {
                    "type": "integer",
                    "description": "Filtrar artigos a partir deste ano (ex: 2020). Omita para sem restrição.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "format_citation",
        "description": (
            "Formata os metadados de um artigo como referência bibliográfica "
            "nos padrões ABNT (NBR 6023), APA (7ª ed.) ou MLA (9ª ed.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "article": {
                    "type": "object",
                    "description": (
                        "Objeto com os campos do artigo: "
                        "title (str), authors (list[str]), year (str|int), "
                        "journal (str), doi (str), url (str)"
                    ),
                    "properties": {
                        "title": {"type": "string"},
                        "authors": {"type": "array", "items": {"type": "string"}},
                        "year": {},
                        "journal": {"type": "string"},
                        "doi": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["title"],
                },
                "style": {
                    "type": "string",
                    "enum": ["ABNT", "APA", "MLA"],
                    "description": "Estilo de citação (padrão: ABNT)",
                    "default": "ABNT",
                },
            },
            "required": ["article"],
        },
    },
]


# ─────────────────────────────────────────────
# Executor de ferramentas
# ─────────────────────────────────────────────

def execute_tool(name: str, inputs: dict) -> str:
    if name == "search_academic_papers":
        results = search_academic_papers(
            query=inputs["query"],
            max_results=inputs.get("max_results", 5),
            year_from=inputs.get("year_from"),
        )
        return json.dumps(results, ensure_ascii=False, indent=2)

    elif name == "format_citation":
        return format_citation(
            article=inputs["article"],
            style=inputs.get("style", "ABNT"),
        )

    return json.dumps({"error": f"Ferramenta '{name}' não encontrada."})


# ─────────────────────────────────────────────
# Sistema prompt do agente
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Você é um assistente especializado em pesquisa acadêmica.
Ajuda pesquisadores, estudantes e profissionais a encontrar os melhores artigos
científicos e gera referências bibliográficas no formato correto.

Fluxo de resposta:
1. Analise a solicitação do usuário e identifique os termos-chave de busca
2. Use search_academic_papers para buscar artigos relevantes
   - Se necessário, faça 2-3 buscas com termos diferentes para cobrir o tema
   - Priorize artigos com alto número de citações (mais impacto científico)
3. Selecione os 3-5 artigos mais relevantes dos resultados
4. Use format_citation para formatar cada artigo selecionado
5. Apresente uma resposta estruturada:
   - Breve introdução sobre o tema
   - Para cada artigo: título, autores, ano, resumo em português, relevância
   - Seção "Referências" com as citações formatadas

Regras importantes:
- Se o usuário não especificar estilo, use ABNT
- Estilos aceitos: ABNT, APA, MLA
- Se não encontrar resultados, tente buscar em inglês (mesmo para pedidos em português)
- Priorize artigos dos últimos 10 anos, salvo se o usuário pedir obras clássicas
- Responda sempre em português brasileiro
- Seja direto e objetivo nas descrições dos artigos"""


# ─────────────────────────────────────────────
# Loop principal do agente
# ─────────────────────────────────────────────

def run_agent(user_message: str, verbose: bool = True) -> str:
    """
    Executa o agente de pesquisa e retorna a resposta final.

    Args:
        user_message: Solicitação do usuário
        verbose: Se True, imprime o progresso em tempo real

    Returns:
        Texto da resposta final do agente
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "❌ ANTHROPIC_API_KEY não configurada. Execute: export ANTHROPIC_API_KEY='sua-chave'"

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": "user", "content": user_message}]

    if verbose:
        print("\n🔍 Iniciando pesquisa acadêmica...\n")

    while True:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=8192,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Adiciona resposta ao histórico
        messages.append({"role": "assistant", "content": response.content})

        # Resposta final
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # Executa ferramentas
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if verbose:
                        args_preview = json.dumps(block.input, ensure_ascii=False)[:80]
                        print(f"  ⚙️  {block.name}({args_preview}...)")

                    result = execute_tool(block.name, block.input)
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
                    return block.text
            return ""


# ─────────────────────────────────────────────
# Interface interativa de linha de comando
# ─────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════╗
║    🎓 Agente de Pesquisa Acadêmica — Google Scholar      ║
║         Powered by Claude Opus 4.6 + Semantic Scholar    ║
╚══════════════════════════════════════════════════════════╝

Exemplos de pesquisa:
  • aprendizado de máquina para diagnóstico de câncer
  • impacto das redes sociais na saúde mental APA
  • quantum computing optimization algorithms (2020-) ABNT
  • mudanças climáticas e biodiversidade marinha MLA

Digite 'sair' para encerrar.
"""


def main():
    print(BANNER)

    while True:
        try:
            user_input = input("🎓 Pesquisa: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nEncerrando... Até logo!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("sair", "exit", "quit", "q"):
            print("Até logo! Bons estudos! 📖")
            break

        result = run_agent(user_input)
        print(f"\n{'─' * 60}")
        print(result)
        print(f"{'─' * 60}\n")


if __name__ == "__main__":
    main()
