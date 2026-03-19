"""
Agente de Pesquisa Acadêmica — Semantic Scholar + OpenAlex + arXiv + Claude Opus 4.6

Busca em fontes acadêmicas confiáveis (artigos de grandes universidades) e retorna
citações formatadas (ABNT, APA, MLA ou STEAM).
"""

import os
import json
import time
import requests

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


# ─────────────────────────────────────────────
# Busca OpenAlex (grandes universidades mundiais)
# ─────────────────────────────────────────────

def search_openalex(query: str, max_results: int = 5) -> list[dict]:
    """
    Busca via OpenAlex API — base aberta com +250 milhões de trabalhos acadêmicos
    de universidades como Harvard, MIT, Stanford, USP, Unicamp etc.
    Gratuita, sem necessidade de chave de API.
    """
    headers = {"User-Agent": "AcademicResearchAgent/1.0"}
    try:
        resp = requests.get(
            "https://api.openalex.org/works",
            params={
                "search": query,
                "per_page": min(max_results, 10),
                "select": (
                    "title,authorships,publication_year,primary_location,"
                    "doi,abstract_inverted_index,cited_by_count,open_access"
                ),
            },
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    papers = []
    for item in data.get("results", []):
        # Reconstrói o abstract a partir do índice invertido
        inv = item.get("abstract_inverted_index") or {}
        if inv:
            word_pos = [(w, p) for w, positions in inv.items() for p in positions]
            word_pos.sort(key=lambda x: x[1])
            abstract = " ".join(w for w, _ in word_pos)
        else:
            abstract = ""

        # Autores
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in (item.get("authorships") or [])
        ]

        # Periódico / fonte
        loc = item.get("primary_location") or {}
        source = loc.get("source") or {}
        journal = source.get("display_name", "")

        # DOI e URL
        doi = (item.get("doi") or "").replace("https://doi.org/", "")
        url = loc.get("landing_page_url", "") or (item.get("open_access") or {}).get("oa_url", "")

        papers.append({
            "source_type": "openalex",
            "title": item.get("title", "Título não disponível"),
            "authors": [a for a in authors if a],
            "year": item.get("publication_year") or "s.d.",
            "journal": journal,
            "abstract": abstract[:600],
            "url": url or (f"https://doi.org/{doi}" if doi else ""),
            "citations": item.get("cited_by_count", 0),
            "doi": doi,
        })

    return papers


# ─────────────────────────────────────────────
# Busca arXiv (Cornell University)
# ─────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 4) -> list[dict]:
    """
    Busca via arXiv API — repositório da Cornell University com preprints
    de física, matemática, computação, biologia, economia etc.
    Gratuita, sem necessidade de chave de API.
    """
    import xml.etree.ElementTree as ET

    headers = {"User-Agent": "AcademicResearchAgent/1.0"}
    try:
        resp = requests.get(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": min(max_results, 10),
                "sortBy": "relevance",
            },
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
        abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
        published = entry.findtext("atom:published", "", ns)
        year = published[:4] if published else "s.d."
        url = entry.findtext("atom:id", "", ns) or ""

        authors = [
            a.findtext("atom:name", "", ns)
            for a in entry.findall("atom:author", ns)
        ]

        # DOI, se disponível
        doi = ""
        doi_el = entry.find("arxiv:doi", ns)
        if doi_el is not None and doi_el.text:
            doi = doi_el.text.strip()

        papers.append({
            "source_type": "arxiv",
            "title": title,
            "authors": [a for a in authors if a],
            "year": year,
            "journal": "arXiv (Cornell University)",
            "abstract": abstract[:600],
            "url": url,
            "citations": 0,
            "doi": doi,
        })

    return papers


def format_citation(article: dict, style: str = "ABNT") -> str:
    """Formata uma fonte acadêmica no estilo ABNT, APA, MLA ou STEAM."""
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

    # ── STEAM ─────────────────────────────────────────
    elif style == "STEAM":
        steam_year = "SD" if str(year).lower() in ("s.d.", "sd", "") else str(year)

        # Referência: todos os autores listados, separados por ponto e vírgula
        # Formato: SOBRENOME, Iniciais.; SOBRENOME2, Iniciais2.
        if not authors:
            author_str = "AUTOR DESCONHECIDO"
        else:
            parts_list = []
            for a in authors:
                last, ini = _split_name(a)
                parts_list.append(f"{last.upper()}, {ini}" if ini else last.upper())
            author_str = "; ".join(parts_list)

        cit = f"{author_str} {title}."
        if journal:
            cit += f" {journal},"
        cit += f" {steam_year}."
        if doi:
            cit += f" DOI: {doi}."
        elif url:
            cit += f" Disponível em: <{url}>. Acesso em: {_today()}."
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
            "nos padrões ABNT (NBR 6023), APA (7ª ed.), MLA (9ª ed.) ou STEAM."
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
                    "enum": ["ABNT", "APA", "MLA", "STEAM"],
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

SYSTEM_PROMPT = """Assistente de pesquisa acadêmica. Fontes: Semantic Scholar, OpenAlex (grandes universidades), arXiv (Cornell).
Sempre responda em português brasileiro. Se não houver fontes, use seu conhecimento geral.
Indique a origem de cada fonte. Estilos: ABNT, APA, MLA, STEAM.
STEAM: citação (SOBRENOME, ano); 2 autores (A & B, ano); 3+ (A et al., ano); sem data: SD.
Seja direto e objetivo."""


# ─────────────────────────────────────────────
# Loop principal do agente
# ─────────────────────────────────────────────

def run_agent(user_message: str, verbose: bool = True) -> str:
    """Stub mantido para compatibilidade com a CLI local."""
    return "Use app.py para executar o servidor web."


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
