from langchain.tools import tool
from exa_py import Exa
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# EXA CLIENT
# =========================================================

exa = Exa(
    api_key=os.getenv("EXA_API_KEY")
)


# =========================================================
# WEB SEARCH TOOL
# =========================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for recent and reliable information
    related to the user's research question.
    """

    results = exa.search(
        query,
        type="auto",
        num_results=5,
        contents={
            "highlights": True
        }
    )

    out = []

    for r in results.results:

        highlights = r.highlights or []

        out.append(
            f"Title: {r.title}\n"
            f"URL: {r.url}\n"
            f"Snippet: {' '.join(highlights)[:500]}\n"
        )

    if not out:
        return "No relevant search results were found."

    return "\n----\n".join(out)


# =========================================================
# WEB SCRAPING TOOL
# =========================================================

@tool
def scrape_url(url: str) -> str:
    """
    Scrape a webpage and return clean text content
    for deeper research.
    """

    try:

        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove unnecessary elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # Limit returned content
        return text[:3000]

    except Exception as e:

        return f"Could not scrape URL: {str(e)}"