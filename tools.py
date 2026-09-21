from langchain.tools import tool
from exa_py import Exa
import requests
from bs4 import BeautifulSoup
import os
import re
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

    try:
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

            snippet = " ".join(highlights)

            # Keep individual snippets small
            snippet = snippet[:600]

            out.append(
                f"Title: {r.title}\n"
                f"URL: {r.url}\n"
                f"Snippet: {snippet}\n"
            )

        if not out:
            return "No relevant search results were found."

        return "\n----\n".join(out)

    except Exception as e:

        return f"Web search failed: {str(e)}"


# =========================================================
# WEB SCRAPING TOOL
# =========================================================

@tool
def scrape_url(url: str) -> str:
    """
    Scrape a webpage and return clean, limited text content
    for deeper research.
    """

    try:

        # -------------------------------------------------
        # DOWNLOAD PAGE
        # -------------------------------------------------

        response = requests.get(
            url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/140.0 Safari/537.36"
                )
            }
        )

        response.raise_for_status()

        # -------------------------------------------------
        # PARSE HTML
        # -------------------------------------------------

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # -------------------------------------------------
        # REMOVE UNNECESSARY HTML
        # -------------------------------------------------

        remove_tags = [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "button",
            "input",
            "textarea",
            "select",
            "option",
            "svg",
            "noscript",
            "iframe"
        ]

        for tag in soup(remove_tags):
            tag.decompose()

        # -------------------------------------------------
        # REMOVE COMMON NON-CONTENT ELEMENTS
        # -------------------------------------------------

        for tag in soup.find_all(
            ["div", "section"],
            class_=re.compile(
                r"(cookie|consent|advert|ads|popup|modal|sidebar|social)",
                re.I
            )
        ):
            tag.decompose()

        # -------------------------------------------------
        # GET TEXT
        # -------------------------------------------------

        text = soup.get_text(
            separator=" ",
            strip=True
        )

        # -------------------------------------------------
        # CLEAN WHITESPACE
        # -------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        # -------------------------------------------------
        # LIMIT CONTENT SIZE
        #
        # This is important because the Reader Agent
        # sends this content to Groq.
        # -------------------------------------------------

        MAX_CHARS = 18000

        if len(text) > MAX_CHARS:

            text = (
                text[:MAX_CHARS]
                + "\n\n"
                "[SCRAPED CONTENT TRUNCATED FOR TOKEN LIMIT]"
            )

        # -------------------------------------------------
        # RETURN CLEAN CONTENT
        # -------------------------------------------------

        return (
            f"Source URL: {url}\n\n"
            f"Scraped Content:\n{text}"
        )

    except requests.exceptions.Timeout:

        return (
            f"Could not scrape URL: {url}\n"
            "Reason: Request timed out."
        )

    except requests.exceptions.RequestException as e:

        return (
            f"Could not scrape URL: {url}\n"
            f"Reason: HTTP request failed: {str(e)}"
        )

    except Exception as e:

        return (
            f"Could not scrape URL: {url}\n"
            f"Reason: {str(e)}"
        )