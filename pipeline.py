import time

from agents import (
    build_search_agent,
    build_openrouter_search_agent,
    build_reader_agent,
    build_openrouter_reader_agent,
    writer_chain,
    critic_chain,
    revision_chain,
)


# =========================================================
# HELPER: EXTRACT FINAL TEXT FROM AGENT
# =========================================================

def extract_agent_text(result):
    """
    Extract the last useful text response from a LangChain agent.
    Avoids relying on messages[-1], which can be a tool message.
    """

    messages = result.get("messages", [])

    for message in reversed(messages):
        content = getattr(message, "content", "")

        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            parts = []

            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "")
                    if text:
                        parts.append(text)

            if parts:
                return "\n".join(parts).strip()

    return ""


# =========================================================
# AGENT WITH PROVIDER FALLBACK
# =========================================================

def invoke_agent_with_fallback(
    primary_agent,
    fallback_agent,
    inputs,
    name
):

    # -----------------------------------------------------
    # TRY PRIMARY
    # -----------------------------------------------------

    try:

        print(f"\n{name} - Primary provider: Groq")

        result = primary_agent.invoke(inputs)

        text = extract_agent_text(result)

        if text:

            print(f"\n{name} completed using Groq.")

            return result

        print(f"\n{name} - Groq returned an empty response.")

    except Exception as e:

        print(f"\n{name} - Groq failed:")
        print(e)

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    print(f"\n{name} - Switching to OpenRouter fallback...")

    try:

        result = fallback_agent.invoke(inputs)

        text = extract_agent_text(result)

        if not text:

            raise RuntimeError(
                f"{name} - OpenRouter returned an empty response."
            )

        print(
            f"\n{name} completed using OpenRouter fallback."
        )

        return result

    except Exception as e:

        print(f"\n{name} - OpenRouter fallback failed:")
        print(e)

        raise


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}

    # =====================================================
    # STEP 1 - SEARCH
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 - Search Agent is working...")
    print("=" * 60)

    search_agent = build_search_agent()

    openrouter_search_agent = (
        build_openrouter_search_agent()
    )

    search_result = invoke_agent_with_fallback(
        search_agent,
        openrouter_search_agent,
        {
            "messages": [
                (
                    "user",
                    f"""
Research this exact question:

{topic}

Find recent, reliable and relevant information.

Search ONLY for information related to this question.

Use the web_search tool.

Return:

- Important findings
- Source titles
- URLs
- Dates
- Relevant facts

Keep the answer concise.
"""
                )
            ]
        },
        name="Search Agent"
    )

    search_text = extract_agent_text(search_result)

    if not search_text:

        raise RuntimeError(
            "Search Agent returned no usable search results."
        )

    state["search_results"] = search_text

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"])


    # =====================================================
    # STEP 2 - READER
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 - Reader Agent is scraping resources...")
    print("=" * 60)

    reader_agent = build_reader_agent()

    openrouter_reader_agent = (
        build_openrouter_reader_agent()
    )

    # Keep the search input small
    search_for_reader = (
        state["search_results"][:6000]
    )

    reader_result = invoke_agent_with_fallback(
        reader_agent,
        openrouter_reader_agent,
        {
            "messages": [
                (
                    "user",
                    f"""
Research question:

{topic}

Below are the search results:

{search_for_reader}

Choose the most relevant URL related to the
research question.

Use the scrape_url tool to read that webpage.

Extract only the important factual information.

Return:

- Important facts
- Main points
- Dates
- Names
- Statistics
- Useful source information

Keep the answer concise.
"""
                )
            ]
        },
        name="Reader Agent"
    )

    scraped_text = extract_agent_text(
        reader_result
    )

    if not scraped_text:

        raise RuntimeError(
            "Reader Agent returned no usable content."
        )

    state["scraped_content"] = scraped_text

    print("\nSCRAPED CONTENT:\n")
    print(state["scraped_content"])


    # =====================================================
    # STEP 3 - WRITER
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 3 - Writer is drafting the report...")
    print("=" * 60)

    research_combined = (
        f"SEARCH RESULTS:\n"
        f"{state['search_results'][:6000]}\n\n"
        f"SCRAPED CONTENT:\n"
        f"{state['scraped_content'][:5000]}"
    )

    state["research"] = research_combined

    try:

        state["report"] = writer_chain.invoke(
            {
                "topic": topic,
                "research": research_combined
            }
        )

        print("\nINITIAL REPORT:\n")
        print(state["report"])

    except Exception as e:

        print("\nWriter failed:")
        print(e)

        raise


    # =====================================================
    # STEP 4 - CRITIC
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 4 - Critic is reviewing the report...")
    print("=" * 60)

    try:

        state["feedback"] = critic_chain.invoke(
            {
                "topic": topic,
                "research": state["research"][:10000],
                "report": state["report"][:7000]
            }
        )

        print("\nCRITIC REPORT:\n")
        print(state["feedback"])

    except Exception as e:

        print("\nCritic failed:")
        print(e)

        raise


    # =====================================================
    # STEP 5 - FINAL WRITER
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 5 - Final Writer is improving the answer...")
    print("=" * 60)

    try:

        state["final_answer"] = revision_chain.invoke(
            {
                "topic": topic,
                "research": state["research"][:10000],
                "report": state["report"][:6000],
                "feedback": state["feedback"][:4000]
            }
        )

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)

        print(state["final_answer"])

    except Exception as e:

        print("\nFinal Writer failed:")
        print(e)

        raise


    return state


# =========================================================
# DIRECT TERMINAL TEST
# =========================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter a research topic: "
    )

    result = run_research_pipeline(topic)