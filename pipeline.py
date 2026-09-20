import time

from agents import (
    build_reader_agent,
    build_search_agent,
    writer_chain,
    critic_chain,
    revision_chain
)


# =========================================================
# HELPER FUNCTION
# =========================================================

def invoke_with_retry(chain, inputs, name="LLM", attempts=3):

    for attempt in range(attempts):

        try:

            print(f"\n{name} - Attempt {attempt + 1}/{attempts}")

            result = chain.invoke(inputs)

            return result

        except Exception as e:

            print(f"\n{name} failed:")
            print(e)

            if attempt < attempts - 1:

                print("\nWaiting 10 seconds before retry...")
                time.sleep(10)

            else:

                raise


# =========================================================
# MAIN RESEARCH PIPELINE
# =========================================================

def run_research_pipeline(topic: str) -> dict:

    state = {}


    # =====================================================
    # STEP 1 - SEARCH AGENT
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 1 - Search Agent is working...")
    print("=" * 60)

    search_agent = build_search_agent()

    search_result = invoke_with_retry(
        search_agent,
        {
            "messages": [
                (
                    "user",
                    f"""
Research this exact question:

{topic}

Find recent, reliable and relevant information.

Search ONLY for information related to this question.

Return:
- Important findings
- Source titles
- URLs
- Dates
- Relevant facts
"""
                )
            ]
        },
        name="Search Agent"
    )

    state["search_results"] = search_result["messages"][-1].content

    print("\nSEARCH RESULTS:\n")
    print(state["search_results"])


    # =====================================================
    # STEP 2 - READER AGENT
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 2 - Reader Agent is scraping resources...")
    print("=" * 60)

    reader_agent = build_reader_agent()

    # Limit search information to avoid unnecessary token usage
    search_for_reader = state["search_results"][:6000]

    reader_result = invoke_with_retry(
        reader_agent,
        {
            "messages": [
                (
                    "user",
                    f"""
Research question:

{topic}

Below are the search results:

{search_for_reader}

Choose the most relevant URL related to the research
question and scrape it.

Extract the important facts from that source.
"""
                )
            ]
        },
        name="Reader Agent"
    )

    state["scraped_content"] = reader_result["messages"][-1].content

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

    state["report"] = invoke_with_retry(
        writer_chain,
        {
            "topic": topic,
            "research": research_combined
        },
        name="Writer"
    )

    print("\nINITIAL REPORT:\n")
    print(state["report"])


    # =====================================================
    # STEP 4 - CRITIC
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 4 - Critic is reviewing the report...")
    print("=" * 60)

    # Give Groq's TPM window some time
    print("\nWaiting for Groq rate limit window...")
    time.sleep(5)

    state["feedback"] = invoke_with_retry(
        critic_chain,
        {
            "topic": topic,
            "report": state["report"][:7000]
        },
        name="Critic"
    )

    print("\nCRITIC REPORT:\n")
    print(state["feedback"])


    # =====================================================
    # STEP 5 - REVISION WRITER
    # =====================================================

    print("\n" + "=" * 60)
    print("STEP 5 - Final Writer is improving the answer...")
    print("=" * 60)

    # Give Groq another moment before final request
    print("\nWaiting for Groq rate limit window...")
    time.sleep(5)

    state["final_answer"] = invoke_with_retry(
        revision_chain,
        {
            "topic": topic,
            "research": state["research"],
            "report": state["report"][:6000],
            "feedback": state["feedback"][:4000]
        },
        name="Final Writer"
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print("\n")
    print(state["final_answer"])


    # =====================================================
    # RETURN COMPLETE STATE
    # =====================================================

    return state


# =========================================================
# PROGRAM START
# =========================================================

if __name__ == "__main__":

    topic = input("\nEnter a research topic: ")

    result = run_research_pipeline(topic)