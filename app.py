"""
Streamlit UI for the Multi-Agent Research Pipeline

Pipeline:
1. Search Agent
2. Reader Agent
3. Writer
4. Critic
5. Revision Writer
"""

import io
import queue
import threading
import time
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =========================================================
# IMPORT PIPELINE
# =========================================================

from pipeline import run_research_pipeline


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔎",
    layout="wide",
)


# =========================================================
# SESSION STATE INITIALIZATION
# IMPORTANT: Do this BEFORE using any session_state values
# =========================================================

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

if "result" not in st.session_state:
    st.session_state.result = None

if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

if "active_topic" not in st.session_state:
    st.session_state.active_topic = ""

if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""


# =========================================================
# HELPERS
# =========================================================

class QueueWriter(io.TextIOBase):
    """
    Sends print() output from the worker thread
    into a queue so Streamlit can display it.
    """

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, text):
        if text:
            self.log_queue.put(text)
        return len(text)

    def flush(self):
        pass


def run_pipeline_worker(topic, log_queue, result_holder):
    """
    Run the research pipeline in a background thread.
    """

    try:
        with redirect_stdout(QueueWriter(log_queue)):
            result_holder["result"] = run_research_pipeline(topic)

    except Exception as exc:
        result_holder["error"] = exc

    finally:
        result_holder["done"] = True


def render_results(result):
    """
    Display all pipeline results in tabs.
    """

    tab_final, tab_draft, tab_critic, tab_search, tab_scraped = st.tabs(
        [
            "✅ Final Report",
            "📝 Draft Report",
            "🧐 Critic Feedback",
            "🌐 Search Results",
            "📄 Scraped Content",
        ]
    )

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    with tab_final:
        final_answer = result.get("final_answer", "")

        if final_answer:
            st.markdown(final_answer)
        else:
            st.info("No final answer was produced.")

        st.download_button(
            label="⬇️ Download final report (.md)",
            data=final_answer,
            file_name="research_report.md",
            mime="text/markdown",
        )

    # -----------------------------------------------------
    # DRAFT
    # -----------------------------------------------------

    with tab_draft:
        draft = result.get("report", "")

        if draft:
            st.markdown(draft)
        else:
            st.info("No draft report was produced.")

    # -----------------------------------------------------
    # CRITIC
    # -----------------------------------------------------

    with tab_critic:
        feedback = result.get("feedback", "")

        if feedback:
            st.markdown(feedback)
        else:
            st.info("No critic feedback was produced.")

    # -----------------------------------------------------
    # SEARCH RESULTS
    # -----------------------------------------------------

    with tab_search:
        search_results = result.get("search_results", "")

        if search_results:
            st.text(search_results)
        else:
            st.info("No search results available.")

    # -----------------------------------------------------
    # SCRAPED CONTENT
    # -----------------------------------------------------

    with tab_scraped:
        scraped_content = result.get("scraped_content", "")

        if scraped_content:
            st.text(scraped_content)
        else:
            st.info("No scraped content available.")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🔎 About this Pipeline")

    st.markdown(
        """
        **1. Search Agent**  
        Finds recent and relevant sources.

        **2. Reader Agent**  
        Reads and extracts useful information.

        **3. Writer**  
        Creates a research draft.

        **4. Critic**  
        Checks the draft for gaps and errors.

        **5. Revision Writer**  
        Produces the final polished answer.
        """
    )

    st.divider()

    st.caption(
        f"Session started "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )


# =========================================================
# MAIN UI
# =========================================================

st.title("🔎 Multi-Agent Research Assistant")

st.write(
    "Ask a research question and the multi-agent pipeline will "
    "search the web, read the best source, draft a report, "
    "critique it, and produce a polished final answer."
)


# =========================================================
# TOPIC INPUT
# =========================================================

st.text_area(
    "Research topic / question",
    key="topic_input",
    placeholder=(
        "e.g. What are the latest breakthroughs "
        "in solid-state batteries?"
    ),
    height=100,
    disabled=st.session_state.pipeline_running,
)


# =========================================================
# RUN BUTTON
# =========================================================

topic_is_valid = bool(
    st.session_state.topic_input.strip()
)

start_clicked = st.button(
    "🚀 Run Research",
    type="primary",
    disabled=(
        st.session_state.pipeline_running
        or not topic_is_valid
    ),
)


# =========================================================
# START PIPELINE
# =========================================================

if start_clicked:

    st.session_state.active_topic = (
        st.session_state.topic_input.strip()
    )

    st.session_state.pipeline_running = True
    st.session_state.result = None
    st.session_state.log_lines = []

    st.rerun()


# =========================================================
# RUN PIPELINE
# =========================================================

if (
    st.session_state.pipeline_running
    and st.session_state.result is None
):

    log_queue = queue.Queue()

    result_holder = {
        "done": False
    }

    worker = threading.Thread(
        target=run_pipeline_worker,
        args=(
            st.session_state.active_topic,
            log_queue,
            result_holder,
        ),
        daemon=True,
    )

    worker.start()

    with st.status(
        "Running the research pipeline...",
        expanded=True,
    ) as status:

        log_box = st.empty()

        # -------------------------------------------------
        # SHOW LIVE LOGS
        # -------------------------------------------------

        while not result_holder.get("done"):

            new_output = False

            while not log_queue.empty():

                output = log_queue.get()

                st.session_state.log_lines.append(
                    output
                )

                new_output = True

            if new_output:

                log_box.code(
                    "".join(
                        st.session_state.log_lines[-500:]
                    )
                )

            time.sleep(0.25)

        # -------------------------------------------------
        # FLUSH REMAINING LOGS
        # -------------------------------------------------

        while not log_queue.empty():

            st.session_state.log_lines.append(
                log_queue.get()
            )

        log_box.code(
            "".join(
                st.session_state.log_lines[-500:]
            )
        )

        # -------------------------------------------------
        # PIPELINE ERROR
        # -------------------------------------------------

        if "error" in result_holder:

            status.update(
                label="Pipeline failed ❌",
                state="error",
            )

            st.session_state.pipeline_running = False

            st.error(
                "The research pipeline raised an error:"
            )

            st.exception(
                result_holder["error"]
            )

        # -------------------------------------------------
        # PIPELINE SUCCESS
        # -------------------------------------------------

        else:

            status.update(
                label="Pipeline complete ✅",
                state="complete",
            )

            st.session_state.result = (
                result_holder.get("result")
            )

            st.session_state.pipeline_running = False

    st.rerun()


# =========================================================
# RESULTS
# =========================================================

if st.session_state.result is not None:

    st.divider()

    st.subheader(
        f"Results for: "
        f"_{st.session_state.active_topic}_"
    )

    render_results(
        st.session_state.result
    )

    st.divider()

    if st.button("🔄 Start a New Research"):

        st.session_state.result = None
        st.session_state.log_lines = []
        st.session_state.active_topic = ""
        st.session_state.topic_input = ""

        st.rerun()