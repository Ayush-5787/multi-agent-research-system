"""
Streamlit UI for the Multi-Agent Research Pipeline
====================================================

Wraps `run_research_pipeline()` from pipeline.py with a web UI:

    1. Search Agent    -> finds recent, relevant sources
    2. Reader Agent     -> scrapes the most relevant source
    3. Writer           -> drafts a report from the research
    4. Critic           -> reviews the draft for gaps/errors
    5. Revision Writer  -> produces the final, polished answer

The pipeline is run on a background thread so the UI can keep showing
live progress (the same messages your terminal normally prints) while
it works, instead of just freezing for several minutes.

HOW TO RUN
----------
    pip install streamlit          # if not already installed
    streamlit run streamlit_app.py

Run this from the same folder as pipeline.py and agents.py, with any
API keys your agents need already available in the environment
(e.g. via a .env file - python-dotenv is loaded automatically below).
"""

import io
import queue
import threading
import time
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
# HELPERS
# =========================================================

class QueueWriter(io.TextIOBase):
    """File-like object that pushes text into a queue instead of a
    real stream, so a worker thread's print() output can be read live
    from Streamlit's main thread."""

    def __init__(self, log_queue: queue.Queue):
        self.log_queue = log_queue

    def write(self, text):
        if text:
            self.log_queue.put(text)
        return len(text)

    def flush(self):
        pass


def run_pipeline_worker(topic, log_queue, result_holder):
    """Runs the pipeline on a background thread and redirects its
    print() output into log_queue so the UI can display it live."""
    try:
        with redirect_stdout(QueueWriter(log_queue)):
            result_holder["result"] = run_research_pipeline(topic)
    except Exception as exc:
        result_holder["error"] = exc
    finally:
        result_holder["done"] = True


def render_results(result: dict):
    tab_final, tab_draft, tab_critic, tab_search, tab_scraped = st.tabs(
        ["✅ Final Report", "📝 Draft Report", "🧐 Critic Feedback",
         "🌐 Search Results", "📄 Scraped Content"]
    )

    with tab_final:
        st.markdown(result.get("final_answer") or "_No final answer produced._")
        st.download_button(
            "⬇️ Download final report (.md)",
            data=result.get("final_answer", ""),
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_draft:
        st.markdown(result.get("report") or "_No draft report produced._")

    with tab_critic:
        st.markdown(result.get("feedback") or "_No feedback produced._")

    with tab_search:
        st.text(result.get("search_results") or "No search results.")

    with tab_scraped:
        st.text(result.get("scraped_content") or "No scraped content.")


# =========================================================
# SESSION STATE
# =========================================================

st.session_state.setdefault("pipeline_running", False)
st.session_state.setdefault("result", None)
st.session_state.setdefault("log_lines", [])
st.session_state.setdefault("active_topic", "")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.header("About this pipeline")
    st.markdown(
        "**1. Search Agent** — finds recent, relevant sources\n\n"
        "**2. Reader Agent** — scrapes the best source in depth\n\n"
        "**3. Writer** — drafts a report from the research\n\n"
        "**4. Critic** — reviews the draft for gaps and errors\n\n"
        "**5. Revision Writer** — produces the final answer"
    )
    st.divider()
    st.caption(f"Session started {datetime.now().strftime('%H:%M:%S')}")


# =========================================================
# MAIN UI - INPUT
# =========================================================

st.title("🔎 Multi-Agent Research Assistant")
st.write(
    "Ask a research question and the agent pipeline will search the "
    "web, read the best source, draft a report, critique it, and hand "
    "back a polished final answer."
)

st.text_area(
    "Research topic / question",
    key="topic_input",
    placeholder="e.g. What are the latest breakthroughs in solid-state batteries?",
    height=80,
    disabled=st.session_state.pipeline_running,
)

start_clicked = st.button(
    "🚀 Run Research",
    type="primary",
    disabled=st.session_state.pipeline_running or not st.session_state.topic_input.strip(),
)

if start_clicked:
    st.session_state.active_topic = st.session_state.topic_input.strip()
    st.session_state.pipeline_running = True
    st.session_state.result = None
    st.session_state.log_lines = []
    st.rerun()


# =========================================================
# RUN PIPELINE (this script run blocks here, updating the UI live)
# =========================================================

if st.session_state.pipeline_running and st.session_state.result is None:

    log_queue: queue.Queue = queue.Queue()
    result_holder = {"done": False}

    worker = threading.Thread(
        target=run_pipeline_worker,
        args=(st.session_state.active_topic, log_queue, result_holder),
        daemon=True,
    )
    worker.start()

    with st.status("Running the pipeline... this can take a few minutes.", expanded=True) as status:
        log_box = st.empty()

        while not result_holder.get("done"):
            new_output = False
            while not log_queue.empty():
                st.session_state.log_lines.append(log_queue.get())
                new_output = True
            if new_output:
                log_box.code("".join(st.session_state.log_lines[-500:]), language=None)
            time.sleep(0.25)

        # Flush anything produced right at the very end
        while not log_queue.empty():
            st.session_state.log_lines.append(log_queue.get())
        log_box.code("".join(st.session_state.log_lines[-500:]), language=None)

        if "error" in result_holder:
            status.update(label="Pipeline failed ❌", state="error")
            st.session_state.pipeline_running = False
            st.error(f"The pipeline raised an error:\n\n```\n{result_holder['error']}\n```")
        else:
            status.update(label="Pipeline complete ✅", state="complete")
            st.session_state.result = result_holder["result"]
            st.session_state.pipeline_running = False

    st.rerun()


# =========================================================
# RESULTS
# =========================================================

if st.session_state.result is not None:
    st.divider()
    st.subheader(f"Results for: _{st.session_state.active_topic}_")
    render_results(st.session_state.result)

    if st.button("🔄 Start a new research"):
        st.session_state.result = None
        st.session_state.log_lines = []
        st.session_state.active_topic = ""
        st.rerun()