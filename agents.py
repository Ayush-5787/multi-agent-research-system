from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# MODEL SETUP
# =========================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_format="hidden",
    max_completion_tokens=1000
)


# =========================================================
# 1. SEARCH AGENT
# =========================================================

def build_search_agent():

    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="""
You are an expert web research agent.

Your job is to search the web for recent and reliable
information about the exact topic provided by the user.

IMPORTANT:
- Focus only on the user's topic.
- Do not search unrelated topics.
- Use the web_search tool.
- Prefer recent sources.
- Return useful facts and source URLs.
- Do not make up information.

Your response should contain:
1. Important findings
2. Source titles
3. Source URLs
4. Relevant details
"""
    )


# =========================================================
# 2. READER AGENT
# =========================================================

def build_reader_agent():

    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt="""
You are an expert research reading agent.

Your job is to read webpages and extract useful information
for the research question.

Use the scrape_url tool to read the provided URL.

IMPORTANT:
- Only read URLs related to the requested topic.
- Extract factual information.
- Preserve important dates, names, numbers and events.
- Do not make up information.

Return:
- Important facts
- Main points
- Dates
- Names
- Statistics
- Useful source information
"""
    )


# =========================================================
# 3. WRITER CHAIN
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert research writer.

Your job is to answer the user's research question using
ONLY the research provided.

Create a complete, useful and factual answer.

Do not invent information.

Every important claim should be supported by the provided
research.

Keep the answer focused on the user's question.
"""
    ),
    (
        "human",
        """
User's Research Question:
{topic}

Research Gathered:
{research}

Write the answer using this structure:

# Answer

Give a short direct introduction answering the question.

# Latest Key Developments

Give 4 to 6 important developments.

For each development include:
- What happened
- Important people/teams/organizations
- Date if available
- Important result or statistic
- Why it matters

# Conclusion

Give a short summary.

# Sources

List the source title and URL for every source used.

IMPORTANT:
- Do NOT leave sections unfinished.
- Do NOT create fake sources.
- Do NOT invent URLs.
- Answer the user's actual question.
- Make the answer complete but concise.
"""
    )
])

writer_chain = writer_prompt | llm | StrOutputParser()


# =========================================================
# 4. CRITIC CHAIN
# =========================================================

critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a strict research report critic.

Check whether the report actually answers the user's
research question.

Be specific and constructive.

Focus on:
- Accuracy
- Completeness
- Relevance
- Missing information
- Missing sources
- Unsupported claims
- Formatting
"""
    ),
    (
        "human",
        """
Research Question:
{topic}

Report:
{report}

Review the report.

Respond in this format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

Required Changes:
- ...
- ...

One line verdict:
...
"""
    )
])

critic_chain = critic_prompt | llm | StrOutputParser()


# =========================================================
# 5. REVISION WRITER
# =========================================================

revision_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the final research editor.

Your job is to improve the research report using the
critic's feedback.

The final answer must directly answer the user's question.

Use only information contained in the original report
and research.

Do not invent facts or sources.

Fix incomplete sections and improve clarity.

Do not mention the critic or the revision process
in the final answer.
"""
    ),
    (
        "human",
        """
Research Question:
{topic}

Original Research:
{research}

Original Report:
{report}

Critic Feedback:
{feedback}

Now produce the FINAL research answer.

Use this structure:

# Answer

Directly answer the user's question.

# Latest Key Developments

Give the most important developments relevant to
the question.

# Conclusion

Summarize the answer.

# Sources

List the source title and URL.

IMPORTANT:
- Complete every section.
- Do not leave unfinished points.
- Do not invent information.
- Do not invent URLs.
- Remove irrelevant information.
- Keep the answer focused on the user's question.
"""
    )
])

revision_chain = revision_prompt | llm | StrOutputParser()