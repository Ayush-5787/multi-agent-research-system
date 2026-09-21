from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import web_search, scrape_url


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# GROQ
# PRIMARY PROVIDER FOR SEARCH + READER
# =========================================================

groq_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=800,
)


# =========================================================
# OPENROUTER
# FALLBACK PROVIDER FOR SEARCH + READER
# =========================================================

openrouter_agent_llm = ChatOpenRouter(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=800,
)


# =========================================================
# OPENROUTER
# WRITER + CRITIC + REVISION
# =========================================================

chain_llm = ChatOpenRouter(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=1200,
)


# =========================================================
# SEARCH AGENT
# PRIMARY → GROQ
# =========================================================

def build_search_agent():

    return create_agent(
        model=groq_llm,
        tools=[web_search],

        system_prompt="""
You are an expert web research agent.

Your task is to research the exact question provided by
the user.

Rules:

1. Focus only on the user's question.
2. Use the web_search tool.
3. Prefer recent and reliable sources.
4. Do not search unrelated topics.
5. Do not invent information.
6. Return useful factual information.
7. Include source titles and URLs.
8. Keep the response concise.

Return:

Important findings
Source titles
Source URLs
Relevant dates
Relevant facts
"""
    )


# =========================================================
# SEARCH AGENT
# FALLBACK → OPENROUTER
# =========================================================

def build_openrouter_search_agent():

    return create_agent(
        model=openrouter_agent_llm,
        tools=[web_search],

        system_prompt="""
You are an expert web research agent.

Your task is to research the exact question provided by
the user.

Rules:

1. Focus only on the user's question.
2. Use the web_search tool.
3. Prefer recent and reliable sources.
4. Do not search unrelated topics.
5. Do not invent information.
6. Return useful factual information.
7. Include source titles and URLs.
8. Keep the response concise.

Return:

Important findings
Source titles
Source URLs
Relevant dates
Relevant facts
"""
    )


# =========================================================
# READER AGENT
# PRIMARY → GROQ
# =========================================================

def build_reader_agent():

    return create_agent(
        model=groq_llm,
        tools=[scrape_url],

        system_prompt="""
You are an expert research reading agent.

Your task is to read the most relevant webpage
identified by the search process.

Rules:

1. Use the scrape_url tool.
2. Choose a URL relevant to the research question.
3. Extract only useful factual information.
4. Preserve important dates, names and numbers.
5. Do not invent information.
6. Ignore advertisements and irrelevant content.
7. Do not repeat unnecessary webpage text.
8. Keep the response concise.

Return:

Important facts
Main points
Dates
Names
Statistics
Source information
"""
    )


# =========================================================
# READER AGENT
# FALLBACK → OPENROUTER
# =========================================================

def build_openrouter_reader_agent():

    return create_agent(
        model=openrouter_agent_llm,
        tools=[scrape_url],

        system_prompt="""
You are an expert research reading agent.

Your task is to read the most relevant webpage
identified by the search process.

Rules:

1. Use the scrape_url tool.
2. Choose a URL relevant to the research question.
3. Extract only useful factual information.
4. Preserve important dates, names and numbers.
5. Do not invent information.
6. Ignore advertisements and irrelevant content.
7. Do not repeat unnecessary webpage text.
8. Keep the response concise.

Return:

Important facts
Main points
Dates
Names
Statistics
Source information
"""
    )


# =========================================================
# WRITER CHAIN
# OPENROUTER
# =========================================================

writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research report writer.

Answer the user's research question using ONLY the
research information supplied to you.

Rules:

1. Do not invent facts.
2. Do not invent sources.
3. Do not invent URLs.
4. Do not use unrelated information.
5. Important claims must be supported by the research.
6. Keep the report clear and factual.
7. If the research does not contain enough information,
   do not make up missing information.

Use this structure:

# Answer

Give a direct answer to the research question.

# Latest Key Developments

Include 4-6 developments when supported by the research.

For each development include:

- What happened
- People, teams or organizations involved
- Date
- Result or important statistic
- Why it matters

# Conclusion

Give a concise conclusion based only on the research.

# Sources

List the sources actually present in the research.

Do not create fake sources.
"""
        ),
        (
            "human",
            """
Research question:

{topic}

Research:

{research}
"""
        ),
    ]
)

writer_chain = (
    writer_prompt
    | chain_llm
    | StrOutputParser()
)


# =========================================================
# CRITIC CHAIN
# OPENROUTER
# =========================================================

critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict research report critic.

Review the draft report against the supplied research.

Check:

1. Accuracy
2. Completeness
3. Relevance
4. Missing information
5. Missing sources
6. Unsupported claims
7. Incorrect dates
8. Incorrect numbers
9. Formatting

Do not invent corrections.

Return:

Score: X/10

Strengths:
- ...

Areas to Improve:
- ...

Required Changes:
- ...

Verdict:
One concise sentence.
"""
        ),
        (
            "human",
            """
Research question:

{topic}

Research:

{research}

Draft report:

{report}
"""
        ),
    ]
)

critic_chain = (
    critic_prompt
    | chain_llm
    | StrOutputParser()
)


# =========================================================
# REVISION WRITER
# OPENROUTER
# =========================================================

revision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the final research report writer.

Improve the original report using the critic's feedback.

Rules:

1. Use only information from the supplied research.
2. Do not invent facts.
3. Do not invent sources.
4. Do not invent URLs.
5. Correct unsupported claims.
6. Add information only when it exists in the research.
7. Do not mention the critic.
8. Do not mention the revision process.
9. Keep the answer factual and concise.

Use exactly this structure:

# Answer

# Latest Key Developments

# Conclusion

# Sources
"""
        ),
        (
            "human",
            """
Research question:

{topic}

Original research:

{research}

Original report:

{report}

Critic feedback:

{feedback}
"""
        ),
    ]
)

revision_chain = (
    revision_prompt
    | chain_llm
    | StrOutputParser()
)