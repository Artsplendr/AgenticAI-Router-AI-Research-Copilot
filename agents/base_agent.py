"""
Base RAG agent – shared retrieval + LLM logic for paper, web_docs, and notes agents.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


SYSTEM_RAG = """You are a helpful research assistant. Answer the user's question using only the following context.

You have been given context that may include multiple papers (titles and abstracts or excerpts). When the context contains relevant papers or material, you must use it to answer the question—synthesize from the given titles and content. Do not respond that "the context does not provide enough information" when multiple relevant items are present; answer from what is provided. Only say the context does not contain enough information when the context is truly empty or irrelevant (e.g. no papers or no related content).

Formatting requirements:
- Use clear Markdown structure with short section headers.
- For compare/contrast questions (e.g. academic vs practical), use this structure:
  1) **Academic perspective**
  2) **Practical perspective**
  3) **Comparison summary**
- Keep each section focused and easy to read (short paragraphs or concise bullets).
- Do not bold arbitrary words; only use bold for section headers or exact key terms from the user's query.

Context:
{context}

Question: {question}"""

# For RAG over the user's own ingested notes: answer only from notes; do not invent or add external knowledge.
SYSTEM_RAG_NOTES = """You are a helpful research assistant. Answer the user's question based only on the following notes (the user's ingested documents and notes). If the notes do not contain enough information to answer, say so briefly and do not add information from outside the notes.

Notes:
{context}

Question: {question}"""

# For pre-assembled context (papers or web search): answer only from this context.
SYSTEM_RAG_WITH_SUPPLEMENT = """You are a helpful research assistant. Answer the user's question using only the following context (from papers or web search).

When the context includes multiple papers or search results, you must use them to answer—synthesize from the given content. Do not say "the context does not provide enough information" when relevant material is present; answer from what is provided. Only say the context does not contain enough information when it is truly empty or irrelevant. Do not add information from outside the context.

Formatting requirements:
- Use clear Markdown structure with short section headers.
- For compare/contrast questions (e.g. academic vs practical), use this structure:
  1) **Academic perspective**
  2) **Practical perspective**
  3) **Comparison summary**
- Keep each section focused and easy to read (short paragraphs or concise bullets).
- Do not bold arbitrary words; only use bold for section headers or exact key terms from the user's query.

Context:
{context}

Question: {question}"""


def get_llm():
    """Chat model from env (OpenAI)."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def run_rag(query: str, retriever: BaseRetriever, llm=None, *, notes_mode: bool = False) -> str:
    """Run RAG: retrieve docs, then generate answer with LLM. Use notes_mode=True for the notes agent (user's ingested notes)."""
    llm = llm or get_llm()
    template = SYSTEM_RAG_NOTES if notes_mode else SYSTEM_RAG
    prompt = ChatPromptTemplate.from_messages([("human", template)])
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)


def run_rag_with_context(
    query: str,
    context: str,
    llm=None,
    *,
    allow_supplement: bool = False,
) -> str:
    """Run RAG when context is already assembled (e.g. retriever + Tavily).
    If allow_supplement is True (e.g. for web search context), the model may extend partial context with brief general knowledge for a complete answer."""
    if not context or not context.strip():
        context = "(No retrieved context available.)"
    llm = llm or get_llm()
    template = SYSTEM_RAG_WITH_SUPPLEMENT if allow_supplement else SYSTEM_RAG
    prompt = ChatPromptTemplate.from_messages([("human", template)])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)
