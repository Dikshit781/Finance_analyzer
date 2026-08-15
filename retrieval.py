import os
import re
from datetime import datetime

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CHROMA_FOLDER = os.getenv("CHROMA_FOLDER", "db/chroma")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2
)


# --------------------------------------------------
# Conversation Memory
# --------------------------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=False
)


# --------------------------------------------------
# Load Vectorstore
# --------------------------------------------------

def load_vectorstore():
    """
    Load the existing Chroma vector database.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_FOLDER,
        embedding_function=embedding_model
    )

    return vectorstore


# --------------------------------------------------
# Detect Company
# --------------------------------------------------

def detect_company(query):
    """
    Detect a supported company symbol from the query.
    """

    query_lower = query.lower()

    company_map = {
        "apple": "AAPL",
        "apples": "AAPL",
        "aapl": "AAPL",

        "amazon": "AMZN",
        "amzn": "AMZN",

        "google": "GOOGL",
        "alphabet": "GOOGL",
        "googl": "GOOGL",

        "microsoft": "MSFT",
        "msft": "MSFT",

        "meta": "META",
        "facebook": "META",

        "nvidia": "NVDA",
        "nvda": "NVDA",

        "tesla": "TSLA",
        "tsla": "TSLA",

        "walmart": "WMT",
        "wmt": "WMT"
    }

    for name, symbol in company_map.items():
        if name in query_lower:
            return symbol

    return None


# --------------------------------------------------
# Detect Document Type
# --------------------------------------------------

def detect_document_type(query):
    """
    Detect which financial document type is relevant.
    """

    query_lower = query.lower()

    growth_keywords = [
        "growth",
        "revenue growth",
        "income growth",
        "profit growth",
        "eps growth"
    ]

    cash_flow_keywords = [
        "cash flow",
        "operating cash flow",
        "free cash flow",
        "capital expenditure",
        "capex"
    ]

    income_keywords = [
        "revenue",
        "sales",
        "net income",
        "gross profit",
        "operating income",
        "operating profit",
        "ebit",
        "ebitda",
        "eps",
        "earnings"
    ]

    balance_keywords = [
        "asset",
        "assets",
        "liability",
        "liabilities",
        "equity",
        "debt",
        "inventory",
        "receivable",
        "receivables",
        "payable",
        "payables"
    ]

    ratio_keywords = [
        "ratio",
        "margin",
        "roe",
        "roa",
        "return on equity",
        "return on assets",
        "price to earnings",
        "p/e"
    ]

    if any(keyword in query_lower for keyword in growth_keywords):
        return "growth"

    if any(keyword in query_lower for keyword in cash_flow_keywords):
        return "cash_flow"

    if any(keyword in query_lower for keyword in income_keywords):
        return "income_statement"

    if any(keyword in query_lower for keyword in balance_keywords):
        return "balance_sheet"

    if any(keyword in query_lower for keyword in ratio_keywords):
        return "ratios"

    return None


# --------------------------------------------------
# Detect Explicit / Relative Year
# --------------------------------------------------

def detect_requested_year(query):
    """
    Resolve year expressions deterministically.

    Examples:
    - "in 2024" -> 2024
    - "previous year" -> current year - 1
    - "last year" -> current year - 1

    Latest-year wording is handled separately from DB metadata.
    """

    query_lower = query.lower()

    explicit_year = re.search(
        r"\b(?:19|20)\d{2}\b",
        query
    )

    if explicit_year:
        return explicit_year.group(0)

    previous_terms = [
        "previous year",
        "previous financial year",
        "previous fiscal year",
        "last year"
    ]

    if any(term in query_lower for term in previous_terms):
        return str(datetime.now().year - 1)

    return None


# --------------------------------------------------
# Get Available Fiscal Years
# --------------------------------------------------

def get_available_fiscal_years(
    vectorstore,
    company,
    document_type=None
):
    """
    Get available fiscal years for a company/document type.
    """

    if not company:
        return []

    filters = [
        {
            "company": company
        }
    ]

    if document_type:
        filters.append(
            {
                "document_type": document_type
            }
        )

    if len(filters) > 1:
        metadata_filter = {
            "$and": filters
        }
    else:
        metadata_filter = filters[0]

    records = vectorstore.similarity_search(
        query=f"{company} annual fiscal year",
        k=100,
        filter=metadata_filter
    )

    years = set()

    for record in records:
        value = record.metadata.get("fiscal_year")

        if value is None:
            continue

        value = str(value).strip()

        if value.isdigit():
            years.add(int(value))

    return sorted(
        years,
        reverse=True
    )


# --------------------------------------------------
# Resolve Latest Year
# --------------------------------------------------

def resolve_latest_year(
    query,
    vectorstore,
    company,
    document_type
):
    """
    Resolve latest/most recent year from DB metadata.
    """

    query_lower = query.lower()

    latest_terms = [
        "latest",
        "latest year",
        "latest financial year",
        "latest fiscal year",
        "most recent",
        "most recent year",
        "most recent financial year",
        "most recent fiscal year"
    ]

    if not any(term in query_lower for term in latest_terms):
        return None

    years = get_available_fiscal_years(
        vectorstore=vectorstore,
        company=company,
        document_type=document_type
    )

    if not years:
        return None

    return str(years[0])


# --------------------------------------------------
# Retrieve Financial Records
# --------------------------------------------------

def retrieve_chunks(query, k=8):
    """
    Retrieve relevant financial records using metadata filters.
    """

    vectorstore = load_vectorstore()

    company = detect_company(query)
    document_type = detect_document_type(query)

    requested_year = detect_requested_year(query)

    if requested_year is None:
        requested_year = resolve_latest_year(
            query=query,
            vectorstore=vectorstore,
            company=company,
            document_type=document_type
        )

    filters = []

    if company:
        filters.append(
            {
                "company": company
            }
        )

    if document_type:
        filters.append(
            {
                "document_type": document_type
            }
        )

    if requested_year:
        filters.append(
            {
                "fiscal_year": requested_year
            }
        )

    if len(filters) > 1:

        metadata_filter = {
            "$and": filters
        }

        results = vectorstore.similarity_search(
            query=query,
            k=k,
            filter=metadata_filter
        )

    elif len(filters) == 1:

        results = vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filters[0]
        )

    else:

        results = vectorstore.similarity_search(
            query=query,
            k=k
        )

    return results, {
        "company": company,
        "document_type": document_type,
        "requested_year": requested_year
    }


# --------------------------------------------------
# Build Context
# --------------------------------------------------

def build_context(chunks):
    """
    Combine retrieved financial records into one context string.
    """

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        context_parts.append(
            f"""
Financial Record {index}

{chunk.page_content}
"""
        )

    return "\n".join(context_parts)


def extract_direct_financial_answer(
    question,
    chunks,
    resolved_company=None,
    resolved_year=None
):
    """
    Extract exact structured financial values directly
    from retrieved financial records.

    Returns None when the question is not a supported
    direct metric query.
    """

    query_lower = question.lower()

    metric_map = {
        "revenue": "Revenue",
        "sales": "Revenue",

        "net income": "Net Income",
        "gross profit": "Gross Profit",

        "operating income": "Operating Income",
        "operating profit": "Operating Income",

        "ebitda": "Ebitda",
        "ebit": "Ebit",

        "eps": "Eps",

        "total assets": "Total Assets",
        "assets": "Total Assets",

        "total liabilities": "Total Liabilities",
        "liabilities": "Total Liabilities",

        "total debt": "Total Debt",
        "debt": "Total Debt",

        "inventory": "Inventory",

        "free cash flow": "Free Cash Flow",
        "operating cash flow": "Operating Cash Flow"
    }

    selected_metric = None
    selected_label = None

    # Longer phrases should win before shorter ones
    sorted_metrics = sorted(
        metric_map.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for user_term, record_label in sorted_metrics:

        if user_term in query_lower:

            selected_metric = user_term
            selected_label = record_label

            break

    if not selected_label:
        return None

    for chunk in chunks:

        lines = chunk.page_content.splitlines()

        for line in lines:

            if ":" not in line:
                continue

            key, value = line.split(":", 1)

            key = key.strip()
            value = value.strip()

            if key.lower() != selected_label.lower():
                continue

            try:
                numeric_value = float(value)
            except ValueError:
                return None

            # ------------------------------------------
            # Format value
            # ------------------------------------------

            if selected_label.lower() == "eps":

                formatted_value = f"${numeric_value:,.2f}"

            elif abs(numeric_value) >= 1_000_000_000:

                formatted_value = (
                    f"${numeric_value / 1_000_000_000:,.3f} billion"
                )

            elif abs(numeric_value) >= 1_000_000:

                formatted_value = (
                    f"${numeric_value / 1_000_000:,.3f} million"
                )

            else:

                formatted_value = (
                    f"${numeric_value:,.2f}"
                )

            company_text = (
                resolved_company
                if resolved_company
                else "The company"
            )

            year_text = (
                f" for fiscal year {resolved_year}"
                if resolved_year
                else ""
            )

            readable_metric = (
                selected_label
                .replace("Ebitda", "EBITDA")
                .replace("Ebit", "EBIT")
                .replace("Eps", "EPS")
            )

            return (
                f"{company_text}'s {readable_metric}"
                f"{year_text} was {formatted_value}."
            )

    return None


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------

def generate_answer(
    question,
    context,
    resolved_company=None,
    resolved_document_type=None,
    resolved_year=None
):
    """
    Generate the answer using only retrieved financial context.
    """

    memory_data = memory.load_memory_variables({})

    chat_history = memory_data.get(
        "chat_history",
        ""
    )

    current_year = datetime.now().year
    previous_year = current_year - 1

    prompt = f"""
You are an expert Financial Analysis Assistant.

Use ONLY the retrieved financial context for financial facts.

The year has already been resolved by Python before this prompt.
Do NOT reinterpret relative year phrases.

Resolved request:
- Company: {resolved_company or "Not explicitly detected"}
- Document Type: {resolved_document_type or "Not explicitly detected"}
- Fiscal Year: {resolved_year or "Not explicitly requested"}

Calendar reference:
- Current calendar year: {current_year}
- Previous calendar year: {previous_year}

Important rules:

1. Never invent financial values.
2. Never use outside financial knowledge.
3. If a resolved fiscal year is provided above, answer ONLY from that fiscal year.
4. Do NOT change the resolved fiscal year.
5. Do NOT decide that "previous year" means a different year.
6. For revenue, sales, gross profit, operating income, net income,
   EBIT, EBITDA, or EPS, use Income Statement information.
7. For assets, liabilities, debt, equity, or inventory,
   use Balance Sheet information.
8. For operating cash flow, free cash flow, or CAPEX,
   use Cash Flow information.
9. For ratios and margins, use Ratio information.
10. For growth questions, use Growth information.
11. Never mix companies.
12. Never mix fiscal years unless comparison is explicitly requested.
13. Never mix quarterly and yearly values unless requested.
14. Never mention chunks, retrieval, embeddings, metadata,
    vector databases, or internal implementation details.
15. Present large currency values in a readable format.
16. If the required value is not present in the retrieved context, say:
    "The requested information is not available in the provided documents."

Previous conversation:
{chat_history}

Retrieved financial context:
{context}

Current question:
{question}

Answer:
"""

    try:

        response = llm.invoke(prompt)

        answer = response.content.strip()

        memory.save_context(
            {
                "question": question
            },
            {
                "answer": answer
            }
        )

        return answer

    except Exception as error:

        return (
            "Could not connect to Ollama. "
            "Start the Ollama application or run 'ollama serve', "
            "then try again.\n"
            f"Technical error: {error}"
        )


# --------------------------------------------------
# Ask Questions
# --------------------------------------------------

def ask():
    """
    Run the financial question-answer loop.
    """

    while True:

        question = input(
            "\nAsk a financial question "
            "(type 'back' for menu, 'clear' to clear memory, "
            "or 'exit' to close): "
        ).strip()

        if not question:

            print(
                "\nPlease enter a question."
            )

            continue

        command = question.lower()

        if command == "back":
            return "menu"

        if command == "exit":

            print(
                "\nThank you for using the Finance Analyzer!"
            )

            return "exit"

        if command == "clear":

            memory.clear()

            print(
                "\nConversation memory cleared."
            )

            continue

        debug = False

        if command.startswith("debug "):

            debug = True
            question = question[6:].strip()

            if not question:

                print(
                    "\nEnter a question after 'debug'."
                )

                continue

        # IMPORTANT:
        # Retrieval uses ONLY the current question.
        # Previous chat history is NOT injected into retrieval.
        chunks, resolved = retrieve_chunks(
            question
        )

        if not chunks:

            print(
                "\nNo relevant financial information found."
            )

            continue

        if debug:

            print(
                "\nResolved Request"
            )

            print(
                "-" * 60
            )

            print(
                "Company:",
                resolved["company"]
            )

            print(
                "Document Type:",
                resolved["document_type"]
            )

            print(
                "Fiscal Year:",
                resolved["requested_year"]
            )

            print(
                "\nRetrieved Financial Records"
            )

            print(
                "-" * 60
            )

            for index, chunk in enumerate(
                chunks,
                start=1
            ):

                print(
                    f"\nFinancial Record {index}"
                )

                print(
                    "Metadata:",
                    chunk.metadata
                )

                print(
                    chunk.page_content
                )

        # --------------------------------------------------
        # Try exact structured answer first
        # --------------------------------------------------

        direct_answer = extract_direct_financial_answer(
            question=question,
            chunks=chunks,
            resolved_company=resolved["company"],
            resolved_year=resolved["requested_year"]
        )

        if direct_answer:

            answer = direct_answer

            memory.save_context(
                {
                    "question": question
                },
                {
                    "answer": answer
                }
            )

        else:

            # ----------------------------------------------
            # Use LLM for analytical / complex questions
            # ----------------------------------------------

            context = build_context(
                chunks
            )

            answer = generate_answer(
                question=question,
                context=context,
                resolved_company=resolved["company"],
                resolved_document_type=resolved["document_type"],
                resolved_year=resolved["requested_year"]
            )

        print(
            "\nAnswer"
        )

        print(
            "-" * 60
        )

        print(
            answer
        )