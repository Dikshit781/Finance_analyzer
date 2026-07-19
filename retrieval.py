import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.memory import ConversationBufferMemory


load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

CHROMA_FOLDER = os.getenv("CHROMA_FOLDER")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.2
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=False
)

# --------------------------------------------------
# Load vectorstore from Chromadb
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
# Retrieve Chunks from ChromaDB
# --------------------------------------------------

def retrieve_chunks(query, k=5):
    """
    Retrieve the most relevant chunks for a user query.

    Args:
        query: User's financial question.
        k: Number of chunks to retrieve.

    Returns:
        List of relevant LangChain Document objects.
    """

    vectorstore = load_vectorstore()

    results = vectorstore.similarity_search(
        query=query,
        k=k
    )

    return results

# --------------------------------------------------
# Generate Context using Ollama.
# --------------------------------------------------

def build_context(chunks):
    """
    Combine all retrieved chunks into one context string.

    Args:
        chunks: List of retrieved documents.

    Returns:
        str: Combined context.
    """

    context = ""

    for index, chunk in enumerate(chunks, start=1):

        context += f"Chunk {index}\n"

        context += chunk.page_content

        context += "\n"

        context += "-" * 80

        context += "\n\n"

    return context

# --------------------------------------------------
# Generate answer using Ollama.
# --------------------------------------------------

def generate_answer(question, context):
    """
    Generate an answer using retrieved financial context
    and previous conversation history.
    """

    memory_data = memory.load_memory_variables({})

    chat_history = memory_data.get(
        "chat_history",
        ""
    )

    prompt = f"""
You are an expert Financial Analysis Assistant.

Use ONLY the retrieved financial context for financial facts.

Use the previous conversation to understand follow-up references such as:
- it
- its
- that company
- previous company
- previous value

Rules:
1. Never use outside knowledge.
2. Never guess missing values.
3. Do not compare values unless the user explicitly asks.
4. Do not compare different periods unless the user asks.
5. Never mention chunks, metadata, retrieval, or internal documents.
6. If the requested information is unavailable, say:
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

        answer = response.content

        memory.save_context(
            {"question": question},
            {"answer": answer}
        )

        return answer

    except Exception as error:
        return (
            "Could not connect to Ollama. Start the Ollama application "
            "or run 'ollama serve', then try again.\n"
            f"Technical error: {error}"
        )
    
# --------------------------------------------------
# Ask questions.
# --------------------------------------------------

def create_retrieval_query(question):
    """
    Add previous conversation history to the current question
    for better follow-up retrieval.
    """

    memory_data = memory.load_memory_variables({})

    chat_history = memory_data.get(
        "chat_history",
        ""
    )

    if not chat_history:
        return question

    return f"""
Previous conversation:
{chat_history}

Current question:
{question}
"""

# --------------------------------------------------
# Ask questions.
# --------------------------------------------------
def ask():
    """
    Run the financial question-answer loop with conversation memory.
    """

    while True:

        question = input(
            "\nAsk a financial question "
            "(type 'back' for menu, 'clear' to clear memory, "
            "or 'exit' to close): "
        ).strip()

        if not question:
            print("\nPlease enter a question.")
            continue

        command = question.lower()

        if command == "back":
            return "menu"

        if command == "exit":
            print("\nThank you for using the Finance Analyzer!")
            return "exit"

        if command == "clear":
            memory.clear()
            print("\nConversation memory cleared.")
            continue

        debug = False

        if command.startswith("debug "):
            debug = True
            question = question[6:].strip()

            if not question:
                print("\nEnter a question after 'debug'.")
                continue

        retrieval_query = create_retrieval_query(question)

        chunks = retrieve_chunks(
            retrieval_query
        )

        if not chunks:
            print("\nNo relevant financial information found.")
            continue

        if debug:

            print("\nRetrieved Chunks")
            print("-" * 60)

            for index, chunk in enumerate(chunks, start=1):

                print(f"\nChunk {index}")
                print("Metadata:", chunk.metadata)
                print(chunk.page_content)

        context = build_context(chunks)

        answer = generate_answer(
            question=question,
            context=context
        )

        print("\nAnswer")
        print("-" * 60)
        print(answer)