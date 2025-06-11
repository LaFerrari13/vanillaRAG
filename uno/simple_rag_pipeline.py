
import os
import faiss
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from uuid import uuid4

# 1. Environment Setup
def setup_environment():
    """Set up Azure OpenAI environment variables"""
    os.environ["AZURE_OPENAI_API_KEY"] = "your-azure-openai-api-key"
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://your-resource.openai.azure.com/"
    os.environ["AZURE_OPENAI_API_VERSION"] = "2024-02-01"

# 2. Initialize Azure OpenAI components
def initialize_azure_components():
    """Initialize Azure OpenAI embeddings and chat models"""

    # Initialize embeddings model
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",  # or your deployment name
        azure_deployment="your-embedding-deployment-name"
    )

    # Initialize chat model
    llm = AzureChatOpenAI(
        azure_deployment="your-chat-deployment-name",  # e.g., "gpt-4"
        temperature=0.3,
        max_tokens=1000
    )

    return embeddings, llm

# 3. Create FAISS vector store
def create_vector_store(embeddings):
    """Create and initialize FAISS vector store"""

    # Create FAISS index
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

    # Initialize vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    return vector_store

# 4. Load and process documents
def load_and_process_documents(file_path, chunk_size=1000, chunk_overlap=200):
    """Load documents and split into chunks"""

    # Load documents (PDF or text)
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    splits = text_splitter.split_documents(documents)

    return splits

# 5. Create the RAG chain
def create_rag_chain(vector_store, llm):
    """Create the retrieval-augmented generation chain"""

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )

    # Create prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Create the document combination chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# 6. Main RAG pipeline class
class SimpleRAGPipeline:
    """Simple RAG pipeline using Azure OpenAI and FAISS"""

    def __init__(self):
        setup_environment()
        self.embeddings, self.llm = initialize_azure_components()
        self.vector_store = create_vector_store(self.embeddings)
        self.rag_chain = None

    def add_documents(self, file_path):
        """Add documents to the vector store"""

        # Process documents
        splits = load_and_process_documents(file_path)

        # Generate IDs for documents
        ids = [str(uuid4()) for _ in range(len(splits))]

        # Add to vector store
        self.vector_store.add_documents(documents=splits, ids=ids)

        # Create/update RAG chain
        self.rag_chain = create_rag_chain(self.vector_store, self.llm)

        print(f"Added {len(splits)} document chunks to vector store")

    def add_text_documents(self, texts, metadatas=None):
        """Add text documents directly to the vector store"""

        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {"source": f"document_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))

        ids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=ids)

        # Create/update RAG chain
        self.rag_chain = create_rag_chain(self.vector_store, self.llm)

        print(f"Added {len(documents)} text documents to vector store")

    def query(self, question):
        """Query the RAG pipeline"""

        if self.rag_chain is None:
            raise ValueError("No documents added to the pipeline. Add documents first.")

        # Get response from RAG chain
        response = self.rag_chain.invoke({"input": question})

        return {
            "answer": response["answer"],
            "source_documents": response["context"]
        }

    def save_vector_store(self, path):
        """Save the vector store to disk"""
        self.vector_store.save_local(path)
        print(f"Vector store saved to {path}")

    def load_vector_store(self, path):
        """Load a vector store from disk"""
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.rag_chain = create_rag_chain(self.vector_store, self.llm)
        print(f"Vector store loaded from {path}")

# 7. Usage example
def example_usage():
    """Example of how to use the RAG pipeline"""

    # Initialize pipeline
    rag = SimpleRAGPipeline()

    # Add some sample documents
    sample_texts = [
        "Azure OpenAI Service provides REST API access to OpenAI's powerful language models including GPT-4, GPT-3.5-turbo, and Embeddings model series.",
        "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
        "LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware and agentic.",
        "Retrieval-Augmented Generation (RAG) is a technique that combines the parametric knowledge of LLMs with non-parametric knowledge from external sources."
    ]

    metadatas = [
        {"source": "azure_docs", "topic": "openai"},
        {"source": "faiss_docs", "topic": "vector_search"},
        {"source": "langchain_docs", "topic": "framework"},
        {"source": "rag_docs", "topic": "technique"}
    ]

    rag.add_text_documents(sample_texts, metadatas)

    # Query the pipeline
    question = "What is FAISS and how does it work with vector search?"
    result = rag.query(question)

    print("Question:", question)
    print("Answer:", result["answer"])
    print("\nSource documents:")
    for i, doc in enumerate(result["source_documents"]):
        print(f"{i+1}. {doc.page_content[:100]}...")

    # Save vector store
    rag.save_vector_store("./my_faiss_index")

if __name__ == "__main__":
    example_usage()
