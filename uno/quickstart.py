
import os
from dotenv import load_dotenv
from simple_rag_pipeline import SimpleRAGPipeline

# Load environment variables from .env file
load_dotenv()

def main():
    """Quick start example for RAG pipeline"""

    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    rag = SimpleRAGPipeline()

    # Add some sample documents
    sample_documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without explicit programming.",
        "Python is a high-level programming language known for its simplicity and readability, making it popular for data science and AI applications.",
        "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
        "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language."
    ]

    print("Adding documents to vector store...")
    rag.add_text_documents(sample_documents)

    # Ask questions
    questions = [
        "What is machine learning?",
        "Why is Python popular for AI?",
        "How do neural networks work?"
    ]

    print("\n" + "="*50)
    print("QUERYING THE RAG PIPELINE")
    print("="*50)

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 40)

        result = rag.query(question)
        print(f"Answer: {result['answer']}")

        print("\nSources:")
        for i, doc in enumerate(result['source_documents'][:2]):  # Show top 2 sources
            print(f"  {i+1}. {doc.page_content[:80]}...")
        print()

if __name__ == "__main__":
    main()
