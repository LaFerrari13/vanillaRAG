# Initialize
from rag_pipeline import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

rag = RAGPipeline(os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY"), os.getenv("AZURE_OPENAI_API_VERSION"))

# Load PDFs
rag.load_and_process_pdfs(["cellstructure.pdf"])
rag.save_vectorstore("./vectorstore")

# Query
result = rag.query("What are organalles?")
print(result["answer"])

    
# Query the system
# while True:
#     question = input("\nEnter your question (or 'quit' to exit): ")
#     if question.lower() == 'quit':
#         break

#     try:
#         result = rag.query(question)
#         print(f"\nAnswer: {result['answer']}")
#         print(f"\nSources: {len(result['source_documents'])} documents")
#     except Exception as e:
#         print(f"Error: {e}")