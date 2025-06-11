# Initialize
rag = RAGPipeline(azure_endpoint, api_key)

# Load PDFs
rag.load_and_process_pdfs(["your_file.pdf"])

# Query
result = rag.query("What is the main topic?")
print(result["answer"])