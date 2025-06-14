import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class RAGPipelinedos:
    def __init__(self, 
                 embeddings_endpoint, embeddings_api_key,
                 llm_endpoint, llm_api_key, 
                 api_version="2023-12-01-preview"):
        """Initialize the RAG pipeline with separate Azure OpenAI credentials"""
        # Set up Azure OpenAI embeddings (original resource)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=embeddings_endpoint,
            api_key=embeddings_api_key,
            api_version=api_version,
            azure_deployment="text-embedding-ada-002"  # Update with your deployment name
        )
        
        # Set up Azure OpenAI LLM (AI Foundry resource)
        self.llm = AzureOpenAI(
            azure_endpoint=llm_endpoint,
            api_key=llm_api_key,
            api_version=api_version,
            deployment_name="gpt-4o-mini",  # Update with your deployment name
            temperature=0.7
        )
        
        self.vectorstore = None
        self.qa_chain = None
    
    def load_and_process_pdfs(self, pdf_paths):
        """Load PDFs and create vector store"""
        documents = []
        
        # Load PDFs
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Create QA chain
        self._create_qa_chain()
        
        return len(chunks)
    
    def _create_qa_chain(self):
        """Create the QA chain with custom prompt"""
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def query(self, question):
        """Query the RAG pipeline"""
        if not self.qa_chain:
            raise ValueError("Pipeline not initialized. Load PDFs first.")
        
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
    
    def save_vectorstore(self, path):
        """Save the vector store to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path):
        """Load vector store from disk"""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        self._create_qa_chain()

# Usage example
if __name__ == "__main__":
    # Initialize the pipeline with separate endpoints
    rag = RAGPipelinedos(
        embeddings_endpoint="https://your-original-openai-resource.openai.azure.com/",
        embeddings_api_key="your-original-api-key",
        llm_endpoint="https://your-ai-foundry-resource.openai.azure.com/",
        llm_api_key="your-ai-foundry-api-key"
    )
    
    # Load and process PDFs
    pdf_files = ["document1.pdf", "document2.pdf"]
    num_chunks = rag.load_and_process_pdfs(pdf_files)
    print(f"Processed {num_chunks} chunks")
    
    # Save vector store (optional)
    rag.save_vectorstore("./vectorstore")
    
    # Query the system
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
        
        try:
            result = rag.query(question)
            print(f"\nAnswer: {result['answer']}")
            print(f"\nSources: {len(result['source_documents'])} documents")
        except Exception as e:
            print(f"Error: {e}")