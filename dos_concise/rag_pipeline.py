import os
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextFileLoader
from langchain.prompts import PromptTemplate

class RAGPipeline:
    def __init__(self, azure_endpoint, api_key, api_version="2024-02-01"):
        """
        Initialize RAG pipeline with Azure OpenAI credentials
        """
        # Set up Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment="text-embedding-ada-002"  # Replace with your embedding deployment name
        )
        
        # Set up Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            azure_deployment="gpt-35-turbo",  # Replace with your chat deployment name
            temperature=0.7
        )
        
        self.vectorstore = None
        self.qa_chain = None
    
    def load_and_process_documents(self, file_paths, chunk_size=1000, chunk_overlap=200):
        """
        Load documents, split into chunks, and create vector store
        """
        # Load documents
        documents = []
        for file_path in file_paths:
            loader = TextFileLoader(file_path)
            documents.extend(loader.load())
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        # Create FAISS vector store
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Create retrieval QA chain
        self._create_qa_chain()
        
        return len(chunks)
    
    def _create_qa_chain(self):
        """
        Create the retrieval QA chain with custom prompt
        """
        prompt_template = """
        Use the following context to answer the question. If you don't know the answer based on the context, say so.
        
        Context: {context}
        
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
        """
        Query the RAG pipeline
        """
        if not self.qa_chain:
            raise ValueError("Pipeline not initialized. Load documents first.")
        
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
    # Set your Azure OpenAI credentials
    AZURE_ENDPOINT = "your-azure-endpoint"
    API_KEY = "your-api-key"
    
    # Initialize pipeline
    rag = RAGPipeline(AZURE_ENDPOINT, API_KEY)
    
    # Load and process documents
    file_paths = ["document1.txt", "document2.txt"]  # Your document files
    num_chunks = rag.load_and_process_documents(file_paths)
    print(f"Processed {num_chunks} chunks")
    
    # Query the pipeline
    question = "What is the main topic discussed in the documents?"
    result = rag.query(question)
    
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Number of source documents: {len(result['source_documents'])}")
    
    # Optionally save the vector store for later use
    # rag.save_vectorstore("./vectorstore")