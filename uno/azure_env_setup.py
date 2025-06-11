
import os
from dotenv import load_dotenv

def setup_azure_openai_environment():
    """
    Set up Azure OpenAI environment variables.
    This function loads variables from .env file and sets them up for the pipeline.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # Check if required environment variables are set
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("Please set these variables in your .env file or environment.")
        print("See .env.template for an example.")
        return False

    print("✅ Azure OpenAI environment variables loaded successfully!")
    return True

def get_azure_openai_credentials():
    """
    Return Azure OpenAI credentials as a dictionary.
    """
    return {
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
        "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "chat_deployment": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "embedding_deployment": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    }

# Example usage
if __name__ == "__main__":
    if setup_azure_openai_environment():
        credentials = get_azure_openai_credentials()
        print("Endpoint:", credentials["endpoint"])
        print("Chat deployment:", credentials["chat_deployment"])
        print("Embedding deployment:", credentials["embedding_deployment"])
