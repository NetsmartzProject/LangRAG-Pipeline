from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.settings import settings
from config.logger import logger


class LLMManager:
    """
    Class for managing LLM models and chains.
    """
    def __init__(self):
        """Initialize the LLMManager."""
        self._llm = None
    
    def get_llm(self):
        """
        Initialize and return the LLM model based on settings.
        
        Returns:
            ChatGroq or ChatOpenAI: Initialized LLM model
        """
        if self._llm is None:
            try:
                # Use environment variables for API keys
                # Uncomment the model you want to use
                
                # Option 1: Use Groq
                self._llm = ChatGroq(
                    model=settings.LLM_MODEL,
                    api_key=settings.GROQ_API_KEY,
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS
                )
                
                # Option 2: Use OpenAI
                # self._llm = ChatOpenAI(
                #     model="gpt-4o",
                #     api_key=settings.OPENAI_API_KEY,
                #     temperature=settings.LLM_TEMPERATURE,
                #     max_tokens=settings.LLM_MAX_TOKENS,
                #     max_retries=2,
                #     request_timeout=30,
                # )
                
                logger.info(f"LLM initialized with model: {settings.LLM_MODEL}")
            except Exception as e:
                logger.error(f"Error initializing LLM: {str(e)}")
                raise
        
        return self._llm
    
    def create_rag_prompt(self):
        """
        Create a RAG prompt template.
        
        Returns:
            ChatPromptTemplate: The RAG prompt template
        """
        rag_prompt_template = """
        You are an intelligent assistant tasked with answering user queries based on provided context. 
        Use the following context to respond to the user's question.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        return ChatPromptTemplate.from_template(rag_prompt_template)
    
    def create_rag_chain(self, retriever):
        """
        Create a RAG chain with structured output.
        
        Args:
            retriever: Document retriever
            
        Returns:
            Chain: The RAG chain
        """
        llm = self.get_llm()
        rag_prompt = self.create_rag_prompt()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Create a chain that properly formats the documents and passes them to the prompt
        rag_chain = (
            {
                "context": retriever | format_docs,
                "query": lambda x: x
            }
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        
        # Wrap the response in the expected format
        def wrap_response(response):
            return f"answer='{response}'"
        
        return rag_chain | wrap_response