from config.logger import logger
from utills.node import AgentNode
from utills.tools import DocumentTools, WeatherTools
from utills.embeddings import EmbeddingsManager
from utills.llm import LLMManager


class AgentWorkflow:
    """
    Class for setting up and managing the agent workflow.
    Implements a singleton pattern to avoid reinitializing components.
    """
    _instance = None
    
    def __new__(cls, collection_name=None):
        """
        Create a new instance if one doesn't exist, otherwise return the existing one.
        
        Args:
            collection_name: Optional name for the vector store collection
        """
        if cls._instance is None:
            cls._instance = super(AgentWorkflow, cls).__new__(cls)
            cls._instance.is_initialized = False
            cls._instance.collection_name = collection_name or "resume_collection"
            logger.info(f"Creating new AgentWorkflow instance with collection: {cls._instance.collection_name}")
        return cls._instance
    
    def __init__(self, collection_name=None):
        """
        Initialize the agent workflow.
        
        Args:
            collection_name: Optional name for the vector store collection
        """
        # Only update collection_name if provided and different
        if collection_name and self.collection_name != collection_name:
            logger.info(f"Updating collection name from {self.collection_name} to {collection_name}")
            self.collection_name = collection_name
            self.is_initialized = False  # Force reinitialization with new collection
            
        if not hasattr(self, 'is_initialized'):
            self.is_initialized = False
            
        # These will be initialized in setup()
        if not hasattr(self, 'embeddings_manager'):
            self.embeddings_manager = None
            self.llm_manager = None
            self.weather_tools = None
            self.document_tools = None
            self.agent_node = None
            self.retriever = None
        
    def setup(self):
        """
        Set up the agent workflow if not already initialized.
        
        Returns:
            AgentWorkflow: Self for method chaining
        """
        if self.is_initialized:
            logger.info("AgentWorkflow already initialized, skipping setup")
            return self
            
        try:
            logger.info("Setting up AgentWorkflow components")
            
            # Initialize LLM manager
            self.llm_manager = LLMManager()
            
            # Initialize embeddings manager
            self.embeddings_manager = EmbeddingsManager(self.collection_name)
            
            # Get retriever from existing collection
            self.retriever = self.embeddings_manager.get_retriever()
            
            # Initialize tools
            self.weather_tools = WeatherTools()
            self.document_tools = DocumentTools(self.retriever, self.llm_manager)
            
            # Initialize agent node
            self.agent_node = AgentNode(self.llm_manager)
            
            # Create specialized agents
            self.weather_agent = self.agent_node.create_weather_agent(self.weather_tools.get_weather_tool)
            self.rag_agent = self.agent_node.create_rag_agent(self.document_tools.query_document_tool)
            
            # Create supervisor agent
            self.agent_node.create_supervisor_agent()
            
            self.is_initialized = True
            logger.info("Agent workflow setup complete")
            return self
        except Exception as e:
            logger.error(f"Error setting up agent workflow: {str(e)}")
            raise
    
    def process_query(self, query):
        """
        Process a user query through the agent workflow.
        
        Args:
            query: User query string
            
        Returns:
            dict: Result from the supervisor
        """
        if not self.is_initialized or not self.agent_node:
            logger.warning("Agent workflow not initialized, setting up now")
            self.setup()
        
        return self.agent_node.process_query(query)
    
    def extract_conversation(self, result):
        """
        Extract the complete conversation from the result including all routing and tool calls.
        
        Args:
            result: Result from the supervisor
            
        Returns:
            list: List of messages in the conversation
        """
        messages = []
        
        # Process all messages in the conversation
        for message in result["messages"]:
            if hasattr(message, "content") and message.content:
                # Determine the role based on message type
                if message.type == "human":
                    role = "human"
                elif message.type == "ai":
                    role = "ai"
                elif message.type == "tool":
                    role = "tool"
                else:
                    role = "system"
                
                content = message.content
                
                # Clean up tool responses if needed
                if role == "tool" and content.startswith("answer="):
                    # Keep the format as is for exact matching with expected output
                    pass
                
                # Add the message to the list
                messages.append({
                    "role": role,
                    "content": content
                })
        
        # If no messages were found, add a fallback
        if not messages:
            messages.append({
                "role": "ai",
                "content": "I'm sorry, there was an issue processing your query. Please try again."
            })
        
        return messages