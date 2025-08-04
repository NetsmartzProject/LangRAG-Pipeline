from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage, AIMessage

from config.logger import logger


class AgentNode:
    """
    Class for creating and managing agent nodes.
    Implements a singleton pattern to avoid recreating agents.
    """
    _instance = None
    
    def __new__(cls, llm_manager=None):
        """
        Create a new instance if one doesn't exist, otherwise return the existing one.
        
        Args:
            llm_manager: LLM manager instance
        """
        if cls._instance is None:
            cls._instance = super(AgentNode, cls).__new__(cls)
            cls._instance.is_initialized = False
            logger.info("Creating new AgentNode instance")
        return cls._instance
    
    def __init__(self, llm_manager=None):
        """
        Initialize the AgentNode.
        
        Args:
            llm_manager: LLM manager instance
        """
        if not hasattr(self, 'is_initialized') or not self.is_initialized:
            self.llm_manager = llm_manager
            self.llm = llm_manager.get_llm() if llm_manager else None
            self.weather_agent = None
            self.rag_agent = None
            self.supervisor = None
            self.is_initialized = True
    
    def create_weather_agent(self, get_weather_tool):
        """
        Create the weather agent.
        
        Args:
            get_weather_tool: Tool for getting weather data
            
        Returns:
            Agent: The weather agent
        """
        if not self.weather_agent:
            logger.info("Creating weather agent")
            self.weather_agent = create_react_agent(
                model=self.llm,
                tools=[get_weather_tool],
                name="weather_agent",
                system_message=(
                    "You are a specialized weather assistant that provides accurate weather information.\n"
                    "Use the get_weather tool to fetch current weather data for any location.\n"
                    "Be helpful, concise, and focus only on weather-related queries.\n"
                    "ALWAYS use the get_weather tool for EVERY query to get accurate, real-time weather data.\n"
                    "If the query is not about weather, politely explain that you can only help with weather information."
                )
            )
            logger.info("Weather agent created")
        else:
            logger.info("Using existing weather agent")
        return self.weather_agent
    
    def create_rag_agent(self, query_document_tool):
        """
        Create the RAG agent.
        
        Args:
            query_document_tool: Tool for querying documents
            
        Returns:
            Agent: The RAG agent
        """
        if not self.rag_agent:
            logger.info("Creating RAG agent")
            self.rag_agent = create_react_agent(
                model=self.llm,
                tools=[query_document_tool],
                name="rag_agent",
                system_message=(
                    "You are a specialized document assistant that answers questions based ONLY on the content in the provided documents.\n"
                    "Use the query_document tool to find information in the documents. ALWAYS use this tool for EVERY question, no exceptions.\n"
                    "IMPORTANT: Always extract and present the answer from the tool's response. The tool response will be in the format answer='...'. Extract just the content inside the quotes.\n"
                    "Only provide information that is explicitly mentioned in the documents. DO NOT use your general knowledge or make assumptions.\n"
                    "Be helpful, concise, and focus only on document-related queries.\n"
                    "If no documents are available, inform the user they need to upload a document first.\n"
                    "NEVER respond with generic messages like 'conversation completed' or 'transferring back'. Always provide a substantive answer based on the document content.\n"
                    "NEVER respond with 'I don't have information' without first attempting to use the query_document tool."
                ),
                max_iterations=3  # Give it more chances to use the tool
            )
            logger.info("RAG agent created")
        else:
            logger.info("Using existing RAG agent")
        return self.rag_agent
    
    def create_supervisor_agent(self):
        """
        Create the supervisor agent.
        
        Returns:
            Agent: The supervisor agent
        """
        if not self.supervisor:
            # Ensure the specialized agents are created
            if not self.weather_agent or not self.rag_agent:
                raise ValueError("Weather and RAG agents must be created before supervisor")
            
            logger.info("Creating supervisor agent")
            supervisor_prompt = """
            You are a supervisor managing two specialized agents:

            1. weather_agent: Use ONLY for weather-related queries.
               - Examples: "What's the weather in London?", "Is it raining in Tokyo?", "Temperature in New York"
               - Keywords: weather, temperature, forecast, humidity, rain, sunny, cloudy, climate, hot, cold
               - ONLY route queries that are EXPLICITLY about current weather conditions to this agent.

            2. rag_agent: Use for ALL OTHER queries.
               - Route ANY query that is not explicitly about current weather conditions to this agent.
               - This includes ALL questions about people, skills, experience, education, technologies, etc.
               - This agent has access to document data and can answer questions about the content.

            IMPORTANT RULES:
            - If the query contains words like "weather", "temperature", "forecast", "rain", "sunny", "cloudy" AND asks about current conditions, route to weather_agent.
            - For ALL OTHER queries, route to rag_agent. This is your DEFAULT choice.
            - You MUST route EVERY query to ONE of these agents - never handle queries yourself.
            - NEVER respond with your own answers - your job is ONLY to route queries.
            - Do NOT add any additional messages like "Please enter your next query" or "Conversation completed".

            Remember: You are ONLY a router. You do not answer questions directly.
            """
            
            self.supervisor = create_supervisor(
                model=self.llm,
                agents=[self.weather_agent, self.rag_agent],
                prompt=supervisor_prompt,
                add_handoff_back_messages=True,
                output_mode="full_history"
            ).compile()
            
            logger.info("Supervisor agent created and workflow compiled")
        else:
            logger.info("Using existing supervisor agent")
        
        return self.supervisor
    
    def process_query(self, query: str):
        """
        Process a user query through the agent pipeline.
        
        Args:
            query: User query string
            
        Returns:
            dict: Result from the supervisor
        """
        if not self.supervisor:
            raise ValueError("Supervisor agent not created")
        
        # Create input state
        input_state = {"messages": [HumanMessage(content=query)]}
        
        # Process through the supervisor
        try:
            result = self.supervisor.invoke(input_state)
            logger.info(f"Query processed successfully: {query[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise