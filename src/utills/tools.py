import requests
from langchain_core.tools import tool

from config.settings import settings
from config.logger import logger


class WeatherTools:
    """
    Class for weather-related tools.
    """
    
    def __init__(self):
        """Initialize the WeatherTools."""
        pass
    
    @property
    def get_weather_tool(self):
        """
        Create and return the get_weather tool.
        
        Returns:
            Tool: The get_weather tool
        """
        @tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city. Use this tool for any weather-related queries."""
            try:
                # Geocode the city name to get coordinates
                geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={settings.OPENWEATHER_API_KEY}"
                geo_response = requests.get(geo_url)
                geo_response.raise_for_status()
                geo_data = geo_response.json()
                
                if not geo_data:
                    return f"I couldn't find the location '{city}'. Please try another city."
                
                lat = geo_data[0]["lat"]
                lon = geo_data[0]["lon"]
                
                # Fetch weather data
                weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={settings.OPENWEATHER_API_KEY}&units=metric"
                weather_response = requests.get(weather_url)
                weather_response.raise_for_status()
                data = weather_response.json()
                
                # Extract weather data
                main = data.get("main", {})
                weather = data.get("weather", [{}])[0]
                sys = data.get("sys", {})
                
                # Format weather data
                weather_info = {
                    "temperature_celsius": main.get("temp", 0.0),
                    "feels_like_celsius": main.get("feels_like", 0.0),
                    "humidity": main.get("humidity", 0),
                    "weather_main": weather.get("main", ""),
                    "weather_description": weather.get("description", ""),
                    "city_name": data.get("name", ""),
                    "country_code": sys.get("country", "")
                }
                
                logger.info(f"Weather data retrieved for {city}")
                return f"Weather in {city}: {weather_info['temperature_celsius']}°C, feels like {weather_info['feels_like_celsius']}°C, {weather_info['weather_description']}, humidity: {weather_info['humidity']}%"
            
            except Exception as e:
                logger.error(f"Error fetching weather data for {city}: {str(e)}")
                return f"Error fetching weather data: {str(e)}"
        
        return get_weather


class DocumentTools:
    """
    Class for document-related tools.
    """
    
    def __init__(self, retriever, llm_manager):
        """
        Initialize document tools with a retriever and LLM manager.
        
        Args:
            retriever: Document retriever (can be None)
            llm_manager: LLM manager instance
        """
        self.retriever = retriever
        self.llm_manager = llm_manager
        self.rag_chain = None
        
        # Create RAG chain if retriever is available
        if self.retriever:
            from schema.schema import AnswerResponse
            self.rag_chain = llm_manager.create_rag_chain(retriever)
    
    def update_retriever(self, new_retriever):
        """
        Update the retriever and recreate the RAG chain.
        
        Args:
            new_retriever: New retriever to use
        """
        self.retriever = new_retriever
        if self.retriever:
            self.rag_chain = self.llm_manager.create_rag_chain(self.retriever)
        else:
            self.rag_chain = None
    
    @property
    def query_document_tool(self):
        """
        Create and return the query_document tool.
        
        Returns:
            Tool: The query_document tool
        """
        @tool
        def query_document(query: str) -> str:
            """Query the document to answer a question. Use this tool for any document-related queries."""
            if not self.retriever or not self.rag_chain:
                return "No documents have been loaded in the resume_collection."
            
            try:
                # Use the RAG chain to answer the question
                response = self.rag_chain.invoke(query)
                
                # Format the response to match the expected output format
                if isinstance(response, str) and not response.startswith("answer="):
                    response = f"answer='{response}'"
                
                logger.info(f"Document query processed: {query[:50]}...")
                return response
            except Exception as e:
                logger.error(f"Error querying document: {str(e)}")
                return f"Error retrieving information from the resume collection: {str(e)}"
        
        return query_document