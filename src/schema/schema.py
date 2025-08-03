from pydantic import BaseModel, Field
from typing import List


class WeatherResponse(BaseModel):
    """Structured weather data response"""
    temperature_celsius: float = Field(description="Current temperature in Celsius", examples=[22.5], ge=-100, le=100)
    feels_like_celsius: float = Field(description="Feels like temperature in Celsius", examples=[23.1], ge=-100, le=100)
    humidity: int = Field(description="Humidity percentage", examples=[65], ge=0, le=100)
    weather_main: str = Field(description="Main weather condition", examples=["Clear", "Clouds", "Rain"], min_length=1)
    weather_description: str = Field(description="Detailed weather description", examples=["clear sky", "scattered clouds"], min_length=1)
    city_name: str = Field(description="City name", examples=["London", "New York"], min_length=1)
    country_code: str = Field(description="Country code", examples=["GB", "US"], min_length=1)


class AnswerResponse(BaseModel):
    """Structured response for document queries"""
    answer: str = Field(description="The answer to the user's question based on the retrieved context")
    

class RoutingDecision(BaseModel):
    """Decision about which agent to route to"""
    agent: str = Field(
        description="The agent to route to", 
        examples=["weather_agent", "rag_agent"]
    )
    reasoning: str = Field(
        description="Reasoning behind the routing decision",
        min_length=10
    )


class Message(BaseModel):
    """Message in a conversation"""
    role: str = Field(description="Role of the message sender (user, assistant, system)")
    content: str = Field(description="Content of the message")


class ConversationResponse(BaseModel):
    """Response for a conversation query"""
    messages: List[Message] = Field(description="The conversation messages")
    processing_time: float = Field(description="Processing time in seconds")