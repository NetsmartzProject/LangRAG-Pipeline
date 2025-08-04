from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
import qdrant_client
from qdrant_client.models import Distance, VectorParams

from config.settings import settings
from config.logger import logger


class EmbeddingsManager:
    """
    Class for managing embeddings and vector stores.
    Implements a singleton pattern to avoid reloading the model.
    """
    _instance = None
    _embeddings_model = None
    _client = None
    
    def __new__(cls, collection_name=None):
        """
        Create a new instance if one doesn't exist, otherwise return the existing one.
        
        Args:
            collection_name: Optional name for the vector store collection
        """
        if cls._instance is None:
            cls._instance = super(EmbeddingsManager, cls).__new__(cls)
            cls._instance.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
            logger.info(f"Creating new EmbeddingsManager instance with collection: {cls._instance.collection_name}")
        return cls._instance
    
    def __init__(self, collection_name=None):
        """
        Initialize the EmbeddingsManager.
        
        Args:
            collection_name: Optional name for the vector store collection
        """
        # Only set collection_name if it's provided and different from current
        if collection_name and self.collection_name != collection_name:
            logger.info(f"Updating collection name from {self.collection_name} to {collection_name}")
            self.collection_name = collection_name
    
    def get_embeddings_model(self):
        """
        Initialize and return the embeddings model based on settings.
        Uses class variable to cache the model across instances.
        
        Returns:
            HuggingFaceEmbeddings: Initialized embeddings model
        """
        if EmbeddingsManager._embeddings_model is None:
            try:
                logger.info("Initializing embeddings model (first time)")
                model_kwargs = {'device': settings.EMBEDDING_MODEL_DEVICE}
                encode_kwargs = {'normalize_embeddings': settings.EMBEDDING_NORMALIZE}
                
                EmbeddingsManager._embeddings_model = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                logger.info(f"Embeddings model initialized: {settings.EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Error initializing embeddings model: {str(e)}")
                raise
        else:
            logger.info("Using cached embeddings model")
        
        return EmbeddingsManager._embeddings_model
    
    def get_qdrant_client(self):
        """
        Get or create a Qdrant client.
        Uses class variable to cache the client across instances.
        
        Returns:
            QdrantClient: Initialized Qdrant client
        """
        if EmbeddingsManager._client is None:
            logger.info("Initializing Qdrant client (first time)")
            EmbeddingsManager._client = qdrant_client.QdrantClient(
                url=settings.QDRANT_URL,
                prefer_grpc=settings.QDRANT_PREFER_GRPC
            )
        else:
            logger.info("Using cached Qdrant client")
            
        return EmbeddingsManager._client
    
    def check_collection_exists(self):
        """
        Check if the collection exists in Qdrant.
        
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            client = self.get_qdrant_client()
            
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            exists = self.collection_name in collection_names
            logger.info(f"Collection {self.collection_name} exists: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking collection existence: {str(e)}")
            return False
    
    def get_retriever(self):
        """
        Get a retriever from an existing vector store.
        
        Returns:
            Retriever: Document retriever
        """
        try:
            # Check if collection exists
            if not self.check_collection_exists():
                logger.warning(f"Collection {self.collection_name} does not exist")
                return None
            
            # Initialize connection to existing vector store
            embeddings = self.get_embeddings_model()
            client = self.get_qdrant_client()
            
            vector_store = Qdrant(
                client=client,
                collection_name=self.collection_name,
                embeddings=embeddings
            )
        
            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": settings.RETRIEVAL_K}
            )
            
            logger.info(f"Retriever initialized with k={settings.RETRIEVAL_K}")
            return retriever
        except Exception as e:
            logger.error(f"Error getting retriever: {str(e)}")
            raise