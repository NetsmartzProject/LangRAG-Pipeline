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
    """
    
    def __init__(self, collection_name=None):
        """
        Initialize the EmbeddingsManager.
        
        Args:
            collection_name: Optional name for the vector store collection
        """
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.embeddings_model = None
        self.client = None
    
    def get_embeddings_model(self):
        """
        Initialize and return the embeddings model based on settings.
        
        Returns:
            HuggingFaceEmbeddings: Initialized embeddings model
        """
        if self.embeddings_model is None:
            try:
                model_kwargs = {'device': settings.EMBEDDING_MODEL_DEVICE}
                encode_kwargs = {'normalize_embeddings': settings.EMBEDDING_NORMALIZE}
                
                self.embeddings_model = HuggingFaceEmbeddings(
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                logger.info(f"Embeddings model initialized: {settings.EMBEDDING_MODEL_NAME}")
            except Exception as e:
                logger.error(f"Error initializing embeddings model: {str(e)}")
                raise
        
        return self.embeddings_model
    
    def get_qdrant_client(self):
        """
        Get or create a Qdrant client.
        
        Returns:
            QdrantClient: Initialized Qdrant client
        """
        if self.client is None:
            self.client = qdrant_client.QdrantClient(
                url=settings.QDRANT_URL,
                prefer_grpc=settings.QDRANT_PREFER_GRPC
            )
        return self.client
    
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