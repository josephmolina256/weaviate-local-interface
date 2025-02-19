import weaviate
import weaviate.classes as wvc
from sentence_transformers import SentenceTransformer

from typing import List, Dict, Optional
import traceback

from .constants import HF_MODEL_NAME, K_RETRIEVALS, CERTAINTY_THRESHOLD


class WeaviateInterface:
    def __init__(self, generate_embeddings: bool = True, hf_model_name: str = HF_MODEL_NAME):
        """Constructor for the WeaviateInterface class.

        Args:
            generate_embeddings (bool, optional): Decide whether you want us to generate embeddings for you. Otherwise, provide your own. Defaults to True.
            hf_model_name (Optional[str], optional): If generate_embeddings=True, set a sentence-tranformers model to use. Defaults to HF_MODEL_NAME ("multi-qa-distilbert-cos-v1").
        """
        self.generate_embeddings = generate_embeddings
        if generate_embeddings:
            self.embedding_model = SentenceTransformer(hf_model_name)
        self.client = weaviate.connect_to_local()
        assert self.client.is_live()

    def store(self, input_data: List[Dict],  collections_name: str, key_to_be_embedded: Optional[str] = None, embeddings: Optional[List[List[float]]] = None) -> Dict[str, str]:
        """General storing function into Weaviate.

        Args:
            input_data (List[dict]): Takes in a list of dictionaries to store in Weaviate. 
                                    Each entire dictionary is stored as a DataObject.
            
            collections_name (str): The name of the collection to store the data in. 
                                    If the collection does not exist, a new collection is created.

            key_to_be_embedded (str): Use if generate_embeddings=True. The key in the dictionary whose value is to be embedded for retrieval purposes. 
                                    Must be present in the highest level of every dictionary.

            embeddings (List[List[float]]): Use if generate_embeddings=False. Takes in a list of embeddings to store in Weaviate. 
                                      NOTE: The length of this list must be equal to the length of input_data 
                                      AND the data type is native floats not np.float64. See alternative store() for example conversions.
  
        """
        if self.generate_embeddings:
            assert key_to_be_embedded
        else:
            assert embeddings
            assert len(input_data) == len(embeddings)

        assert self.client.is_live()

        try:            
            wv_objs = list()

            for i in range(len(input_data)):
                properties = input_data[i]

                if self.generate_embeddings:
                    embedding = self.embedding_model.encode(object[key_to_be_embedded]).tolist()
                else:
                    embedding = embeddings[i]

                wv_objs.append(wvc.data.DataObject(
                    properties=properties,
                    vector=embedding
                ))

            collection = self.client.collections.get(collections_name)
            if collection is None:
                collection = self.client.collections.create(
                    collections_name,
                    vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                )

            collection.data.insert_many(wv_objs)
            print("Stored successfully!")
            return {"status": f"Stored {len(wv_objs)} items successfully!"}
        except Exception as e:
            print("An error occurred in Storer.store:")
            traceback.print_exc()  # Prints full traceback
            self.client.close()
            return {"status": f"an error occurred {e}"}

    
    def retrieve(
            self, 
            query: str, 
            collections_name: str, 
            query_embedding: Optional[List[float]] = None, 
            limit: int = K_RETRIEVALS, 
            certainty_threshold: float=CERTAINTY_THRESHOLD
        ) -> Optional[List[Dict]]:
        """General retrieval function from Weaviate.

        Args:
            query (str): The query string to search for in Weaviate.

            collections_name (str): The name of the collection to search in. COLLECTION MUST EXIST.
            
            query_embedding (Optional[List[float]], optional): Optional provided vector for retrieval. Defaults to None.

            limit (int, optional): Max number of objects retrieved. Defaults to K_RETRIEVALS (3).

            certainty_threshold (float, optional): minimum threshold of certainty to retreive. Defaults to CERTAINTY_THRESHOLD (0.5).

        Returns:
            Optional[List[Dict]]: List of dictionaries containing the properties of the retrieved objects and their associated confidence scores.
        """

        assert self.generate_embeddings or query_embedding
        assert self.client.is_live()
        
        try:
            collection = self.client.collections.get(collections_name)
            if self.generate_embeddings:
                query_embedding = self.embedding_model.encode(query).tolist()
            else:
                assert type(query_embedding) == list # redundant check for type hinting in line 112 onward

            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(certainty=True),
                certainty=certainty_threshold,
            )

            res = []
            for object in response.objects:
                res.append(
                    {
                        "certainty": object.metadata.certainty,
                        "properties": object.properties
                    }
                )

            if len(res) == 0:
                print("No results found.")
                return None
            return res
        except Exception as e:
            print("An error occurred in Storer.retrieve: {e}")
            traceback.print_exc()  # Prints full traceback
            self.client.close()
