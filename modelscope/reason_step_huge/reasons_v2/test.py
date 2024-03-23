#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
openai.api_type = "azure"
openai.api_base = "https://.openai.azure.com/"
openai.api_version = "2023-09-15-preview"
openai.api_key = ""

def get_embedding_vector(text):
    # Use the text-embedding-ada-002 model to generate an embedding
    response = openai.Embedding.create(
    input=text,
    model="text-embedding-ada-002",
    engine="text-embedding-ada-002",
    )

    # The response will contain the embedding vector
    embedding_vector = response['data']
    # get the length of the embedding vector

    return embedding_vector[0].embedding
