import pandas as pd
import boto3
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import json

# AWS S3 Setup
#S3_BUCKET = os.getenv("S3_BUCKET", "faiss-sumit")
S3_BUCKET="faiss-sumit"
S3_CLIENT = boto3.client("s3")

def get_secret():
    """Retrieve OpenAI API key from environment variable or Secrets Manager"""
    # First check environment variables
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    
    # Fallback to Secrets Manager
    secret_name = "shoptalk_dev"
    region_name = "us-east-2"

    try:
        client = boto3.client("secretsmanager", region_name=region_name)
        response = client.get_secret_value(SecretId=secret_name)
        
        if "SecretString" in response:
            secret = json.loads(response["SecretString"])
            return secret.get("OPENAI_API_KEY")
        else:
            raise ValueError("No SecretString found in secret response")
            
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve secret: {str(e)}") from e

# Load Dataset from S3
def load_csv_from_s3(file_key):
    temp_file = "/tmp/temp_data.csv"
    try:
        print("downloading..")
        S3_CLIENT.download_file(S3_BUCKET, file_key, temp_file)
        #print(temp_file)
        return pd.read_csv(temp_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV from S3: {str(e)}") from e

# Save FAISS index to S3
def save_faiss_to_s3(faiss_store, s3_key):
    temp_index_path = "/tmp/faiss_index"
    try:
        faiss_store.save_local(temp_index_path)
        for file in os.listdir(temp_index_path):
            local_path = f"{temp_index_path}/{file}"
            if os.path.exists(local_path):
                S3_CLIENT.upload_file(local_path, S3_BUCKET, f"{s3_key}/{file}")
            else:
                raise FileNotFoundError(f"Index file {local_path} not found")
    except Exception as e:
        raise RuntimeError(f"Failed to save FAISS index: {str(e)}") from e

# Embedding Processing
def embed_batch(batch_texts, embeddings_model):
    return embeddings_model.embed_documents(batch_texts)

def parallelize_embeddings(texts, embeddings_model, batch_size=50, max_workers=5):
    embeddings = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        results = list(executor.map(
            lambda batch: embed_batch(batch, embeddings_model), 
            batches
        ))
        embeddings = [emb for batch in results for emb in batch]
    return embeddings
#def handler(event, context):
def handler(event, context):
    try:
        # Load OpenAI API Key
        #openai_api_key = get_secret()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set!")

        # Load data from S3
        print('loading...')
        data = load_csv_from_s3("cleaneddf.csv")
        #print(data)
        if data is None:
            print("Data is empty")
        #print("data is "+data)
        
        # Data preprocessing
        if "country" in data.columns:
            data = data[~data["country"].isin(["JP", "IT", "FR"])]
            #print(data)
        
        if "path" in data.columns:
            data["path"] = data["path"].apply(lambda x: f"small/{x}" if pd.notnull(x) else x)
        
        data["price"] = data["price"].fillna(0).apply(lambda x: f"${x:.2f}")
        data["description"] = data[
            ["item_name", "product_type", "product_description", "color", "price"]
        ].fillna("").agg(" ".join, axis=1)

        metadata = [{"path": row["path"], "item_name": row["item_name"], "price": row["price"], "text": row["description"]} 
                   for _, row in data.iterrows()]
        #print(type(metadata))
        # Initialize OpenAI Embeddings
        openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Generate Embeddings
        texts = data["description"].tolist()
        print(texts)
        embeddings = parallelize_embeddings(texts, openai_embeddings)

        # Create FAISS index
        faiss_store = FAISS.from_texts(texts, openai_embeddings, metadatas=metadata)

        # Save FAISS index to S3
        save_faiss_to_s3(faiss_store, "faiss_index")

        return {"status": "Embeddings created and stored successfully!"}
    
    except Exception as e:
        return {"error": str(e), "status": "Failed to create embeddings"}

handler(None,None)