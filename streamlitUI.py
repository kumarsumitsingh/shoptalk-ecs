import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
import boto3
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

S3_BUCKET = "faiss-sumit"  # Your bucket name
S3_PREFIX = "faiss_index/"       # Your S3 prefix containing index files
S3_CLIENT = boto3.client("s3")
S3_IMAGE_BUCKET = "shoptalk-data-bucket"  # Your bucket name


def get_secret():

    secret_name = "shoptalk_dev"
    region_name = "us-east-1"

    client = boto3.client("secretsmanager", region_name=region_name)
    response = client.get_secret_value(SecretId=secret_name)

    if "SecretString" in response:
        secret = json.loads(response["SecretString"])
        return secret["OPENAI_API_KEY"]
    else:
        raise ValueError("Secret not found!")
    
def load_faiss_from_s3(embeddings):
    """Load FAISS index from your specific S3 location"""
    temp_index_path = "/tmp/faiss_temp"
    os.makedirs(temp_index_path, exist_ok=True)
    
    # List and download both index files
    objects = S3_CLIENT.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX
    )
    
    for obj in objects.get("Contents", []):
        s3_key = obj["Key"]
        filename = os.path.basename(s3_key)
        local_path = os.path.join(temp_index_path, filename)
        S3_CLIENT.download_file(S3_BUCKET, s3_key, local_path)
        print(f"Downloaded {s3_key} → {local_path}")

    # Load index using LangChain
    return FAISS.load_local(
        folder_path=temp_index_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True  # Use same embeddings used during index creation
    )

def download_image_from_s3(s3_key):
    """Download image from S3 and return local temp path"""
    try:
        # Create temp directory
        temp_dir = "/tmp/s3_images"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download file
        local_path = os.path.join(temp_dir, os.path.basename(s3_key))
        S3_CLIENT.download_file(S3_IMAGE_BUCKET, s3_key, local_path)
        return local_path
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None


#faiss_idx = "/Users/sumitkumar/capstone/faiss_index"
openai_api_key = os.getenv("OPENAI_API_KEY")

# System message (instructions for the AI)
system_template = SystemMessagePromptTemplate.from_template(
    "You are a shopping assistant. Use the product data to answer the query. Focus on prices, descriptions, and product names."
)

# Human message (user query and context)
human_template = HumanMessagePromptTemplate.from_template(
    "Retrieved Product Data:\n{context}\n\nQuery: {query}\nAnswer:"
)

# Combine into a ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([system_template, human_template])
#prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model="gpt-3.5-turbo"  # or another model
)
#openai_api_key=get_secret()
if not openai_api_key or not openai_api_key.startswith("sk-"):
    raise ValueError(f"Invalid API Key retrieved: {repr(openai_api_key)}")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# Load FAISS Vector Store
faiss_store=load_faiss_from_s3(embeddings)

# Query Interface
st.title("Shop Bot - Beta")
query = st.text_input("Enter your query:")

if query:
    results = faiss_store.similarity_search(query, k=5)
    print(results[0].metadata)
    # Extract context from metadata
    context = "\n".join([
        f"Price: {res.metadata.get('price', 'N/A')}, Description: {res.metadata.get('text', '').replace('{', '{{').replace('}', '}}')}"
        for res in results
    ])
    
    # Generate response using .invoke()
    response = llm.invoke(prompt.format(context=context, query=query))
    st.write("### Assistant Response")
    st.write(response.content)
    
    st.subheader(f"Results")
    # Display results
    if results:
        for i, result in enumerate(results):
            metadata = result.metadata
            #st.write(metadata)
            # Display metadata information 
            
            #st.write(metadata.get("text", "No description available"))

            # Display text product_description
            description = metadata.get("text", "No description available")
            #description = metadata.get("text", "No description available")
            st.write(f"**Description**: {description}")

            # Show image if the path exists
            image_path = metadata.get("path")
            #image_path=image_path.lstrip("small/")

            if image_path:
                # Resolve full path
                #full_image_path = os.path.join("/Users/sumitkumar/capstone/", image_path)
                full_image_path = download_image_from_s3(image_path)
                
                if full_image_path and os.path.exists(full_image_path):
                    st.image(full_image_path, caption=metadata.get("item_name", "No Item Name"), use_container_width=True)
                else:
                    st.warning("Image file not found."+image_path)
            else:
                st.write("No image available for this result.")
            
            # Display the price if available
            price = metadata.get("price")
            if price:
                st.write(f"**Price:** {price}")

            # Add spacing between items
            st.markdown("---")
    else:
        st.write("No results found for your query.")
