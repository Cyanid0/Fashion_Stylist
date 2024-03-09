from flask import Flask, jsonify

from langchain.vectorstores import Qdrant
from langchain.retrievers import EnsembleRetriever
from langchain.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient
import pandas as pd

app = Flask(__name__)

embeddings = SentenceTransformerEmbeddings(model_name='sentence-transformers/clip-ViT-B-32')

def get_results(search_results):
    filtered_img_ids = [doc.metadata.get("image_id") for doc in search_results]
    return filtered_img_ids

client = QdrantClient(
    url="https://763bc1da-0673-4535-91ac-b5538ec0287f.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key='UOqiBgqhhu8BBWP98mwjGl7h4IhL2vMAqzO4EI9PEB66A50n9GoIiQ',
) # Persists changes to disk, fast prototyping

COLLECTION_NAME="semantic_image_search_cleaned"

dense_vector_retriever = Qdrant(client, COLLECTION_NAME, embeddings)
images_data = pd.read_csv("/content/images.csv", on_bad_lines='skip')

def get_link(query):
    Search_Query = query
    neutral_retiever = dense_vector_retriever.as_retriever()
    result = neutral_retiever.get_relevant_documents(Search_Query)
    filtered_images = get_results(result)
    filtered_img_ids = [doc.metadata.get("image_id") for doc in result]

    links = [images_data.loc[id, 'link'] for id in filtered_img_ids]
#     final = '[' + ','.join(links) + ']'
    return filtered_img_ids

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        result = get_link(query)
        return jsonify(result)
    else:
        return jsonify({"error": "No query provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)


