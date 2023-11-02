from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import pandas as pd
import torch 
from datasets import load_dataset
from sentence_transformers.util import semantic_search

#scaping the docker documentation
url = "https://docs.docker.com/engine/reference/builder/"
page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text()
text = text.split("\n")

#using the huggingface pipeline to get the embeddings
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

output = query(text)

embeddings = pd.DataFrame(output)

#saving the embeddings
embeddings.to_csv("embeddings.csv", index=False)

#loading the embeddings
faqs_embeddings = load_dataset('csv', data_files='embeddings.csv')
dataset_embeddings = torch.from_numpy(faqs_embeddings["train"].to_pandas().to_numpy()).to(torch.float)

#asking the question
question = ["DockerFile must begin with?"]
output = query(question)
query_embeddings = torch.FloatTensor(output)
print(f"The size of our embedded dataset is {dataset_embeddings.shape} and of our embedded query is {query_embeddings.shape}.")


#Sementic Search
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=5)

print([text[hits[0][i]['corpus_id']] for i in range(len(hits[0]))])