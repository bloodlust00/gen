import gdown
//pip install langchain cohere langchain-community google-colab

# Replace this with your actual Google Drive file ID (from a shareable link)
file_id = "10Tc7hsxqEA6o8mRZrGrbphNJ5Ep1pdsj"  # <-- Change this to your file ID
url = f"https://docs.google.com/document/d/10Tc7hsxqEA6o8mRZrGrbphNJ5Ep1pdsj/edit?usp=sharing&ouid=110258667970037103275&rtpof=true&sd=true"
output_file = "document.txt"
gdown.download(url, output_file, quiet=False)

# Step 3: Read the Document
with open(output_file, 'r') as file:
    document_text = file.read()

import os
from langchain.llms import Cohere
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Replace this with your actual Cohere API Key from https://dashboard.cohere.com
os.environ['COHERE_API_KEY'] = 'HZnJtIYxY0Nnwye8tU43nFPA76QqqlhH2qCU3nxL'  # <-- Replace this!
llm = Cohere(cohere_api_key=os.environ['COHERE_API_KEY'])

# Step 5: Create a Prompt Template
template = PromptTemplate(
    input_variables=["content"],
    template="Given the following content:\n\n{content}\n\nSummarize the key points in bullet format:"
)

# Step 6: Split the Document into Chunks
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
chunks = text_splitter.split_text(document_text)

# Step 7: Generate Summaries for Each Chunk
summaries = []


for i, chunk in enumerate(chunks):
    try:
        prompt = template.format(content=chunk)
        response = llm.invoke(prompt)
        summaries.append(f"### Summary {i+1} ###\n{response}\n")
    except Exception as e:
        print(f"Error processing chunk {i+1}: {e}")

# Step 8: Display Full Summary
full_summary = "\n".join(summaries)
print("\n==== FINAL SUMMARY ====\n")
print(full_summary)
