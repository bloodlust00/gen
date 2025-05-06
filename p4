import gensim.downloader as api
from transformers import pipeline
import torch


# Load the GloVe embedding model
embedding_model = api.load("glove-wiki-gigaword-100")

original_prompt = "Describe the beautiful landscape during sunset"

def enrich_prompt(prompt, embedding_model, n=5):
    words = prompt.split()
    enriched_prompt = []

    for word in words:
        # Convert word to lowercase to match the embeddings vocabulary
        word_lower = word.lower()
        # Check if the word exists in embedding model's vocabulary
        if word_lower in embedding_model:
            similar_words = embedding_model.most_similar(word_lower, topn=n)
            similar_word_list = [w[0] for w in similar_words]
            # Add the original word and top similar word (optional: more words)
            enriched_prompt.append(word + " (" + ', '.join(similar_word_list[:1]) + ")")
        else:
            enriched_prompt.append(word)
    
    return ' '.join(enriched_prompt)

# Enrich the prompt
enriched_prompt = enrich_prompt(original_prompt, embedding_model)

# Load the GPT-2 generator pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text using original prompt
original_response = generator(original_prompt, max_length=50, num_return_sequences=1)

# Generate text using enriched prompt
enriched_response = generator(enriched_prompt, max_length=50, num_return_sequences=1)

print("Original Prompt Response:")
print(original_response[0]['generated_text'])

print("\nEnriched Prompt:")
print(enriched_prompt)
print("Enriched Prompt Response:")
print(enriched_response[0]['generated_text'])
