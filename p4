import gensim.downloader as api
from transformers import pipeline

embedding_model = api.load("glove-wiki-gigaword-100")

original_prompt = "Describe the beautiful landscape during sunset"

def enrich_prompt(prompt, embedding_model, n=5):
    words = prompt.split()
    enriched_prompt = []

    for word in words:
        
        word_lower = word.lower()
        
        if word_lower in embedding_model:
            similar_words = embedding_model.most_similar(word_lower, topn=n)
            similar_word_list = [w[0] for w in similar_words]
            enriched_prompt.append(word + " (" + ', '.join(similar_word_list[:1]) + ")")
        else:
            enriched_prompt.append(word)
    return ' '.join(enriched_prompt)
enriched_prompt = enrich_prompt(original_prompt, embedding_model)
generator = pipeline("text-generation", model="gpt2")
original_response = generator(original_prompt, max_length=50, num_return_sequences=1)
enriched_response = generator(enriched_prompt, max_length=50, num_return_sequences=1)
print("Original Prompt Response:")
print(original_response[0]['generated_text'])
print("\nEnriched Prompt:")
print(enriched_prompt)
print("Enriched Prompt Response:")
print(enriched_response[0]['generated_text'])