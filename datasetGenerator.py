from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') #gpt2-large
model = GPT2LMHeadModel.from_pretrained('gpt2') #gpt2-large

def generate_sentences(prompt, num_sentences=2, max_length=50, top_k=50, top_p=0.95, temperature=1.0, num_return_sequences=2):
    generated_sentences = set()  # Use a set to avoid duplicates
    
    while len(generated_sentences) < num_sentences:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate text
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            do_sample=True  # Enable sampling
        )
        
        # Decode the generated text and add to the set
        for sequence in output:
            generated_sentence = tokenizer.decode(sequence, skip_special_tokens=True)
            generated_sentences.add(generated_sentence)
        
        # Print progress
        if len(generated_sentences) % 100 == 0:
            print(f"Generated {len(generated_sentences)} unique sentences")
    
    return list(generated_sentences)

# Example prompts
prompts = [
    "I feel so relaxed when",
    "The best way to relax is",
    "Relaxing moments include",
    "When I'm feeling calm, I",
    "To unwind, I usually"
]

# Generate sentences using multiple prompts
generated_sentences = []
for prompt in prompts:
    generated_sentences.extend(generate_sentences(prompt, num_sentences=200))

# Ensure we have exactly 1000 unique sentences
generated_sentences = generated_sentences[:5]

# Create a DataFrame
data = {
    'sentence': generated_sentences,
    'label': ['relax'] * len(generated_sentences)
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('relax_sentiment_dataset.csv', index=False)

# Verify the dataset
print(df.head())
print(f"Total number of sentences: {len(df)}")
