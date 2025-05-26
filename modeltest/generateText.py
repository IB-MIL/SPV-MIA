from transformers import pipeline, set_seed

# Initialize text generation pipeline
generator = pipeline('text-generation', model='gpt2-xl')

# Set seed for reproducibility


# Generate text based on your prompt
output = generator("Yo ho ho and a bottle of rum,", max_length=30, num_return_sequences=5)

# Print the generated texts
for idx, text in enumerate(output):
    print(f"Generated text {idx + 1}: {text['generated_text']}")
