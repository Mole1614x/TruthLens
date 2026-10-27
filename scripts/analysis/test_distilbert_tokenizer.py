import keras_hub

# Load the pretrained DistilBERT tokenizer.
tokenizer = keras_hub.models.DistilBertTokenizer.from_preset(
    "distil_bert_base_en_uncased"
)

# Sample news text.
news_text = "The government announced a new policy today."

# Convert text into token IDs.
token_ids = tokenizer.tokenize(news_text)

# Display the results.
print("Original text:")
print(news_text)

print("\nToken IDs:")
print(token_ids)

# Convert the tensor into a list of individual IDs.
token_id_list = token_ids.numpy().tolist()

# Convert each individual ID into a readable token.
tokens = [
    tokenizer.id_to_token(token_id)
    for token_id in token_id_list
]

print("\nTokens:")
print(tokens)
