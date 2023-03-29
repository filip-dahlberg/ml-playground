import transformers

model_name = "Helsinki-NLP/opus-mt-pl-en"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Set the maximum sequence length to 512 tokens
max_length = 512

# Read in the Polish text file
with open("input_file.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Prepend "translate Polish to English: " to the input text
text = "translate Polish to English: " + text

# Tokenize the text into chunks of up to 512 tokens
input_chunks = []
for i in range(0, len(text), max_length):
    chunk = text[i:i+max_length]
    input_chunks.append(chunk)

# Generate the English translation for each input chunk
output_chunks = []
for chunk in input_chunks:
    input_ids = tokenizer.encode(chunk, return_tensors="pt")
    output_ids = model.generate(input_ids)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_chunks.append(output_text)

# Concatenate the output chunks into a single string
output_text = "".join(output_chunks)

# Remove the "translate Polish to English: " prefix from the output text
output_text = output_text.replace("translate Polish to English: ", "")

# Write the English translation to a file
with open("output_file.txt", "w", encoding="utf-8") as f:
    f.write(output_text)
