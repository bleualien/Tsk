import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load your dataset
df = pd.read_csv("data/books.csv")

# Combine books into a context string
books_text = "\n".join(
    [f"Title: {row['title']}, Price: {row['price']}, Rating: {row['rating']}" for _, row in df.iterrows()]
)
    
# Load a small free LLM from Hugging Face
model_name = "google/flan-t5-small"#huggingface
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def ask_question(question, context):
    # Combine context and question for the model
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Example questions
questions = [
    "Which book is the cheapest?",
    "Which book is the most expensive?",
    "List all books with a Three-star rating or higher.",
    "How many books are there in total?",
]

for q in questions:
    print("\nQ:", q)
    print("A:", ask_question(q, books_text))
