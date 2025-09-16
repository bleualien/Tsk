import json
from llama_cpp import Llama

# Load products
def load_products(path="data/cleaned_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for k in ("products", "items", "results"):
            if k in data:
                data = data[k]
                break
    return data if isinstance(data, list) else []

# Filter products
def filter_products(products, category=None, price_min=None, price_max=None, availability_only=False):
    result = []
    for p in products:
        c = p.get("category", "").lower()
        if category and category.lower() not in c:
            continue
        price = p.get("price_min") or p.get("price_max")
        if price_min is not None and (price is None or price < price_min):
            continue
        if price_max is not None and (price is None or price > price_max):
            continue
        if availability_only and "in stock" not in p.get("availability", "").lower():
            continue
        result.append(p)
    return result

# Prepare product snippet
def prepare_snippet(products, limit=20):
    snippet = []
    for p in products[:limit]:
        snippet.append({
            "name": p.get("name"),
            "brand": p.get("brand"),
            "category": p.get("category"),
            "price_min": p.get("price_min"),
            "price_max": p.get("price_max"),
            "availability": p.get("availability")
        })
    return json.dumps(snippet, indent=2)

# LLaMA local recommendation
def get_recommendations_local(model, products_snippet, user_question, max_tokens=300):
    prompt = f"""
You are a helpful product recommendation assistant.
Here are some products:

{products_snippet}

User question: {user_question}

Give exactly 3 recommendations with 1-2 sentence reasoning each.
"""
    response = model(prompt, max_tokens=max_tokens)
    return response['choices'][0]['text'].strip()

# Example usage
if __name__ == "__main__":
    # Load products
    products = load_products("data/cleaned_data.json")

    # User filters
    category = input("Enter category (or skip): ").strip() or None
    price_min = input("Enter minimum price (or skip): ").strip()
    price_max = input("Enter maximum price (or skip): ").strip()
    availability_only = input("Only show available items? yes/no: ").strip().lower() == "yes"

    price_min = float(price_min) if price_min else None
    price_max = float(price_max) if price_max else None

    filtered = filter_products(products, category, price_min, price_max, availability_only)
    snippet = prepare_snippet(filtered)

    question = input("Enter your question for recommendation: ").strip()

    # Load local LLaMA model
    llama_model_path = "models/ggml-model-q4_0.bin"  # replace with your local LLaMA GGUF/ggml path
    model = Llama(model_path=llama_model_path)

    # Get recommendations
    answer = get_recommendations_local(model, snippet, question)
    print("\n---- Local Recommendations ----\n")
    print(answer)
