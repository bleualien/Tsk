
import os
import json
import textwrap
import argparse
from typing import List, Dict, Optional

try:
    import openai
except Exception as e:
    raise SystemExit(
        "This script requires the 'openai' package. Install it with: pip install openai"
    )


#  Utility (data loading + normalization) 

def find_json_file(preferred_paths: List[str]) -> Optional[str]:
    """Return the first existing path from a list or None."""
    for p in preferred_paths:
        if os.path.exists(p):
            return p
    return None


def load_and_normalize_products(path: str) -> List[Dict]:
    """
    Load JSON from `path` and produce a list of normalized product dicts with these keys:
      - name (str)
      - brand (str)
      - category (str)
      - price_min (float or None)
      - price_max (float or None)
      - availability (str)
      - image (str)

    The loader is forgiving and tries to handle common variations in field names.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # The file might be a dict with a top-level key (e.g. {'products': [...]})
    if isinstance(raw, dict):
        # try some common keys
        for k in ("products", "items", "results"):
            if k in raw and isinstance(raw[k], list):
                raw = raw[k]
                break

    if not isinstance(raw, list):
        raise ValueError("Expected JSON file to contain a list of products or a dict with a products list.")

    normalized = []

    def to_float(v):
        try:
            return float(v)
        except Exception:
            return None

    for obj in raw:
        if not isinstance(obj, dict):
            continue
        # name
        name = obj.get("name") or obj.get("title") or obj.get("product_name") or ""

        brand = obj.get("brand") or obj.get("manufacturer") or ""

        # categories: could be string or list
        cat = obj.get("categories") or obj.get("category") or obj.get("category_name") or ""
        if isinstance(cat, list) and cat:
            category = str(cat[0])
        else:
            category = str(cat)

        # prices: try multiple possible shapes
        price_min = None
        price_max = None

        prices = obj.get("prices") or obj.get("price") or obj.get("pricing")
        if isinstance(prices, dict):
            # common field names
            price_min = prices.get("amountMin") or prices.get("min") or prices.get("price_min")
            price_max = prices.get("amountMax") or prices.get("max") or prices.get("price_max")
        else:
            # top-level numeric fields
            price_min = obj.get("price_min") or obj.get("amountMin") or obj.get("amount") or obj.get("price")
            price_max = obj.get("price_max") or obj.get("amountMax") or obj.get("price")

        price_min = to_float(price_min)
        price_max = to_float(price_max)

        # availability
        availability = ""
        if isinstance(prices, dict):
            availability = prices.get("availability") or obj.get("availability") or ""
        else:
            availability = obj.get("availability") or ""

        # image
        image = obj.get("imageURLs") or obj.get("image") or obj.get("image_url") or ""
        # if image is a list, pick the first
        if isinstance(image, list) and image:
            image = image[0]

        normalized.append({
            "name": str(name),
            "brand": str(brand),
            "category": str(category),
            "price_min": price_min,
            "price_max": price_max,
            "availability": str(availability),
            "image": str(image),
        })

    return normalized


# Filtering / Small helpers 

def unique_categories(products: List[Dict]) -> List[str]:
    raw_cats = {(p.get("category") or "").strip() for p in products if (p.get("category") or "")}
    cleaned = []
    for c in raw_cats:
        # Take only the first part before a comma
        main = c.split(",")[0].strip()
        # Singularize if ends with "s"
        if main.endswith("s") and len(main) > 3:
            main = main[:-1]
        cleaned.append(main.capitalize())
    return sorted(set(cleaned))




def filter_products(products: List[Dict],
                    category: Optional[str] = None,
                    price_min: Optional[float] = None,
                    price_max: Optional[float] = None,
                    availability_only: bool = False) -> List[Dict]:
    out = []
    for p in products:
        # category match (if requested) - case-insensitive substring match
        if category:
            if not p.get("category") or category.lower() not in p.get("category", "").lower():
                continue
        # price_min: compare to product price_min if available; if product has no price_min but has price_max,
        # use that; if neither, keep the product (unknown price)
        p_min = p.get("price_min")
        p_max = p.get("price_max")

        if price_min is not None:
            # if both None -> unknown price -> exclude (common for marketplaces) -> we exclude to be predictable
            if p_min is None and p_max is None:
                continue
            # use p_min if available else p_max
            p_val = p_min if p_min is not None else p_max
            if p_val is None or p_val < price_min:
                continue
        if price_max is not None:
            if p_min is None and p_max is None:
                continue
            p_val = p_min if p_min is not None else p_max
            if p_val is None or p_val > price_max:
                continue

        if availability_only:
            avail = (p.get("availability") or "").lower()
            if not any(tok in avail for tok in ("in stock", "available", "instock", "yes")):
                continue

        out.append(p)
    return out


#  LLM integration 

def prepare_products_snippet(products: List[Dict], limit: int = 25) -> str:
    """
    Create a compact, human-readable JSON-like snippet of the top `limit` products to include in the prompt.
    We keep only the most useful fields to keep the prompt small.
    """
    snippet = []
    for p in products[:limit]:
        snippet.append({
            "name": p.get("name"),
            "brand": p.get("brand"),
            "category": p.get("category"),
            "price_min": p.get("price_min"),
            "price_max": p.get("price_max"),
            "availability": p.get("availability"),
        })
    return json.dumps(snippet, indent=2)


def ask_llm_for_recommendations(api_key: str,
                                products_snippet: str,
                                user_question: str,
                                price_min, price_max,
                                availability_text: str,
                                model: str = "gpt-3.5-turbo") -> str:
    """
    Call OpenAI ChatCompletion to get recommendations. Returns assistant text output.
    """
    openai.api_key = api_key

    system_prompt = (
        "You are a helpful product recommendation assistant.\n"
        "Given a set of products (name, brand, category, price range, availability) and a user's question,"
        " return the top 3 recommendations with a 1-2 sentence justification for each.\n"
        "When relevant, prefer products that match the user's price constraints and availability.\n"
        "Output format:\n"
        "1) <product name> (brand) — Reason. Price: <min>-<max>\n"
        "2) ...\n"
        "Also, if the user's question is impossible to satisfy (no matches), explain clearly and suggest next steps."
    )

    user_prompt = textwrap.dedent(f"""
    Here are the products (up to 25):
    {products_snippet}

    Constraints: availability = {availability_text}
    Price range: min={price_min} max={price_max}

    User question: "{user_question}"

    Please provide exactly 3 numbered recommendations (if 3 are available). If fewer than 3 match, return the matching ones and explain why fewer were returned.
    Keep each recommendation to 1-2 short sentences.
    """)

    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=450,
        )
    except Exception as e:
        raise RuntimeError(f"LLM request failed: {e}")

    return resp["choices"][0]["message"]["content"].strip()


#  CLI / main flow

def interactive_choice(prompt_text: str, choices: List[str]) -> Optional[str]:
    """Show numbered choices and return the selected value or None for skip."""
    if not choices:
        return None
    print(prompt_text)
    for i, c in enumerate(choices, start=1):
        print(f"  {i}. {c}")
    print("  (press Enter to skip)")
    while True:
        s = input("Select a number or press Enter to skip: ").strip()
        if s == "":
            return None
        if s.isdigit() and 1 <= int(s) <= len(choices):
            return choices[int(s) - 1]
        else:
            print("Invalid selection, try again.")


def get_float_input(prompt_text: str) -> Optional[float]:
    s = input(prompt_text + " (press Enter to skip): ").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        print("Couldn't parse that as a number. Please try again.")
        return get_float_input(prompt_text)


def main():
    parser = argparse.ArgumentParser(description="Simple product recommender using an LLM")
    parser.add_argument("--file", "-f", default=None, help="Path to cleaned_data.json (optional)")
    parser.add_argument("--model", "-m", default="gpt-3.5-turbo", help="LLM model to use (default gpt-3.5-turbo)")
    args = parser.parse_args()

    # locate JSON file
    candidate = [
        args.file,
        "cleaned_data.json",
        os.path.join("data", "cleaned_data.json"),
        os.path.join("data", "cleaned_data_minimal.json"),
    ]
    candidate = [c for c in candidate if c]
    path = find_json_file(candidate)
    if not path:
        print("Could not find cleaned_data.json. Please place your JSON file as 'cleaned_data.json' or supply --file path.")
        return

    try:
        products = load_and_normalize_products(path)
    except Exception as e:
        print(f"Failed to load products: {e}")
        return

    if not products:
        print("No products found in the JSON file.")
        return

    print(f"Loaded {len(products)} products from {path}")

       # interactive selection
    cats = unique_categories(products)

        # Ask user to choose a category
    selected_category = None
    print("Available categories:")
    for i, c in enumerate(cats, start=1):
        print(f"  {i}. {c}")
    print("  (press Enter to skip)")

    while True:
        choice = input("Select a category number (or press Enter to skip): ").strip()
        if choice == "":
            selected_category = None
            break
        if choice.isdigit() and 1 <= int(choice) <= len(cats):
            selected_category = cats[int(choice) - 1]
            break
        else:
            print("Invalid choice. Try again.")

    # price inputs
    price_min = get_float_input("Enter minimum price")
    price_max = get_float_input("Enter maximum price")

    avail_input = input("Only show available items? (yes/no, default no): ").strip().lower()
    availability_only = avail_input == "yes"

    # now filter products (this was missing before!)
    filtered = filter_products(products, selected_category, price_min, price_max, availability_only)

    if not filtered:
        print("No products matched your filters. Showing products from the selected category (if any) or all products.")
        filtered = filter_products(products, selected_category)



    # show a short preview of first few matches
    print(f"\nFound {len(filtered)} candidate products after filtering. Showing first 8 as preview:\n")
    for p in filtered[:8]:
        print(f"- {p['name'][:80]} | {p['brand']} | {p['category']} | price: {p['price_min']}-{p['price_max']} | avail: {p['availability']}")

    # get user's free-form question
    print("\nYou can now ask a natural language question about these products. Examples:\n  - 'Which is best value under $800?'\n  - 'Recommend a budget monitor for office use'\n")
    user_question = input("Enter your question: ").strip()
    if not user_question:
        print("No question entered — exiting.")
        return

    # ensure API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set. Please set it and try again. See script header for instructions.")
        return

    # prepare snippet and call LLM
    snippet = prepare_products_snippet(filtered, limit=25)
    availability_text = "available only" if availability_only else "all"
    try:
        answer = ask_llm_for_recommendations(
            api_key=api_key,
            products_snippet=snippet,
            user_question=user_question,
            price_min=price_min if price_min is not None else "any",
            price_max=price_max if price_max is not None else "any",
            availability_text=availability_text,
            model=args.model,
        )
    except Exception as e:
        print(f"Request to LLM failed: {e}")
        return

    print("\n---- LLM Recommendations ----\n")
    print(answer)
    print("\n---- End ----\n")


if __name__ == "__main__":
    main()
