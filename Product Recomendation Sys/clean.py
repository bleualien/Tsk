import pandas as pd
import json

# Step 1: Load CSV
file_path = "data/products.csv"
df = pd.read_csv(file_path)


# Step 2: Keep only important columns
columns_needed = [
    "name", "brand", "categories",
    "prices.amountMin", "prices.amountMax",
    "prices.availability", "imageURLs"
]

# Keep only columns that exist in CSV
df = df[[col for col in columns_needed if col in df.columns]]


# Step 3: Convert prices to numeric

df['prices.amountMin'] = pd.to_numeric(df['prices.amountMin'], errors='coerce')
df['prices.amountMax'] = pd.to_numeric(df['prices.amountMax'], errors='coerce')


# Step 4: Drop rows missing essential info

df = df.dropna(subset=["name", "categories", "prices.amountMin", "prices.amountMax", "prices.availability"])


# Step 5: Randomly select 20 products

if len(df) > 20:
    df = df.sample(n=20, random_state=42)


# Step 6: Convert to JSON-friendly structure

cleaned_products = []
for _, row in df.iterrows():
    product = {
        "name": row["name"],
        "brand": row.get("brand", ""),
        "categories": row["categories"],
        "prices": {
            "amountMin": float(row["prices.amountMin"]),
            "amountMax": float(row["prices.amountMax"]),
            "currency": "USD"  # default
        },
        "availability": row["prices.availability"],
        "imageURLs": row.get("imageURLs", "")
    }
    cleaned_products.append(product)


# Step 7: Save cleaned data as JSON

with open("data/cleaned_data.json", "w") as f:
    json.dump(cleaned_products, f, indent=4)

print(f"Saved {len(cleaned_products)} cleaned products to data/cleaned_data.json")
