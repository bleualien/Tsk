# scraper.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

BASE_URL = "https://books.toscrape.com/catalogue/category/books_1/page-{}.html"

books_data = []

# Scrape first 5 pages (adjust as needed)
for page in range(1, 6):
    url = BASE_URL.format(page)
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch page {page}")
        continue

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    for book in books:
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text
        rating_class = book.find("p", class_="star-rating")["class"]
        rating = rating_class[1]  # 'One', 'Two', etc.
        books_data.append({
            "title": title,
            "price": price,
            "rating": rating
        })

# Save to CSV
df = pd.DataFrame(books_data)
df.to_csv("data/books.csv", index=False)
print(f"Scraped {len(books_data)} books. Saved to data/books.csv")
