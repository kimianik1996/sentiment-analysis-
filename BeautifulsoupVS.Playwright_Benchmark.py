import streamlit as st
import pandas as pd
import plotly.express as px

#Sample Scraped Data
book_records = [
    {"Title": "A Light in the Attic", "Price": "£51.77", "Availability": "In stock"},
    {"Title": "Tipping the Velvet", "Price": "£53.74", "Availability": "In stock"},
    {"Title": "Soumission", "Price": "£50.10", "Availability": "In stock"},
    {"Title": "Sharp Objects", "Price": "£47.82", "Availability": "In stock"},
    {"Title": "Sapiens: A Brief History of Humankind", "Price": "£54.23", "Availability": "In stock"},
    {"Title": "The Requiem Red", "Price": "£22.65", "Availability": "In stock"},
    {"Title": "The Dirty Little Secrets of Getting Your Dream Job", "Price": "£33.34", "Availability": "In stock"},
    {"Title": "The Coming Woman", "Price": "£17.93", "Availability": "In stock"},
    {"Title": "The Boys in the Boat", "Price": "£22.60", "Availability": "In stock"},
    {"Title": "The Black Maria", "Price": "£52.15", "Availability": "In stock"},
]

#Benchmark Times
benchmark_data = {
    "BeautifulSoup": 0.45,
    "Playwright": 5.59
}

# layout
st.set_page_config(page_title="Beautifulsoup VS. Playwright", layout="wide")
st.title("📊Benchmark Dashboard")

st.markdown("This dashboard displays scraped sample data from [BooksToScrape](http://books.toscrape.com), and compares performance of **BeautifulSoup** and **Playwright**.")

#Collested data
st.subheader("📘 Sample Book Data")
st.dataframe(pd.DataFrame(book_records))

# Chart
st.subheader("⏱️ Benchmark Time Comparison (in seconds)")
st.bar_chart(benchmark_data)


# Outcome Section
st.subheader("💡 Final Thoughts & Recommendation")
st.markdown("""
- ✅ **BeautifulSoup** is excellent for fast, lightweight scraping of static HTML content.
- 🧪 **Playwright** is ideal for scraping dynamic websites that require full browser rendering.
- ⚡ In this benchmark, BeautifulSoup was over **12x faster**.
- 📌 **Recommendation**: I will Use BeautifulSoup for simple tasks like BooksToScrape. Use Playwright when pages use JavaScript or need user interaction.\n
-Note: Altough each time running scripts gave me different time taken to run but the one thing is stable which is playwright is more time consuming for this task.
""")
