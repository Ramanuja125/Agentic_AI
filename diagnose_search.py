import requests
from bs4 import BeautifulSoup

response = requests.get(
    "https://html.duckduckgo.com/html/",
    params={"q": "population of Berlin"},
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=10,
)

print("Status code:", response.status_code)
print("Response length:", len(response.text))
print("\nFirst 2000 chars of response:")
print(response.text[:2000])
print("\n\nLooking for common result containers:")
soup = BeautifulSoup(response.text, "html.parser")
for selector in [".result", ".result__body", ".web-result", "div.results_links", "a.result__a"]:
    matches = soup.select(selector)
    print(f"  {selector}: {len(matches)} matches")