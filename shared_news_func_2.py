import os
import re
import html
from pathlib import Path
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DATA_DIR = Path("data_v20")
LANCE_DB_PATH = DATA_DIR / "news_lance"
ENTITY_MAP_FILE = DATA_DIR / "entity_map.json"

ENTITIES_CSV = DATA_DIR / "entities_intelligence.csv"
SOURCES_CSV  = DATA_DIR / "sources_intelligence.csv"
ARTICLES_CSV = DATA_DIR / "articles_database.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)

RSS_FEEDS = [
    "https://www.vox.com/rss/world-politics/index.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://www.cnbc.com/id/100727362/device/rss/rss.html",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.bing.com/news/search?q=USA&format=rss",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "https://rss.csmonitor.com/feeds/usa",
    "https://www.newsweek.com/rss",
    "https://qz.com/rss",
    "https://www.theguardian.com/us-news/rss",
    "https://globalnews.ca/feed/",
    "http://news.yahoo.com/rss/",
    "https://www.uscourts.gov/news/rss",
]

SOURCE_TO_ENTITY_MAP = {
    "bbc":           ["BBC", "British Broadcasting Corporation"],
    "nytimes":       ["New York Times", "NYT"],
    "cnbc":          ["CNBC", "NBCUniversal"],
    "aljazeera":     ["Al Jazeera"],
    "foxnews":       ["Fox News", "Fox"],
    "theguardian":   ["The Guardian", "Guardian News"],
    "vox":           ["Vox", "Vox Media"],
    "newsweek":      ["Newsweek"],
    "csmonitor":     ["CS Monitor", "Christian Science Monitor"],
    "globalnewssite":["Global News"],
    "yahoonews":     ["Yahoo", "Yahoo News"],
    "uscourts":      ["US Courts", "US Courts News"],
}

# ---------------------------------------------------------------------------
# SHARED TEXT CLEANER
# ---------------------------------------------------------------------------
def clean_text_robust(raw_text: str) -> str:
    """Decode HTML entities, strip tags, collapse whitespace."""
    if not raw_text:
        return ""
    text = str(raw_text)
    for _ in range(3):
        text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text(separator=" ")
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())
