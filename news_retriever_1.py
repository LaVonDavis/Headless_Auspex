import asyncio
import aiohttp
import feedparser
import lancedb
import polars as pl
import hashlib
import time
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
import shared_news_func_2 # Import our new shared config

class AsyncNewsFetcher:
    def __init__(self, user_agent="NewsAnalyzer/9.0", max_concurrent=15):
        self.headers = {"User-Agent": user_agent}
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_feed(self, session, url):
        try:
            async with session.get(url, timeout=8) as response:
                if response.status != 200: return []
                xml = await response.text()
                feed = await asyncio.to_thread(feedparser.parse, xml)
                return feed.entries
        except Exception: return []

    async def fetch_full_article(self, session, url):
        async with self.semaphore:
            try:
                async with session.get(url, timeout=12, allow_redirects=True) as response:
                    if response.status != 200:
                        return ""
                    html_content = await response.text()
                    return await asyncio.to_thread(self._extract_text_bs4, html_content)
            except Exception:
                return ""

    def _extract_text_bs4(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "form", "meta", "noscript", "button"]):
            element.decompose()
            
        text_blocks = []
        articles = soup.find_all('article')
        if articles:
            for article in articles:
                text_blocks.extend([p.get_text(separator=" ", strip=True) for p in article.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])])
        else:
            text_blocks.extend([p.get_text(separator=" ", strip=True) for p in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])])
            
        valid_blocks = [b for b in text_blocks if len(b.split()) > 4]
        text = ' '.join(valid_blocks)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:25000]

    async def gather_data(self, feed_urls, existing_ids):
        async with aiohttp.ClientSession(headers=self.headers) as session:
            print(f"📡 Scanning {len(feed_urls)} feeds...")
            feed_tasks = [self.fetch_feed(session, url) for url in feed_urls]
            all_feeds = await asyncio.gather(*feed_tasks)
            entries = [item for sublist in all_feeds for item in sublist]
            
            pending_entries = []
            seen_uids = set()
            
            for entry in entries:
                link = getattr(entry, 'link', '')
                if not link: continue
                uid = hashlib.md5(link.encode()).hexdigest()
                
                if uid in existing_ids or uid in seen_uids: 
                    continue
                seen_uids.add(uid)
                pending_entries.append((uid, entry, link))
                
            if not pending_entries:
                return []

            print(f"📄 Fetching full text for {len(pending_entries)} new articles...")
            article_tasks = [self.fetch_full_article(session, link) for _, _, link in pending_entries]
            full_texts = await asyncio.gather(*article_tasks)
            
            final_data = []
            for (uid, entry, link), full_text in zip(pending_entries, full_texts):
                content = full_text
                if not content or len(content) < 200:
                    content = getattr(entry, 'summary', getattr(entry, 'description', ""))
                    if not content or len(content) < 10: 
                        content = getattr(entry, 'title', "")

                content = shared_news_func_2.clean_text_robust(content)
                title = shared_news_func_2.clean_text_robust(getattr(entry, 'title', ''))

                pub_date = datetime.now().isoformat()
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    dt = datetime(*entry.published_parsed[:6])
                    pub_date = dt.isoformat()

                source = urlparse(link).netloc.replace('www.', '')
                
                # Notice we add a status flag for the queue system!
                final_data.append({
                    "id": uid, "title": title, "url": link, "source": source, 
                    "published": pub_date, "text": content, "status": "pending"
                })
                
            return final_data

def run_fetcher():
    start_time = time.time()
    db = lancedb.connect(shared_news_func_2.LANCE_DB_PATH)
    fetcher = AsyncNewsFetcher()
    
    # Check existing IDs in BOTH raw and processed tables to avoid fetching again
    existing_ids = set()
    for table_name in ["raw_articles", "articles"]:
        if table_name in db.table_names():
            try:
                ids = db.open_table(table_name).search().select(["id"]).limit(100000).to_pandas()["id"].tolist()
                existing_ids.update(ids)
            except Exception: pass

    raw_data = asyncio.run(fetcher.gather_data(shared_news_func_2.RSS_FEEDS, existing_ids))

    if not raw_data:
        print("💤 No new articles found to fetch.")
        return

    print("💾 Saving raw data to queue (LanceDB)...")
    df_raw = pl.DataFrame(raw_data)
    
    if "raw_articles" in db.table_names(): 
        db.open_table("raw_articles").add(df_raw)
    else:
        db.create_table("raw_articles", data=df_raw)

    print(f"🎉 Fetcher Done! Queued {len(raw_data)} items in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    run_fetcher()
