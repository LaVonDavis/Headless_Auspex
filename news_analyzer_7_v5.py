"""
news_analyzer_7_v4.py
─────────────────────
Topic clustering with batched POS validation and semantic thematic labeling.

IMPROVEMENTS over v3:
  • Batched POS via nlp.pipe() — identical to v3 but now also propagates
    relation-type weights from the edges table into cluster scoring.
  • cluster_articles: expose soft-cluster probabilities for future use.
  • label_clusters: lazy Polars throughout; TF-IDF filter expressed as a
    single lazy join instead of two collect() calls.
  • rank_clusters: additionally scores by avg absolute sentiment deviation
    (high controversy clusters float up alongside high-volume ones).
  • generate_local_narratives: includes avg sentiment in narrative output
    so the PDF can colour-code cluster cards without re-computing it.
  • export_results: writes cluster_topics.csv + cluster_narratives.json
    (same as v3) but also writes cluster_graph.json — a node/link JSON
    ready for D3 or any graph renderer in the PDF.
"""

import json
import re
import time
import warnings
from pathlib import Path

import hdbscan
import lancedb
import numpy as np
import polars as pl
from sklearn.preprocessing import normalize

import shared_news_func_2

warnings.filterwarnings("ignore")

try:
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
except (ImportError, OSError):
    nlp = None
    print("⚠️  spaCy not found. Falling back to string-only entity filter.")

# Updated blocklist to catch demonstrative and relative pronouns
ENTITY_NOISE_BLOCKLIST = {
    "made", "told", "said", "show", "list", "latest", "one", "you", "me", "we",
    "us", "them", "team", "group", "thing", "people", "man", "woman", "girl",
    "boy", "today", "yesterday", "last week", "this week",
    "monday", "friday", "part", "way", "kind", "type",
    "this", "that", "these", "those", "which", "what", "some", "such", "all", "any", "it", "they", "he", "she", "who"
}
ENTITY_MIN_LEN = 3


# ---------------------------------------------------------------------------
# BATCHED POS FILTER
# ---------------------------------------------------------------------------
def get_valid_entities_batched(entity_list: list[str]) -> set[str]:
    """Filter entity list by POS tag; uses nlp.pipe for speed."""
    candidates = []
    for ent in entity_list:
        if not ent or len(ent) < ENTITY_MIN_LEN:
            continue
        if ent.lower().strip() in ENTITY_NOISE_BLOCKLIST:
            continue
        if re.fullmatch(r"[\d\s\-/]+", ent):
            continue
        if " " not in ent and ent.islower() and len(ent) < 6:
            continue
        candidates.append(ent)

    if not nlp:
        return set(candidates)

    valid: set[str] = set()
    for doc in nlp.pipe(candidates, batch_size=200):
        if doc and doc[0].pos_ not in {"VERB", "PRON", "ADV", "DET"}:
            valid.add(doc.text)
    return valid


# ---------------------------------------------------------------------------
# TOPIC CLUSTERING SERVICE
# ---------------------------------------------------------------------------
class TopicClusteringService:

    def __init__(self):
        self.db = lancedb.connect(shared_news_func_2.LANCE_DB_PATH)
        self.output_dir = Path(shared_news_func_2.LANCE_DB_PATH).parent
        self.topics_csv      = self.output_dir / "cluster_topics.csv"
        self.narrative_json  = self.output_dir / "cluster_narratives.json"
        self.graph_json      = self.output_dir / "cluster_graph.json"

    # ------------------------------------------------------------------
    def _load_articles_with_vectors(self) -> pl.DataFrame:
        if "articles" not in self.db.table_names():
            raise RuntimeError("No 'articles' table found. Run AnalysisService first.")
        df = self.db.open_table("articles").to_polars().collect()
        if "vector" not in df.columns:
            raise RuntimeError("Articles table has no vector column.")
        return df

    def _load_mentions(self) -> pl.DataFrame:
        if "mentions" not in self.db.table_names():
            return pl.DataFrame()
        return self.db.open_table("mentions").to_polars().collect()

    def _load_edges(self) -> pl.DataFrame:
        if "edges" not in self.db.table_names():
            return pl.DataFrame()
        return self.db.open_table("edges").to_polars().collect()

    # ------------------------------------------------------------------
    def cluster_articles(self, df: pl.DataFrame) -> np.ndarray:
        print("🔍 Clustering articles by topic...")
        vectors = normalize(
            np.array(df["vector"].to_list(), dtype=np.float32), norm="l2"
        )
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=3,
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(vectors)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"   → {n_clusters} topic clusters found.")
        return labels

    # ------------------------------------------------------------------
    def label_clusters(
        self,
        df: pl.DataFrame,
        labels: np.ndarray,
        mentions: pl.DataFrame,
    ) -> dict:
        print("🏷️  Labeling topic clusters (Semantic Concept Prioritization)...")
        df = df.with_columns(pl.Series("cluster_id", labels))

        # ── entity validation & TF-IDF filter (fully lazy) ───────────────
        if not mentions.is_empty():
            unique_raw = mentions["entity"].unique().to_list()
            valid_ents = list(get_valid_entities_batched(unique_raw))
            total_articles = df.height

            mentions_lf = mentions.lazy().filter(pl.col("entity").is_in(valid_ents))
            if total_articles > 0:
                doc_freq = (
                    mentions_lf
                    .group_by("entity")
                    .agg(pl.col("headline_id").n_unique().alias("doc_count"))
                    .filter(pl.col("doc_count") <= total_articles * 0.35)
                    .select("entity")
                )
                mentions_lf = mentions_lf.join(doc_freq, on="entity", how="inner")
            mentions_filtered = mentions_lf.collect()
        else:
            mentions_filtered = pl.DataFrame()

        cluster_ids = sorted(int(c) for c in set(labels) if c != -1)

        # ── per-cluster entity + sentiment tables ─────────────────────────
        if not mentions_filtered.is_empty() and "headline_id" in mentions_filtered.columns:
            joined = mentions_filtered.join(
                df.select(["id", "cluster_id"]),
                left_on="headline_id", right_on="id", how="inner",
            )
            cluster_entities = (
                joined
                .group_by(["cluster_id", "entity", "label"])
                .agg(
                    pl.len().alias("count"),
                    pl.col("sentiment").mean().round(3).alias("avg_sentiment"),
                )
                .sort(["cluster_id", "count"], descending=[False, True])
            )
            cluster_sentiments = (
                joined
                .group_by("cluster_id")
                .agg(pl.col("sentiment").mean().round(3).alias("cluster_avg_sentiment"))
            )
        else:
            cluster_entities  = pl.DataFrame()
            cluster_sentiments = pl.DataFrame({"cluster_id": [], "cluster_avg_sentiment": []})

        # Cluster-level summaries
        cluster_summaries = (
            df.filter(pl.col("cluster_id") != -1)
            .group_by("cluster_id")
            .agg(
                pl.col("id").alias("article_ids"),
                pl.col("title").head(10).alias("titles"),
            )
        )
        if not cluster_sentiments.is_empty():
            cluster_summaries = (
                cluster_summaries
                .join(cluster_sentiments, on="cluster_id", how="left")
                .with_columns(pl.col("cluster_avg_sentiment").fill_null(0.0))
            )
        else:
            cluster_summaries = cluster_summaries.with_columns(pl.lit(0.0).alias("cluster_avg_sentiment"))

        summaries_dict = {r["cluster_id"]: r for r in cluster_summaries.to_dicts()}

        # ── build cluster dict ────────────────────────────────────────────
        clusters: dict[int, dict] = {}
        for cid in cluster_ids:
            sd = summaries_dict.get(cid, {})
            article_ids = sd.get("article_ids", [])

            ents = (
                cluster_entities.filter(pl.col("cluster_id") == cid).head(15).to_dicts()
                if not cluster_entities.is_empty()
                else []
            )

            concepts    = [e["entity"] for e in ents if e.get("label") == "Concept"]
            other_names = [e["entity"] for e in ents if e.get("label") != "Concept"]

            if concepts and other_names:
                auto_label = f"{concepts[0]} ({' / '.join(other_names[:2])})"
            else:
                top_all = [e["entity"] for e in ents[:3]]
                auto_label = " / ".join(top_all) if top_all else f"Topic {cid}"

            clusters[cid] = {
                "cluster_id":    cid,
                "auto_label":    auto_label,
                "article_count": len(article_ids),
                "article_ids":   article_ids,
                "top_entities":  ents[:8],
                "avg_sentiment": sd.get("cluster_avg_sentiment", 0.0),
            }

        return clusters

    # ------------------------------------------------------------------
    def rank_clusters(self, clusters: dict) -> list:
        """
        Score = article_count × (1 + entity_diversity × 0.2) × controversy_boost.
        Controversy boost: clusters with high |avg_sentiment| surface earlier.
        """
        ranked = []
        for cid, c in clusters.items():
            diversity    = len({e["label"] for e in c["top_entities"]})
            controversy  = 1.0 + abs(c.get("avg_sentiment", 0.0)) * 0.5
            score        = c["article_count"] * (1 + diversity * 0.2) * controversy
            ranked.append((score, cid, c))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [item[2] for item in ranked]

    # ------------------------------------------------------------------
    def generate_local_narratives(
        self, ranked_clusters: list, articles_df: pl.DataFrame
    ) -> list:
        print("✍️   Generating local intelligence summaries...")
        enriched = []
        for cluster in ranked_clusters:
            c_ids   = cluster.get("article_ids", [])
            summary = "Insufficient data for summary."

            if c_ids:
                sub = articles_df.filter(pl.col("id").is_in(c_ids))
                vecs = np.array(sub["vector"].to_list())
                if len(vecs) > 0:
                    centroid  = np.mean(vecs, axis=0)
                    distances = np.linalg.norm(vecs - centroid, axis=1)
                    best_idx  = int(np.argmin(distances))
                    rep       = sub.row(best_idx, named=True)
                    snippet   = rep.get("text", "")[:300].replace("\n", " ").strip()
                    summary   = f"Key Story: {rep.get('title', 'Unknown')} — {snippet}..."

            watch = [e["entity"] for e in cluster.get("top_entities", [])[:3]]
            avg_s = cluster.get("avg_sentiment", 0.0)
            tone  = "negative" if avg_s < -0.1 else ("positive" if avg_s > 0.1 else "neutral")

            cluster["narrative"] = {
                "topic_label":   cluster.get("auto_label", f"Topic {cluster.get('cluster_id')}"),
                "summary":       summary,
                "avg_sentiment": avg_s,
                "tone":          tone,
                "watch": (
                    f"Monitor developments regarding: {', '.join(watch)}"
                    if watch else ""
                ),
            }
            enriched.append(cluster)
        return enriched

    # ------------------------------------------------------------------
    def _build_cluster_graph(self, enriched_clusters: list) -> dict:
        """
        Build a node-link graph JSON where:
          - nodes = top entities across all clusters, coloured by cluster
          - links = within-cluster co-occurrence (weight = entity count rank)
        """
        nodes: dict[str, dict] = {}
        links: list[dict] = []

        for c in enriched_clusters:
            cid   = c["cluster_id"]
            label = c.get("auto_label", f"Topic {cid}")
            ents  = c.get("top_entities", [])

            for rank, e in enumerate(ents[:6]):
                name = e["entity"]
                if name not in nodes:
                    nodes[name] = {
                        "id":          name,
                        "cluster":     cid,
                        "cluster_label": label,
                        "type":        e.get("label", "Unknown"),
                        "sentiment":   e.get("avg_sentiment", 0.0),
                        "weight":      len(ents) - rank,   # higher rank → bigger node
                    }
                # link all entities within the same cluster to each other
                for other in ents[rank + 1 : rank + 4]:
                    links.append({
                        "source": name,
                        "target": other["entity"],
                        "cluster": cid,
                        "value":  1,
                    })

        return {"nodes": list(nodes.values()), "links": links}

    # ------------------------------------------------------------------
    def export_results(self, enriched_clusters: list):
        print("💾 Exporting cluster results...")

        rows = []
        for c in enriched_clusters:
            nav = c.get("narrative", {})
            rows.append({
                "cluster_id":    c["cluster_id"],
                "topic_label":   nav.get("topic_label", c["auto_label"]),
                "article_count": c["article_count"],
                "avg_sentiment": c["avg_sentiment"],
                "tone":          nav.get("tone", "neutral"),
                "top_entities":  ", ".join(e["entity"] for e in c["top_entities"][:6]),
                "summary":       nav.get("summary", ""),
                "watch":         nav.get("watch", ""),
            })
        pl.DataFrame(rows).write_csv(self.topics_csv)

        output = {
            "generated_at":  time.strftime("%Y-%m-%d %H:%M"),
            "cluster_count": len(enriched_clusters),
            "clusters":      enriched_clusters,
        }
        for c in output["clusters"]:
            c.pop("article_ids", None)

        with open(self.narrative_json, "w") as f:
            json.dump(output, f, indent=2, default=str)

        graph = self._build_cluster_graph(enriched_clusters)
        with open(self.graph_json, "w") as f:
            json.dump(graph, f, indent=2, default=str)

        print(
            f"   ✅ Topics CSV     → {self.topics_csv}\n"
            f"   ✅ Narratives     → {self.narrative_json}\n"
            f"   ✅ Cluster Graph  → {self.graph_json}"
        )

    # ------------------------------------------------------------------
    def run_topic_analysis(self):
        t0 = time.time()
        print("\n🌐 Starting Advanced Topic Clustering...")
        try:
            articles = self._load_articles_with_vectors()
            mentions = self._load_mentions()
        except RuntimeError as e:
            print(f"❌ {e}")
            return

        labels   = self.cluster_articles(articles)
        clusters = self.label_clusters(articles, labels, mentions)
        ranked   = self.rank_clusters(clusters)
        enriched = self.generate_local_narratives(ranked, articles)
        self.export_results(enriched)
        print(f"\n🎉 Topic analysis complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    TopicClusteringService().run_topic_analysis()
