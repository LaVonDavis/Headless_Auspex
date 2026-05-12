"""
news_report_builder_v6.py
─────────────────────────
Unified PDF intelligence report generator.

CHANGES vs v4
─────────────
OUTPUT
  • HTML and plaintext outputs removed; PDF is the sole deliverable.
  • DATA_DIR now reads from shared_news_func_2 (no more v12/v15 mismatch).

NEW VISUALISATIONS IN PDF
  • Topic Cluster page  — ranked topic-cluster table with colour-coded
    sentiment, article count, and key entities per cluster, sourced from
    cluster_topics.csv / cluster_narratives.json produced by analyzer_7.
  • Relationship Network chart  — matplotlib graph of the top-N entity
    nodes, edges weighted by SVO relation count vs co-occurrence.
  • Entity Sentiment Distribution histogram  — shows the full spread of
    entity average sentiment scores (not just crisis/positive buckets).
  • Controversy Heatmap table  — entities ranked by |current − average|
    sentiment delta, surfacing fast-moving stories.
  • Source Self-Reference scatter  — article volume vs self-ref % per
    source, distinguishing independent vs self-promotional outlets.

ARCHITECTURE IMPROVEMENTS
  • _build_chart_images() is the single chart-rendering entry point;
    all charts share one matplotlib figure lifecycle (no leaked figures).
  • Charts are rendered once and reused (same as v4, kept).
  • _build_executive_summary() extended to include cluster data.
  • generate_all() loop now also loads cluster data if available.
  • Removed generate_substack_html(), generate_substack_plaintext(),
    generate_api_json(), _save_chart_pngs() — PDF-only mode.
  • Lazy Polars frames used in _process_entities() and _process_sources().
  • _process_entities() now passes relation-type (SVO vs co-occurrence)
    into the Relations column so the PDF deep-dive can label them.
"""

import html
import io
import json
import os
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import lancedb
import matplotlib
import networkx as nx
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

import shared_news_func_2

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
DATA_DIR      = shared_news_func_2.DATA_DIR
LANCE_DB_PATH = shared_news_func_2.LANCE_DB_PATH
MODEL_PATH    = "/home/lichking/Desktop/AI_Lab/amoral-gemma3-4b-v2-qat-q4_0.gguf"

# ---------------------------------------------------------------------------
# FILTERING CONFIG
# ---------------------------------------------------------------------------
# Updated blocklist to retroactively filter bad pronouns out of existing data
ENTITY_BLOCKLIST = {
    "he","she","they","we","you","who","whom","whose","her","his","their","it","i",
    "him","them","us","our","my","your","its",
    "monday","tuesday","wednesday","thursday","friday","saturday","sunday",
    "today","tomorrow","yesterday","years","year","months","month","week","weeks",
    "days","day","time","times","continue reading","breaking news","subscribe here",
    "subscribe","read more","sign up","newsletter","live updates","live blog",
    "click here","tap here","end","work","game","gulf","social media","case",
    "plan","talks","2026","2025","people","residents","investors","country","world",
    "this", "that", "these", "those", "which", "what", "some", "such", "all", "any", "where", "why", "how"
}
ENTITY_NOISE_PATTERNS = [
    r"^\d+$", r"^[a-z]{1,2}$", r"\.\.\.", r"^(the|a|an|and|or|but|in|on|at|to|for|of|with|by)$",
]
SNIPPET_BLOCKLIST = [
    "continue reading","subscribe","sign up","live updates","newsletter",
    "read more","click here","tap here","this story appeared in",
    "sign up for the","get newsletter","email alerts","breaking news",
    "you can find","follow us",
]
SNIPPET_MIN_LENGTH          = 150
SENTIMENT_CRISIS_THRESHOLD  = -0.30
SENTIMENT_POSITIVE_THRESHOLD = 0.20
MIN_MENTION_COUNT           = 3
MIN_POWER_SCORE             = 0.05
HIGH_VOLUME_OVERRIDE        = 15
SOURCE_SENTIMENT_BLOCKLIST  = {"finance.yahoo.com", "yahoo.com"}
HIGH_SELF_REF_EXCLUSION_THRESHOLD = 15.0

LABEL_NORMALISE = {
    "jnknown":"Unknown","unknown":"Unknown","PERSON":"Person",
    "ORG":"Organization","GPE":"Location","LOC":"Location",
    "EVENT":"Event","NORP":"Organization","PRODUCT":"Organization",
    "WORK_OF_ART":"Concept","LAW":"Concept","LANGUAGE":"Concept",
}
ENTITY_ALIASES = {
    "u.s.":"US","united states":"US","u.s.a.":"US","the us":"US",
    "trump":"Donald Trump","president trump":"Donald Trump",
    "u.k.":"UK","great britain":"UK","britain":"UK","england":"UK",
}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _is_noise_entity(name: str) -> bool:
    n = name.strip().lower()
    if not n or n in ENTITY_BLOCKLIST: return True
    for p in ENTITY_NOISE_PATTERNS:
        if re.search(p, n): return True
    if n[-1] in ".,;:!?-–—": return True
    return False

def _normalise_entity_name(name: str) -> str:
    return ENTITY_ALIASES.get(name.strip().lower(), name.strip())

def _normalise_label(label: str) -> str:
    return LABEL_NORMALISE.get(label.strip().lower(), label.strip().title()) if label else "Unknown"

def _clean_snippets(snippets: list) -> list:
    out = []
    for s in (snippets or []):
        if not s or len(s) < SNIPPET_MIN_LENGTH: continue
        if any(b in s.lower() for b in SNIPPET_BLOCKLIST): continue
        out.append(s.strip())
    return out

def _sentiment_arrow(current: float, average: float) -> str:
    delta = current - average
    if abs(delta) < 0.05: return "→ stable"
    return f"↑ (+{delta:.2f})" if delta > 0 else f"↓ ({delta:.2f})"

def _sentiment_colour(value: float) -> colors.Color:
    if value <= SENTIMENT_CRISIS_THRESHOLD:  return colors.HexColor("#c0392b")
    if value >= SENTIMENT_POSITIVE_THRESHOLD: return colors.HexColor("#27ae60")
    return colors.HexColor("#555555")

def _gloss_fallback(name: str, avg_sent: float, mentions: int, etype: str) -> str:
    tone = "negative" if avg_sent < -0.1 else ("positive" if avg_sent > 0.1 else "neutral")
    return (
        f"{name} ({etype.lower()}) drew {mentions} mentions "
        f"with predominantly {tone} coverage (avg {avg_sent:+.3f})."
    )

def _tbl_style(header_bg: str = "#2c3e50") -> list:
    return [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor(header_bg)),
        ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
        ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#f4f4f4")]),
        ("GRID",       (0,0), (-1,-1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1),4),
    ]

# ---------------------------------------------------------------------------
# MAIN ENGINE
# ---------------------------------------------------------------------------
class NewsIntelligenceEngine:

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.db  = None
        self.llm = None

        if not LANCE_DB_PATH.exists():
            print("❌ LanceDB not found. Run News Analyzer first.")
            return
        self.db = lancedb.connect(LANCE_DB_PATH)

        if Llama and os.path.exists(MODEL_PATH):
            print(f"🤖 Loading Local LLM ({MODEL_PATH})...")
            try:
                self.llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_gpu_layers=0, verbose=False)
                print("✅ AI Engine Ready.")
            except Exception as e:
                print(f"⚠️  LLM load failed: {e}")
        else:
            print("⚠️  Local LLM not found — AI glosses use data-driven fallback.")

    # ------------------------------------------------------------------
    # LLM HELPERS
    # ------------------------------------------------------------------
    def _llm_generate(self, messages: list, max_tokens: int = 1000) -> str:
        if not self.llm:
            return ""
        try:
            resp = self.llm.create_chat_completion(messages=messages, max_tokens=max_tokens)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️  LLM error: {e}")
            return ""

    def generate_ai_briefing(self, final_entities: pl.DataFrame) -> str:
        if not self.llm or final_entities.is_empty():
            return ""
        context = "\n".join(
            f"- {snip}"
            for ent in final_entities.head(10).to_dicts()
            for snip in (ent.get("SnippetsList") or [])[:3]
        )[:6000]
        return self._llm_generate([
            {"role": "system", "content": (
                "You are a Senior News Intelligence Analyst. Read the following recent news "
                "snippets. Write a cohesive 2-3 paragraph summary of the major actual news "
                "events happening today. Do NOT mention sentiment scores or data metrics."
            )},
            {"role": "user", "content": f"Raw News Snippets:\n{context}\n\nSummarise the day's major events."},
        ])

    def _generate_entity_glosses(self, final_entities: pl.DataFrame) -> dict:
        glosses: dict[str, str] = {}
        for ent in final_entities.head(20).to_dicts():
            name = ent.get("Entity_Name", "")
            avg  = ent.get("Average_Sentiment", 0.0)
            n    = ent.get("Mention_Count", 0)
            etype = ent.get("Type", "")
            ctx  = " ".join(_clean_snippets(list(ent.get("SnippetsList") or []))[:2])[:1200]

            if not ctx or not self.llm:
                glosses[name] = _gloss_fallback(name, avg, n, etype)
                continue

            gloss = self._llm_generate([{"role": "user", "content": (
                f"Entity: {name}\nSource text: {ctx}\n\n"
                f"Write exactly one concise sentence (max 30 words) summarising what is "
                f"happening with {name} in today's news. No sentiment scores. No preamble."
            )}], max_tokens=80)
            glosses[name] = gloss.split("\n")[0] if gloss else _gloss_fallback(name, avg, n, etype)

        return glosses

    # ------------------------------------------------------------------
    # DATA LOADING & PERIOD FILTER
    # ------------------------------------------------------------------
    def _load_tables(self):
        tables = list(self.db.table_names())
        arts   = self.db.open_table("articles").to_polars().collect()
        ments  = self.db.open_table("mentions").to_polars().collect()
        edges  = (
            self.db.open_table("edges").to_polars().collect()
            if "edges" in tables
            else pl.DataFrame({"source": [], "target": [], "relation": []})
        )
        return arts, ments, edges

    def _filter_by_period(self, articles_df, mentions_df, edges_df, period: str):
        now    = datetime.now()
        deltas = {"daily": 1, "weekly": 7, "monthly": 30}
        cutoff = (now - timedelta(days=deltas.get(period, 1))).strftime("%Y-%m-%d")

        filtered_arts = articles_df.filter(pl.col("published").str.slice(0,10) >= cutoff)
        if filtered_arts.is_empty():
            return filtered_arts, pl.DataFrame(schema=mentions_df.schema), pl.DataFrame(schema=edges_df.schema)

        valid_ids   = filtered_arts["id"].to_list()
        f_mentions  = mentions_df.filter(pl.col("headline_id").is_in(valid_ids))
        valid_ents  = f_mentions["entity"].unique().to_list() if not f_mentions.is_empty() else []
        f_edges     = (
            edges_df.filter(pl.col("source").is_in(valid_ents) & pl.col("target").is_in(valid_ents))
            if not edges_df.is_empty() else edges_df
        )
        return filtered_arts, f_mentions, f_edges

    # ------------------------------------------------------------------
    # ENTITY & SOURCE PROCESSING
    # ------------------------------------------------------------------
    def _process_entities(self, mentions_df, edges_df, articles_df) -> pl.DataFrame:
        print("🕸️  Calculating Entity Graphs...")

        if not edges_df.is_empty():
            G = nx.Graph()
            for e in edges_df.to_dicts():
                w = 2.0 if e.get("relation","co-occurrence") != "co-occurrence" else 1.0
                src, tgt = e["source"], e["target"]
                if G.has_edge(src, tgt):
                    G[src][tgt]["weight"] += w
                else:
                    G.add_edge(src, tgt, weight=w)
            try:
                pr = nx.pagerank(G, weight="weight")
                pr_df = pl.DataFrame({
                    "entity":            list(pr.keys()),
                    "Graph_Power_Score": [round(v*100, 2) for v in pr.values()],
                })
            except Exception:
                pr_df = pl.DataFrame({"entity":[], "Graph_Power_Score":[]})

            e1 = edges_df.select([
                pl.col("source").alias("entity"),
                pl.format("{} ({})", pl.col("target"), pl.col("relation")).alias("related"),
            ])
            e2 = edges_df.select([
                pl.col("target").alias("entity"),
                pl.format("{} ({})", pl.col("source"), pl.col("relation")).alias("related"),
            ])
            rels_grouped = (
                pl.concat([e1, e2])
                .group_by("entity")
                .agg(pl.col("related").drop_nulls().unique().alias("RelationsList"))
                .with_columns(pl.col("RelationsList").list.head(5).list.join(", ").alias("Relations"))
            )
            if not pr_df.is_empty():
                rels_grouped = rels_grouped.join(pr_df, on="entity", how="left")
        else:
            rels_grouped = pl.DataFrame(
                schema={"entity": pl.Utf8, "Relations": pl.Utf8,
                        "RelationsList": pl.List(pl.Utf8), "Graph_Power_Score": pl.Float64}
            )

        if mentions_df.is_empty():
            return pl.DataFrame()

        if not articles_df.is_empty() and "source" in articles_df.columns:
            mentions_src = (
                mentions_df
                .join(articles_df.select(["id","source"]), left_on="headline_id", right_on="id", how="left")
                .with_columns(pl.col("source").fill_null("unknown"))
                .with_columns((pl.col("source") + "||" + pl.col("context")).alias("sourced_snippet"))
            )
        else:
            mentions_src = mentions_df.with_columns(
                ("unknown||" + pl.col("context")).alias("sourced_snippet")
            )

        ent_grouped = (
            mentions_src
            .group_by("entity")
            .agg(
                pl.len().alias("Mention_Count"),
                pl.col("label").filter(pl.col("label") != "Unknown").drop_nulls().first().alias("Type"),
                pl.col("sentiment").last().round(3).alias("Current_Sentiment"),
                pl.col("sentiment").mean().round(3).alias("Average_Sentiment"),
                pl.col("context").unique().alias("SnippetsList"),
                pl.col("sourced_snippet").unique().alias("SourcedSnippetsList"),
            )
            .with_columns(
                pl.col("Type").fill_null("Unknown"),
                pl.col("SnippetsList").map_elements(
                    lambda lst: _clean_snippets(list(lst))[:2], return_dtype=pl.List(pl.Utf8)
                ),
                pl.col("SourcedSnippetsList").map_elements(
                    lambda lst: [
                        ss for ss in (list(lst) or [])
                        if "||" in ss
                        and len(ss.split("||",1)[1]) >= SNIPPET_MIN_LENGTH
                        and not any(b in ss.split("||",1)[1].lower() for b in SNIPPET_BLOCKLIST)
                    ][:2],
                    return_dtype=pl.List(pl.Utf8),
                ),
                (pl.col("Current_Sentiment") - pl.col("Average_Sentiment")).abs().round(3).alias("Sentiment_Delta"),
            )
            .with_columns(pl.col("SnippetsList").list.join(" | ").alias("Snippets"))
        )

        final = (
            ent_grouped
            .join(rels_grouped, on="entity", how="left")
            .rename({"entity": "Entity_Name"})
            .with_columns(pl.col("Graph_Power_Score").fill_null(0.0))
            .filter(
                ((pl.col("Mention_Count") >= MIN_MENTION_COUNT) & (pl.col("Graph_Power_Score") >= MIN_POWER_SCORE))
                | (pl.col("Mention_Count") >= HIGH_VOLUME_OVERRIDE)
            )
            .filter(~pl.col("Entity_Name").map_elements(_is_noise_entity, return_dtype=pl.Boolean))
            .sort(["Graph_Power_Score","Mention_Count"], descending=[True,True])
        )
        return final

    def _process_sources(self, articles_df, mentions_df) -> pl.DataFrame:
        if articles_df.is_empty():
            return pl.DataFrame()
        src = (
            articles_df
            .group_by("source")
            .agg(
                pl.len().alias("Total_Articles_Found"),
                pl.col("self_reference").sum().alias("Self_Reference_Count"),
                pl.col("title").unique().alias("Sample_Headlines"),
            )
            .with_columns(
                pl.col("Sample_Headlines").list.head(3).list.join(" | "),
                ((pl.col("Self_Reference_Count") / pl.col("Total_Articles_Found")) * 100)
                .round(1).alias("Self_Reference_Pct"),
            )
        )
        if not mentions_df.is_empty():
            art_sent = mentions_df.group_by("headline_id").agg(pl.col("sentiment").mean().alias("avg"))
            src_sent = (
                articles_df
                .join(art_sent, left_on="id", right_on="headline_id", how="left")
                .group_by("source")
                .agg(pl.col("avg").mean().round(3).alias("Average_Sentiment"))
            )
            src = src.join(src_sent, on="source", how="left")
        else:
            src = src.with_columns(pl.lit(0.0).alias("Average_Sentiment"))

        src = src.rename({"source":"Source_Domain"}).sort("Total_Articles_Found", descending=True)
        src = src.with_columns(
            pl.when(
                pl.col("Source_Domain").is_in(list(SOURCE_SENTIMENT_BLOCKLIST))
                | (pl.col("Self_Reference_Pct") >= HIGH_SELF_REF_EXCLUSION_THRESHOLD)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias("Sentiment_Excluded")
        )
        return src

    # ------------------------------------------------------------------
    # CLUSTER DATA LOADER
    # ------------------------------------------------------------------
    def _load_cluster_data(self) -> tuple[pl.DataFrame, list]:
        topics_csv = DATA_DIR / "cluster_topics.csv"
        narr_json  = DATA_DIR / "cluster_narratives.json"

        topics_df = pl.DataFrame()
        narratives = []

        if topics_csv.exists():
            try:
                topics_df = pl.read_csv(topics_csv)
            except Exception as e:
                print(f"⚠️  Could not read cluster topics CSV: {e}")

        if narr_json.exists():
            try:
                with open(narr_json) as f:
                    data = json.load(f)
                narratives = data.get("clusters", [])
            except Exception as e:
                print(f"⚠️  Could not read cluster narratives JSON: {e}")

        return topics_df, narratives

    # ------------------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------------------
    def _build_executive_summary(self, final_entities, articles_df, mentions_df, final_sources, topics_df) -> dict:
        summary: dict = {}

        if not final_entities.is_empty():
            top = final_entities.head(1).to_dicts()[0]
            summary["dominant_entity"] = {
                "name": top["Entity_Name"], "type": top["Type"],
                "mentions": top["Mention_Count"], "power": top["Graph_Power_Score"],
                "sentiment": top["Average_Sentiment"],
            }
            summary["crisis_entities"]   = final_entities.filter(
                pl.col("Average_Sentiment") <= SENTIMENT_CRISIS_THRESHOLD
            ).head(8).select(["Entity_Name","Type","Average_Sentiment","Mention_Count"]).to_dicts()

            summary["positive_entities"] = final_entities.filter(
                pl.col("Average_Sentiment") >= SENTIMENT_POSITIVE_THRESHOLD
            ).head(5).select(["Entity_Name","Type","Average_Sentiment","Mention_Count"]).to_dicts()

            summary["story_hubs"] = (
                final_entities.sort("Graph_Power_Score", descending=True)
                .head(5)
                .select(["Entity_Name","Type","Graph_Power_Score","Relations"])
                .to_dicts()
            )

            if "Sentiment_Delta" in final_entities.columns:
                summary["controversy_entities"] = (
                    final_entities.sort("Sentiment_Delta", descending=True)
                    .head(6)
                    .select(["Entity_Name","Type","Current_Sentiment","Average_Sentiment","Sentiment_Delta","Mention_Count"])
                    .to_dicts()
                )

            pairs, seen = [], set()
            for ent in final_entities.head(30).to_dicts():
                for rel in (ent.get("RelationsList") or [])[:5]:
                    key = tuple(sorted([ent["Entity_Name"], rel.split(" (")[0]]))
                    if key not in seen:
                        seen.add(key)
                        pairs.append({"entity_a": key[0], "entity_b": key[1]})
            summary["top_pairs"] = pairs[:10]

        summary["total_articles"] = len(articles_df)
        summary["total_entities"] = len(final_entities) if not final_entities.is_empty() else 0
        summary["source_count"]   = articles_df["source"].n_unique() if not articles_df.is_empty() else 0

        if not final_sources.is_empty():
            valid_src = final_sources.filter(~pl.col("Sentiment_Excluded"))
            if not valid_src.is_empty():
                row = valid_src.sort("Average_Sentiment").head(1).to_dicts()[0]
                summary["most_negative_source"] = row

        if not final_entities.is_empty():
            avg = final_entities["Average_Sentiment"].mean()
            summary["global_avg_sentiment"] = round(float(avg), 3) if avg is not None else 0.0

        if not topics_df.is_empty():
            summary["top_clusters"] = topics_df.head(5).to_dicts()
            summary["cluster_count"] = len(topics_df)

        return summary

    # ------------------------------------------------------------------
    # CHART BUILDER
    # ------------------------------------------------------------------
    def _build_chart_images(self, final_entities, articles_df, mentions_df, edges_df, period) -> dict:
        charts = {
            "sentiment_timeline":   None,
            "top_sources":          None,
            "entity_types":         None,
            "entity_sentiment_dist":None,
            "relationship_network": None,
            "source_scatter":       None,
            "controversy_heatmap":  None,
        }

        def _save(fig) -> io.BytesIO:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)
            return buf

        # ── 1. Sentiment timeline ────────────────────────────────────────
        if not mentions_df.is_empty() and not articles_df.is_empty():
            try:
                merged = mentions_df.join(articles_df, left_on="headline_id", right_on="id")
                if period == "daily":
                    merged = merged.with_columns(pl.col("published").str.slice(0,13).alias("ds"))
                    title  = "Global Sentiment — Last 24 Hours (by hour)"
                else:
                    merged = merged.with_columns(pl.col("published").str.slice(0,10).alias("ds"))
                    title  = "Global Sentiment Over Time (by day)"

                ts = merged.group_by("ds").agg(pl.col("sentiment").mean().alias("avg")).sort("ds")
                if len(ts) > 1:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    dates = ts["ds"].to_list()
                    sents = ts["avg"].to_list()
                    ax.plot(dates, sents, marker="o", color="#2c3e50", linewidth=2)
                    ax.fill_between(dates, sents, 0, where=[s>=0 for s in sents], color="#2ecc71", alpha=0.3, interpolate=True)
                    ax.fill_between(dates, sents, 0, where=[s<0  for s in sents], color="#e74c3c", alpha=0.3, interpolate=True)
                    ax.axhline(0, color="black", linewidth=1, linestyle="--")
                    ax.set_title(title, fontsize=12, fontweight="bold")
                    ax.set_ylabel("Sentiment Score", fontsize=10)
                    step   = max(1, len(dates)//10)
                    labels = [d[-2:]+":00" for d in dates] if period=="daily" else dates
                    ax.set_xticks(range(0, len(dates), step))
                    ax.set_xticklabels([labels[i] for i in range(0, len(dates), step)], rotation=45, ha="right")
                    ax.grid(True, linestyle=":", alpha=0.5)
                    fig.tight_layout()
                    charts["sentiment_timeline"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Sentiment timeline chart: {e}")
                plt.close("all")

        # ── 2. Top sources bar ───────────────────────────────────────────
        if not articles_df.is_empty():
            try:
                src = articles_df.group_by("source").agg(pl.len().alias("n")).sort("n", descending=True).head(8)
                fig, ax = plt.subplots(figsize=(4.5, 3.5))
                srcs = src["source"].to_list()[::-1]
                vals = src["n"].to_list()[::-1]
                bars = ax.barh(srcs, vals, color="#34495e", height=0.6)
                ax.bar_label(bars, padding=3, fontsize=8, color="#444")
                ax.set_title("Top Sources", fontsize=11, fontweight="bold")
                ax.set_xlabel("Articles", fontsize=9)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                charts["top_sources"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Top sources chart: {e}")
                plt.close("all")

        # ── 3. Entity type pie ───────────────────────────────────────────
        if not final_entities.is_empty():
            try:
                tc = final_entities.group_by("Type").agg(pl.len().alias("n")).sort("n", descending=True).head(6)
                fig, ax = plt.subplots(figsize=(4, 3.5))
                ax.pie(tc["n"].to_list(), labels=tc["Type"].to_list(),
                       autopct="%1.1f%%", startangle=140,
                       colors=plt.cm.Set3.colors,
                       wedgeprops={"linewidth":1,"edgecolor":"white"})
                ax.set_title("Entity Type Mix", fontsize=11, fontweight="bold")
                fig.tight_layout()
                charts["entity_types"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Entity type chart: {e}")
                plt.close("all")

        # ── 4. Entity sentiment distribution histogram ───────────────────
        if not final_entities.is_empty() and "Average_Sentiment" in final_entities.columns:
            try:
                sents = final_entities["Average_Sentiment"].to_list()
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(sents, bins=20, color="#2c3e50", edgecolor="white", alpha=0.85)
                ax.axvline(0, color="#e74c3c", linewidth=1.5, linestyle="--", label="Neutral")
                ax.axvline(SENTIMENT_CRISIS_THRESHOLD,  color="#c0392b", linewidth=1, linestyle=":", label="Crisis")
                ax.axvline(SENTIMENT_POSITIVE_THRESHOLD, color="#27ae60", linewidth=1, linestyle=":", label="Positive")
                ax.set_title("Entity Sentiment Distribution", fontsize=11, fontweight="bold")
                ax.set_xlabel("Avg Sentiment Score", fontsize=9)
                ax.set_ylabel("Entity Count", fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(True, linestyle=":", alpha=0.4)
                fig.tight_layout()
                charts["entity_sentiment_dist"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Sentiment dist chart: {e}")
                plt.close("all")

        # ── 5. Relationship network graph ────────────────────────────────
        if not edges_df.is_empty() and not final_entities.is_empty():
            try:
                top_ents = set(final_entities.head(20)["Entity_Name"].to_list())
                sub_edges = edges_df.filter(
                    pl.col("source").is_in(top_ents) & pl.col("target").is_in(top_ents)
                )
                if len(sub_edges) > 0:
                    G = nx.Graph()
                    for e in sub_edges.to_dicts():
                        w = 2.0 if e.get("relation","co-occurrence") != "co-occurrence" else 0.5
                        if G.has_edge(e["source"], e["target"]):
                            G[e["source"]][e["target"]]["weight"] += w
                        else:
                            G.add_edge(e["source"], e["target"], weight=w)

                    pr = nx.pagerank(G, weight="weight")
                    node_sizes  = [pr.get(n, 0.01) * 8000 for n in G.nodes()]
                    edge_widths = [min(G[u][v].get("weight",1)*0.4, 4) for u,v in G.edges()]

                    type_map = dict(zip(
                        final_entities["Entity_Name"].to_list(),
                        final_entities["Type"].to_list()
                    ))
                    palette = {"Person":"#3498db","Organization":"#e67e22","Location":"#2ecc71","Event":"#9b59b6","Concept":"#95a5a6","Unknown":"#bdc3c7"}
                    node_colors = [palette.get(type_map.get(n,"Unknown"),"#bdc3c7") for n in G.nodes()]

                    fig, ax = plt.subplots(figsize=(8, 6))
                    pos = nx.spring_layout(G, seed=42, k=1.5/max(len(G.nodes())**0.5, 1))
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.85, ax=ax)
                    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.4, edge_color="#888", ax=ax)
                    nx.draw_networkx_labels(G, pos, font_size=7, font_color="#222", ax=ax)

                    for ltype, lcolor in palette.items():
                        if any(type_map.get(n) == ltype for n in G.nodes()):
                            ax.scatter([], [], c=lcolor, label=ltype, s=60)
                    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
                    ax.set_title("Entity Relationship Network (Top 20)", fontsize=12, fontweight="bold")
                    ax.axis("off")
                    fig.tight_layout()
                    charts["relationship_network"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Network graph chart: {e}")
                plt.close("all")

        # ── 6. Source scatter ───────────────────────────────────────────
        if not articles_df.is_empty():
            try:
                src_data = (
                    articles_df
                    .group_by("source")
                    .agg(
                        pl.len().alias("count"),
                        pl.col("self_reference").mean().alias("self_ref_rate"),
                    )
                    .sort("count", descending=True)
                    .head(15)
                )
                if len(src_data) >= 3:
                    fig, ax = plt.subplots(figsize=(5.5, 4))
                    x = src_data["count"].to_list()
                    y = [v*100 for v in src_data["self_ref_rate"].to_list()]
                    labels = src_data["source"].to_list()
                    scatter_c = ["#c0392b" if yv >= HIGH_SELF_REF_EXCLUSION_THRESHOLD else "#2c3e50" for yv in y]
                    ax.scatter(x, y, c=scatter_c, s=60, alpha=0.8)
                    ax.axhline(HIGH_SELF_REF_EXCLUSION_THRESHOLD, color="#c0392b", linewidth=1, linestyle="--", label="Exclusion threshold")
                    for xi, yi, lab in zip(x, y, labels):
                        ax.annotate(lab, (xi, yi), fontsize=6.5, xytext=(4,4), textcoords="offset points", color="#444")
                    ax.set_xlabel("Article Volume", fontsize=9)
                    ax.set_ylabel("Self-Reference Rate (%)", fontsize=9)
                    ax.set_title("Source Independence Analysis", fontsize=11, fontweight="bold")
                    ax.legend(fontsize=8)
                    ax.grid(True, linestyle=":", alpha=0.4)
                    fig.tight_layout()
                    charts["source_scatter"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Source scatter chart: {e}")
                plt.close("all")

        # ── 7. Controversy heatmap table ─────────────────────────────────
        if not final_entities.is_empty() and "Sentiment_Delta" in final_entities.columns:
            try:
                top_delta = (
                    final_entities
                    .filter(pl.col("Mention_Count") >= MIN_MENTION_COUNT)
                    .sort("Sentiment_Delta", descending=True)
                    .head(12)
                )
                if len(top_delta) >= 3:
                    names   = top_delta["Entity_Name"].to_list()
                    deltas  = top_delta["Sentiment_Delta"].to_list()
                    cur_s   = top_delta["Current_Sentiment"].to_list()
                    avg_s   = top_delta["Average_Sentiment"].to_list()

                    fig, ax = plt.subplots(figsize=(6, max(2, len(names)*0.4)))
                    norm_deltas = [d/max(deltas) if max(deltas)>0 else 0 for d in deltas]
                    bar_colors  = [
                        "#c0392b" if c < a else "#27ae60"
                        for c, a in zip(cur_s, avg_s)
                    ]
                    bars = ax.barh(names[::-1], [d for d in deltas[::-1]], color=bar_colors[::-1], height=0.6)
                    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
                    ax.set_title("Sentiment Controversy — Largest Δ (Current vs Average)", fontsize=10, fontweight="bold")
                    ax.set_xlabel("|Current − Average Sentiment|", fontsize=9)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(True, linestyle=":", alpha=0.4, axis="x")
                    fig.tight_layout()
                    charts["controversy_heatmap"] = _save(fig)
            except Exception as e:
                print(f"⚠️  Controversy chart: {e}")
                plt.close("all")

        return charts

    # ------------------------------------------------------------------
    # CSV EXPORT
    # ------------------------------------------------------------------
    def _export_csvs(self, final_entities, final_sources, period):
        ents_path = DATA_DIR / f"entities_intelligence_{period}_ai.csv"
        src_path  = DATA_DIR / f"sources_intelligence_{period}_ai.csv"

        if not final_entities.is_empty():
            if ents_path.exists():
                try:
                    notes = pl.read_csv(ents_path).select(["Entity_Name","User_Notes"])
                    final_entities = final_entities.join(notes, on="Entity_Name", how="left").with_columns(pl.col("User_Notes").fill_null(""))
                except Exception:
                    final_entities = final_entities.with_columns(pl.lit("").alias("User_Notes"))
            else:
                final_entities = final_entities.with_columns(pl.lit("").alias("User_Notes"))

            final_entities.drop(["SnippetsList","RelationsList","SourcedSnippetsList"], strict=False).write_csv(ents_path)

        if not final_sources.is_empty():
            final_sources.with_columns(pl.lit("").alias("User_Notes")).write_csv(src_path)

    # ------------------------------------------------------------------
    # PDF REPORT
    # ------------------------------------------------------------------
    def generate_pdf_report(
        self,
        final_entities, articles_df, mentions_df,
        final_sources, edges_df, period,
        chart_images: dict, entity_glosses: dict,
        topics_df: pl.DataFrame, narratives: list,
    ):
        report_pdf = DATA_DIR / f"intelligence_report_{period}.pdf"
        print(f"📑 Generating PDF '{report_pdf.name}'...")
        summary = self._build_executive_summary(
            final_entities, articles_df, mentions_df, final_sources, topics_df
        )

        styles    = getSampleStyleSheet()
        DARK      = "#2c3e50"
        ACCENT    = "#2980b9"
        CRISIS_C  = "#c0392b"
        POS_C     = "#27ae60"

        title_style   = ParagraphStyle("RPTitle",   parent=styles["Heading1"], fontSize=18, spaceAfter=4, textColor=colors.HexColor(DARK))
        sub_style     = ParagraphStyle("RPSub",     parent=styles["Heading2"], fontSize=14, spaceAfter=4, textColor=colors.HexColor(DARK))
        h3_style      = ParagraphStyle("RPH3",      parent=styles["Heading3"], fontSize=11, spaceAfter=3, textColor=colors.HexColor(ACCENT))
        normal_style  = styles["Normal"]
        snippet_style = ParagraphStyle("RPSnippet", parent=styles["Normal"], fontName="Helvetica-Oblique",
                                       textColor=colors.HexColor("#444"), leftIndent=16, spaceAfter=5, fontSize=9)
        label_style   = ParagraphStyle("RPLabel",   parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#888"))
        meta_style    = ParagraphStyle("RPMeta",    parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#aaa"), spaceAfter=3)

        def _hr(): return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd"), spaceAfter=8)
        def _sp(h=0.1): return Spacer(1, h*inch)
        def _add_chart(buf, w=7.0, h=3.0):
            if buf:
                try:
                    buf.seek(0); return RLImage(buf, width=w*inch, height=h*inch)
                except Exception: pass
            return None

        def _page_num(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.HexColor("#aaa"))
            canvas.drawRightString(
                letter[0] - 0.5*inch, 0.4*inch,
                f"Page {doc.page}  |  {period.title()} Intelligence Report"
            )
            canvas.restoreState()

        doc   = SimpleDocTemplate(
            str(report_pdf), pagesize=letter,
            rightMargin=0.75*inch, leftMargin=0.75*inch,
            topMargin=0.75*inch,  bottomMargin=0.75*inch,
        )
        story = []

        # ================================================================
        # COVER / HEADER
        # ================================================================
        story.append(Paragraph(f"News Intelligence Report — {period.title()}", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            meta_style,
        ))
        story.append(Paragraph(
            f"Articles: {summary.get('total_articles',0)}  |  "
            f"Entities Tracked: {summary.get('total_entities',0)}  |  "
            f"Sources: {summary.get('source_count',0)}"
            + (f"  |  Topic Clusters: {summary.get('cluster_count','—')}" if summary.get("cluster_count") else ""),
            normal_style,
        ))
        story.append(_hr())
        story.append(_sp(0.15))

        # ── AI Briefing ─────────────────────────────────────────────────
        ai_text = self.generate_ai_briefing(final_entities)
        if ai_text:
            story.append(Paragraph("AI Intelligence Briefing", sub_style))
            for para in ai_text.split("\n"):
                para = para.strip()
                if para:
                    clean = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", html.escape(para))
                    story.append(Paragraph(clean, normal_style))
                    story.append(_sp(0.08))
            story.append(_hr())
            story.append(_sp(0.1))

        # ── Sentiment timeline ───────────────────────────────────────────
        chart = _add_chart(chart_images.get("sentiment_timeline"), w=7.0, h=2.8)
        if chart:
            story.append(chart)
            story.append(_sp(0.2))

        # ── Side-by-side: sources + entity types ─────────────────────────
        src_img  = _add_chart(chart_images.get("top_sources"),   w=3.3, h=2.8)
        type_img = _add_chart(chart_images.get("entity_types"),  w=3.3, h=2.8)
        if src_img and type_img:
            story.append(Table([[src_img, type_img]]))
            story.append(_sp(0.2))
        elif src_img:
            story.append(src_img); story.append(_sp(0.15))
        elif type_img:
            story.append(type_img); story.append(_sp(0.15))

        # ── Source self-reference scatter ────────────────────────────────
        scatter = _add_chart(chart_images.get("source_scatter"), w=5.5, h=3.8)
        if scatter:
            story.append(Paragraph("Source Independence Analysis", h3_style))
            story.append(scatter)
            story.append(_sp(0.2))

        # ── Source bias table ────────────────────────────────────────────
        if not final_sources.is_empty():
            story.append(Paragraph("Source Sentiment & Self-Reference", sub_style))
            rows = [["Source","Articles","Avg Sentiment","Self-Ref %","Excluded?"]]
            footnote_srcs = []
            for r in final_sources.head(12).to_dicts():
                excl = r.get("Sentiment_Excluded", False)
                sv   = r.get("Average_Sentiment") or 0.0
                sent = "— excl.*" if excl else (
                    f"⚠ {sv:.3f}" if sv <= SENTIMENT_CRISIS_THRESHOLD
                    else (f"✓ {sv:.3f}" if sv >= SENTIMENT_POSITIVE_THRESHOLD else f"{sv:.3f}")
                )
                if excl:
                    footnote_srcs.append(r.get("Source_Domain",""))
                rows.append([
                    r.get("Source_Domain",""),
                    str(r.get("Total_Articles_Found",0)),
                    sent,
                    f"{r.get('Self_Reference_Pct',0.0):.1f}%",
                    "✗" if excl else "✓",
                ])
            tbl = Table(rows, colWidths=[2.6*inch, 0.9*inch, 1.2*inch, 0.9*inch, 0.8*inch])
            tbl.setStyle(TableStyle(_tbl_style()))
            story.append(tbl)
            if footnote_srcs:
                story.append(Paragraph(
                    f"* Sentiment excluded for: {', '.join(footnote_srcs)}.",
                    ParagraphStyle("FN", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#aaa"), spaceAfter=4),
                ))
            story.append(_sp(0.3))

        # ================================================================
        # PAGE 2: EXECUTIVE SUMMARY
        # ================================================================
        story.append(PageBreak())
        story.append(Paragraph("Executive Summary", title_style))
        story.append(_hr())
        story.append(_sp(0.1))

        global_sent = summary.get("global_avg_sentiment", 0.0)
        sent_word   = "negative" if global_sent < -0.1 else ("positive" if global_sent > 0.1 else "neutral")
        story.append(Paragraph(
            f"Global sentiment for this period is <b>{sent_word}</b> "
            f"(avg: {global_sent:.3f}) across {summary.get('total_articles',0)} articles "
            f"from {summary.get('source_count',0)} sources.",
            normal_style,
        ))
        story.append(_sp(0.1))

        dom = summary.get("dominant_entity")
        if dom:
            story.append(Paragraph(
                f"<b>Most prominent entity:</b> {html.escape(str(dom['name']))} ({dom['type']}) — "
                f"{dom['mentions']} mentions, power {dom['power']:.2f}, sentiment {dom['sentiment']:+.3f}.",
                normal_style,
            ))
        story.append(_sp(0.15))

        hubs = summary.get("story_hubs",[])
        if hubs:
            story.append(Paragraph("Key Story Hubs (Highest Graph Centrality)", h3_style))
            hub_rows = [["Entity","Type","Power Score","Connected To"]]
            for h in hubs:
                hub_rows.append([h.get("Entity_Name",""), h.get("Type",""),
                                  f"{h.get('Graph_Power_Score',0.0):.2f}",
                                  (h.get("Relations") or "")[:60]])
            hub_tbl = Table(hub_rows, colWidths=[1.7*inch, 1.1*inch, 1.0*inch, 3.2*inch])
            hub_tbl.setStyle(TableStyle(_tbl_style("#34495e")))
            story.append(hub_tbl)
            story.append(_sp(0.2))

        crisis = summary.get("crisis_entities",[])
        if crisis:
            story.append(Paragraph("⚠  Sentiment Alerts — Negative Coverage", h3_style))
            cr = [["Entity","Type","Avg Sentiment","Mentions"]]
            for c in crisis:
                cr.append([c.get("Entity_Name",""), c.get("Type",""),
                           f"{c.get('Average_Sentiment',0.0):.3f}", str(c.get("Mention_Count",0))])
            cr_tbl = Table(cr, colWidths=[2.5*inch, 1.3*inch, 1.3*inch, 1.0*inch])
            cr_tbl.setStyle(TableStyle([
                *_tbl_style(CRISIS_C),
                ("TEXTCOLOR",(2,1),(2,-1), colors.HexColor(CRISIS_C)),
                ("FONTNAME", (2,1),(2,-1), "Helvetica-Bold"),
            ]))
            story.append(cr_tbl)
            story.append(_sp(0.15))

        positive = summary.get("positive_entities",[])
        if positive:
            story.append(Paragraph("✓  Positively Covered Entities", h3_style))
            for p in positive:
                story.append(Paragraph(
                    f"<b>{p['Entity_Name']}</b> ({p['Type']}) — "
                    f"avg sentiment <font color='{POS_C}'>{p['Average_Sentiment']:.3f}</font>, "
                    f"{p['Mention_Count']} mentions",
                    normal_style,
                ))
            story.append(_sp(0.15))

        controversy = summary.get("controversy_entities",[])
        if controversy:
            story.append(Paragraph("⚡  Fast-Moving Stories (Largest Sentiment Δ)", h3_style))
            con_rows = [["Entity","Type","Current","Average","Δ","Mentions"]]
            for c in controversy:
                direction = "↑" if c.get("Current_Sentiment",0) > c.get("Average_Sentiment",0) else "↓"
                con_rows.append([
                    c.get("Entity_Name",""), c.get("Type",""),
                    f"{c.get('Current_Sentiment',0.0):+.3f}",
                    f"{c.get('Average_Sentiment',0.0):+.3f}",
                    f"{direction} {c.get('Sentiment_Delta',0.0):.3f}",
                    str(c.get("Mention_Count",0)),
                ])
            con_tbl = Table(con_rows, colWidths=[2.0*inch, 1.0*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch])
            con_tbl.setStyle(TableStyle(_tbl_style("#8e44ad")))
            story.append(con_tbl)
            story.append(_sp(0.15))

        cont_chart = _add_chart(chart_images.get("controversy_heatmap"), w=6.0, h=3.5)
        if cont_chart:
            story.append(cont_chart)
            story.append(_sp(0.2))

        dist_chart = _add_chart(chart_images.get("entity_sentiment_dist"), w=5.0, h=2.8)
        if dist_chart:
            story.append(Paragraph("Entity Sentiment Distribution", h3_style))
            story.append(dist_chart)
            story.append(_sp(0.2))

        # ================================================================
        # PAGE 3: TOPIC CLUSTERS
        # ================================================================
        if not topics_df.is_empty():
            story.append(PageBreak())
            story.append(Paragraph("Topic Cluster Intelligence", title_style))
            story.append(_hr())
            story.append(Paragraph(
                f"{len(topics_df)} topic clusters detected. "
                "Ranked by article volume × entity diversity × sentiment controversy.",
                normal_style,
            ))
            story.append(_sp(0.15))

            cluster_rows = [["#","Topic","Articles","Sentiment","Key Entities"]]
            for rank, r in enumerate(topics_df.head(20).to_dicts(), 1):
                avg_s = r.get("avg_sentiment", 0.0)
                tone  = r.get("tone","")
                sent_str = f"{avg_s:+.3f} ({tone})" if tone else f"{avg_s:+.3f}"
                cluster_rows.append([
                    str(rank),
                    r.get("topic_label","")[:55],
                    str(r.get("article_count",0)),
                    sent_str,
                    r.get("top_entities","")[:60],
                ])
            c_tbl = Table(cluster_rows, colWidths=[0.35*inch, 2.5*inch, 0.65*inch, 1.3*inch, 2.2*inch])
            style_cmds = _tbl_style("#1a5276")

            for row_i, r in enumerate(topics_df.head(20).to_dicts(), 1):
                avg_s = r.get("avg_sentiment", 0.0)
                if avg_s <= SENTIMENT_CRISIS_THRESHOLD:
                    style_cmds.append(("TEXTCOLOR",(3,row_i),(3,row_i), colors.HexColor(CRISIS_C)))
                elif avg_s >= SENTIMENT_POSITIVE_THRESHOLD:
                    style_cmds.append(("TEXTCOLOR",(3,row_i),(3,row_i), colors.HexColor(POS_C)))

            c_tbl.setStyle(TableStyle(style_cmds))
            story.append(c_tbl)
            story.append(_sp(0.25))

            if narratives:
                story.append(Paragraph("Top Cluster Summaries", h3_style))
                for narr in narratives[:5]:
                    nav = narr.get("narrative", {})
                    avg_s = narr.get("avg_sentiment", 0.0)
                    sent_col = CRISIS_C if avg_s <= SENTIMENT_CRISIS_THRESHOLD else (POS_C if avg_s >= SENTIMENT_POSITIVE_THRESHOLD else "#555")
                    story.append(Paragraph(
                        f"<b>{html.escape(nav.get('topic_label',''))}</b>  |  "
                        f"{narr.get('article_count',0)} articles  |  "
                        f"Sentiment: <font color='{sent_col}'>{avg_s:+.3f}</font>",
                        normal_style,
                    ))
                    summary_text = nav.get("summary","")
                    if summary_text:
                        story.append(Paragraph(html.escape(summary_text)[:400], snippet_style))
                    watch = nav.get("watch","")
                    if watch:
                        story.append(Paragraph(f"<i>{html.escape(watch)}</i>", label_style))
                    story.append(_sp(0.12))

        # ================================================================
        # PAGE 4: RELATIONSHIP NETWORK
        # ================================================================
        net_chart = _add_chart(chart_images.get("relationship_network"), w=7.0, h=5.5)
        if net_chart:
            story.append(PageBreak())
            story.append(Paragraph("Entity Relationship Network", title_style))
            story.append(_hr())
            story.append(Paragraph(
                "Node size = PageRank centrality. Edge thickness = relation weight (SVO > co-occurrence). "
                "Node colour = entity type.",
                meta_style,
            ))
            story.append(_sp(0.1))
            story.append(net_chart)
            story.append(_sp(0.2))

            pairs = summary.get("top_pairs",[])
            if pairs:
                story.append(Paragraph("Top Entity Co-occurrence Pairs", h3_style))
                pr = [["Entity A","Entity B"]]
                for p in pairs[:10]: pr.append([p.get("entity_a",""), p.get("entity_b","")])
                pr_tbl = Table(pr, colWidths=[3.5*inch, 3.5*inch])
                pr_tbl.setStyle(TableStyle(_tbl_style("#7f8c8d")))
                story.append(pr_tbl)
                story.append(_sp(0.2))

        # ================================================================
        # PAGE 5+: ENTITY DEEP DIVE
        # ================================================================
        if not final_entities.is_empty():
            story.append(PageBreak())
            story.append(Paragraph("Entity Deep-Dive by Category", title_style))
            story.append(_hr())
            story.append(_sp(0.1))

            unique_types = (
                final_entities.select("Type").drop_nulls().unique().to_series().to_list()
            )
            for e_type in sorted(unique_types):
                type_ents = final_entities.filter(pl.col("Type") == e_type).head(10).to_dicts()
                if not type_ents: continue

                story.append(Paragraph(f"Top {e_type}s", sub_style))
                story.append(_sp(0.05))

                for ent in type_ents:
                    raw_name   = str(ent.get("Entity_Name","Unknown"))
                    safe_name  = html.escape(raw_name)
                    score      = ent.get("Graph_Power_Score", 0.0)
                    cur_sent   = ent.get("Current_Sentiment", 0.0)
                    avg_sent   = ent.get("Average_Sentiment", 0.0)
                    delta      = ent.get("Sentiment_Delta", abs(cur_sent - avg_sent))
                    mentions   = ent.get("Mention_Count", 0)
                    trend      = _sentiment_arrow(cur_sent, avg_sent)
                    sent_hex   = _sentiment_colour(avg_sent).hexval()

                    story.append(Paragraph(
                        f"<b>{safe_name}</b>  |  Power: {score:.1f}  |  "
                        f"Sentiment: <font color='#{sent_hex}'>{avg_sent:+.3f}</font>  |  "
                        f"{trend}  |  Δ {delta:.3f}  |  Mentions: {mentions}",
                        normal_style,
                    ))

                    gloss = entity_glosses.get(raw_name,"")
                    if gloss:
                        story.append(Paragraph(
                            f"<i>{html.escape(str(gloss))}</i>",
                            ParagraphStyle("Gloss", parent=styles["Normal"], fontSize=9,
                                           textColor=colors.HexColor("#2c3e50"),
                                           leftIndent=10, spaceAfter=3, spaceBefore=1),
                        ))

                    rels = ent.get("Relations","")
                    if rels:
                        story.append(Paragraph(f"<b>Connected to:</b> {html.escape(str(rels))}", label_style))

                    sourced  = ent.get("SourcedSnippetsList") or []
                    rendered = 0
                    for ss in list(sourced)[:2]:
                        if "||" not in str(ss): continue
                        src_d, snip_t = str(ss).split("||",1)
                        story.append(Paragraph(
                            f'<font color="#888" size="8">[{html.escape(src_d.strip())}]</font> '
                            f'"{html.escape(snip_t.strip())}"',
                            snippet_style,
                        ))
                        rendered += 1
                    if rendered == 0:
                        for snip in _clean_snippets(list(ent.get("SnippetsList") or []))[:2]:
                            story.append(Paragraph(f'"{html.escape(str(snip))}"', snippet_style))

                    story.append(_sp(0.12))
                story.append(_sp(0.2))

        # ── BUILD ────────────────────────────────────────────────────────
        try:
            if story:
                doc.build(story, onFirstPage=_page_num, onLaterPages=_page_num)
                print(f"📄 PDF saved: {report_pdf}")
            else:
                print("⚠️  No data — PDF not generated.")
        except Exception as e:
            print(f"⚠️  PDF build error: {e}")

    # ------------------------------------------------------------------
    # ORCHESTRATOR
    # ------------------------------------------------------------------
    def generate_all(self):
        if not self.db:
            return

        print("📊 Loading data from LanceDB...")
        articles_df, mentions_df, edges_df = self._load_tables()

        print("⚙️  Normalising entities...")
        mentions_df = mentions_df.with_columns(
            pl.col("entity").map_elements(_normalise_entity_name, return_dtype=pl.Utf8),
            pl.col("label").map_elements(_normalise_label, return_dtype=pl.Utf8),
        )
        if not edges_df.is_empty():
            edges_df = edges_df.with_columns(
                pl.col("source").map_elements(_normalise_entity_name, return_dtype=pl.Utf8),
                pl.col("target").map_elements(_normalise_entity_name, return_dtype=pl.Utf8),
            )

        topics_df, narratives = self._load_cluster_data()

        for period in ["daily"]:
            print(f"\n{'='*60}\n🚀 GENERATING {period.upper()} REPORT\n{'='*60}")
            p_arts, p_ments, p_edges = self._filter_by_period(articles_df, mentions_df, edges_df, period)
            if p_arts.is_empty():
                print(f"⚠️  No articles for '{period}'. Skipping.")
                continue

            final_entities = self._process_entities(p_ments, p_edges, p_arts)
            final_sources  = self._process_sources(p_arts, p_ments)

            self._export_csvs(final_entities, final_sources, period)

            print("✍️   Generating entity editorial glosses...")
            entity_glosses = self._generate_entity_glosses(final_entities)

            print("📈  Rendering charts...")
            chart_images = self._build_chart_images(final_entities, p_arts, p_ments, p_edges, period)

            self.generate_pdf_report(
                final_entities, p_arts, p_ments,
                final_sources, p_edges, period,
                chart_images, entity_glosses,
                topics_df, narratives,
            )

        print(f"\n✅ All reports generated in '{DATA_DIR}'!")


if __name__ == "__main__":
    NewsIntelligenceEngine().generate_all()
