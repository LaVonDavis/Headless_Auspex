

import os

# ---------------------------------------------------------------------------
# STRICT OFFLINE ENFORCEMENT & TELEMETRY BLOCKING
# Must be declared BEFORE importing machine learning libraries.
# ---------------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_TELEMETRY"] = "1"

import json
import re
import time
import warnings

import lancedb
import networkx as nx
import polars as pl
import yake
from flashtext import KeywordProcessor
from gliner import GLiNER
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import shared_news_func_2

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# ENTITY RESOLVER — O(1) canonical lookup with inverted index
# ---------------------------------------------------------------------------
class FastEntityResolver:
    def __init__(self):
        self.map_file = shared_news_func_2.ENTITY_MAP_FILE
        self.mapping: dict[str, dict] = self._load_mapping()
        # inverted index: canonical_lower → canonical (original-case)
        self._canonical_index: dict[str, str] = {
            v["canonical"].lower(): v["canonical"]
            for v in self.mapping.values()
        }

    def _load_mapping(self) -> dict:
        if self.map_file.exists():
            try:
                with open(self.map_file) as f:
                    data = json.load(f)
                if data:
                    # AUTO-PURGE: Remove corrupted pronouns from existing cache
                    bad_keys = {"this", "that", "these", "those", "which", "what", "some", "such", "all", "any", "it", "they", "them", "he", "she", "who", "whom"}
                    cleaned_data = {}
                    for k, v in data.items():
                        canon_val = v if isinstance(v, str) else v.get("canonical", "")
                        if k.lower() not in bad_keys and canon_val.lower() not in bad_keys:
                            cleaned_data[k] = v
                    data = cleaned_data

                    first = next(iter(data.values())) if data else None
                    if isinstance(first, str):
                        return {k: {"canonical": v, "label": "Unknown"} for k, v in data.items()}
                return data
            except Exception:
                return {}
        return {}

    def save(self):
        try:
            with open(self.map_file, "w") as f:
                json.dump(self.mapping, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save entity resolver map: {e}")

    def resolve(self, name: str, label: str = "Unknown"):
        name = name.strip()
        if len(name) < 2:
            return None, None

        # Exact hit
        if name in self.mapping:
            entry = self.mapping[name]
            if entry["label"] in ("Unknown", "Concept") and label not in ("Unknown", "Concept", None):
                entry["label"] = label
            return entry["canonical"], entry["label"]

        name_lower = name.lower()

        # Fast O(1) exact-lower hit
        if name_lower in self._canonical_index:
            canonical = self._canonical_index[name_lower]
            existing_label = self._label_for_canonical(canonical, label)
            self.mapping[name] = {"canonical": canonical, "label": existing_label}
            return canonical, existing_label

        # Fuzzy fallback
        name_words = set(name_lower.split())
        best_match, best_score = None, 0
        for canon_lower, canonical in self._canonical_index.items():
            canon_words = set(canon_lower.split())
            score = fuzz.ratio(name_lower, canon_lower)
            if name_words.issubset(canon_words) or canon_words.issubset(name_words):
                if min(len(name_lower), len(canon_lower)) > 4:
                    score = 100
            if score > best_score:
                best_score, best_match = score, canonical

        if best_score > 85 and best_match:
            existing_label = self._label_for_canonical(best_match, label)
            self.mapping[name] = {"canonical": best_match, "label": existing_label}
            return best_match, existing_label

        # New entity
        self.mapping[name] = {"canonical": name, "label": label}
        self._canonical_index[name_lower] = name
        return name, label

    def _label_for_canonical(self, canonical: str, preferred_label: str) -> str:
        for v in self.mapping.values():
            if v["canonical"] == canonical:
                existing = v["label"]
                if existing in ("Unknown", "Concept") and preferred_label not in ("Unknown", "Concept", None):
                    return preferred_label
                return existing
        return preferred_label or "Unknown"


# ---------------------------------------------------------------------------
# ANALYSIS SERVICE
# ---------------------------------------------------------------------------
class AnalysisService:
    def __init__(self):
        self.db = lancedb.connect(shared_news_func_2.LANCE_DB_PATH)
        self.nlp = None

    # ------------------------------------------------------------------
    # DB HELPERS
    # ------------------------------------------------------------------
    def get_pending_articles(self) -> list[dict]:
        if "raw_articles" not in self.db.table_names():
            return []
        return (
            self.db.open_table("raw_articles")
            .to_polars()
            .filter(pl.col("status") == "pending")
            .collect()
            .to_dicts()
        )

    def mark_as_processed(self):
        self.db.open_table("raw_articles").update(
            where="status = 'pending'", values={"status": "processed"}
        )

    # ------------------------------------------------------------------
    # TEXT / MODELS
    # ------------------------------------------------------------------
    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"(?i)^[-=\s]*Chunk Separator[-=\s]*$", "", text, flags=re.MULTILINE)
        return shared_news_func_2.clean_text_robust(text)

    def load_models(self):
        print("🧠 Loading AI Models from local disk...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
        self.gliner = GLiNER.from_pretrained("urchade/gliner_medium-v2.1", local_files_only=True)
        
        self.vader = SentimentIntensityAnalyzer()
        self.keyword_processor = KeywordProcessor()
        self.yake_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=3, features=None)

        import spacy
        self.nlp = spacy.load("en_core_web_sm", disable=["textcat", "ner"])

        self.resolver = FastEntityResolver()
        self.canonical_labels: dict[str, str] = {}
        for name, data in self.resolver.mapping.items():
            canonical = data["canonical"]
            self.keyword_processor.add_keyword(name, canonical)
            self.canonical_labels[canonical] = data["label"]

    # ------------------------------------------------------------------
    # NER HYBRID  (per-article)
    # ------------------------------------------------------------------
    def process_ner_hybrid(self, text: str, doc, gliner_preds: list[dict]) -> list[dict]:
        found: list[dict] = []
        seen_names: set[str] = set()

        def _add(name, label, score, method):
            if name.lower() not in seen_names:
                seen_names.add(name.lower())
                found.append({"name": name, "label": label, "score": score, "method": method})

        # 1. FlashText (Highest priority, known exact matches)
        for canonical_name, _s, _e in self.keyword_processor.extract_keywords(text, span_info=True):
            _add(canonical_name, self.canonical_labels.get(canonical_name, "Unknown"), 1.0, "fast")

        # 2. GLiNER (High priority AI extraction)
        for pred in gliner_preds:
            name = pred.get("text", "")
            if name:
                _add(name, pred.get("label", "Unknown"), pred.get("score", 0.5), "ai")

        # 3. YAKE keywords (Lower priority concepts)
        for kw, score in self.yake_extractor.extract_keywords(text):
            _add(kw.title(), "Concept", max(0, 1 - score), "yake")

        # 4. Noun chunks (Lowest priority concepts)
        # Added strict pronouns to prevent them from becoming noun chunks
        _ignore = {
            "part", "way", "kind", "type", "sort", "lot", "example",
            "one", "thing", "time", "day", "week", "month", "year",
            "this", "that", "these", "those", "which", "what", "some", "such", "all", "any", "it"
        }
        for chunk in doc.noun_chunks:
            clean = " ".join(t.text for t in chunk if not t.is_stop and not t.is_punct)
            if len(clean) > 3 and clean.lower() not in _ignore:
                _add(clean.title(), "Concept", 0.7, "linguistic")

        return found

    def get_context(self, text: str, entity_name: str) -> str:
        idx = text.lower().find(entity_name.lower())
        if idx == -1:
            return text[:150]
        start, end = max(0, idx - 75), min(len(text), idx + len(entity_name) + 75)
        return f"...{text[start:end].replace(chr(10), ' ').strip()}..."

    def check_self_reference(self, source_domain: str, resolved_entities: set) -> bool:
        src_lower = source_domain.lower()
        ents_lower = {e.lower() for e in resolved_entities}
        for domain_key, aliases in shared_news_func_2.SOURCE_TO_ENTITY_MAP.items():
            if domain_key in src_lower:
                if ents_lower.intersection(a.lower() for a in aliases):
                    return True
        return False

    # ------------------------------------------------------------------
    # MAIN PIPELINE
    # ------------------------------------------------------------------
    def run_analysis(self):
        pending_data = self.get_pending_articles()
        if not pending_data:
            print("💤 No pending articles. Regenerating CSVs...")
            self.export_intelligence_csvs()
            return

        print(f"⚙️  Found {len(pending_data)} pending articles. Starting analysis...")
        t0 = time.time()
        self.load_models()

        # Clean texts up front
        for d in pending_data:
            d["text"] = self.clean_text(d.get("text", ""))
            d.pop("status", None)

        # ── BATCH EMBEDDING ──────────────────────────────────────────────
        texts_for_embed = [f"{r['title']}. {r['text']}" for r in pending_data]
        vectors = self.embedder.encode(texts_for_embed, normalize_embeddings=True, batch_size=64)
        for i, row in enumerate(pending_data):
            row["vector"] = list(vectors[i])

        # ── BATCH SPACY ──────────────────────────────────────────────────
        texts_only = [r["text"] for r in pending_data]
        docs = list(self.nlp.pipe(texts_only, batch_size=32))

        # ── BATCH GLINER ─────────────────────────────────────────────────
        GLINER_LABELS = ["Person", "Organization", "Location", "Event"]
        GLINER_BATCH  = 32
        all_gliner: list[list[dict]] = []
        
        for start in range(0, len(texts_only), GLINER_BATCH):
            batch_texts = texts_only[start : start + GLINER_BATCH]
            if not batch_texts:
                continue

            try:
                batch_preds = self.gliner.batch_predict_entities(
                    batch_texts, GLINER_LABELS, threshold=0.4
                )
                all_gliner.extend(batch_preds)
            except AttributeError:
                batch_preds = [
                    self.gliner.predict_entities(text, GLINER_LABELS, threshold=0.4)
                    for text in batch_texts
                ]
                all_gliner.extend(batch_preds)

        # ── PER-ARTICLE NER + SVO ────────────────────────────────────────
        mentions_buffer: list[dict] = []
        edges_buffer:   list[dict] = []

        if len(docs) != len(pending_data) or len(all_gliner) != len(pending_data):
            print("⚠️ Data arrays out of sync! Halting to prevent data corruption.")
            return

        # SVO guard words
        svo_stop_words = {"this", "that", "these", "those", "which", "what", "some", "such", "it", "they", "them", "he", "she", "all", "any", "who"}

        for i, row in enumerate(pending_data):
            uid, text, source = row["id"], row["text"], row["source"]
            doc = docs[i]

            raw_ents = self.process_ner_hybrid(text, doc, all_gliner[i])

            unique_resolved: set[str] = set()
            for ent in raw_ents:
                resolved_name, resolved_label = self.resolver.resolve(ent["name"], ent["label"])
                if not resolved_name or resolved_name in unique_resolved:
                    continue
                unique_resolved.add(resolved_name)
                snippet = self.get_context(text, ent["name"])
                vader_scores = self.vader.polarity_scores(snippet)
                adj_sentiment = vader_scores["compound"] * (1.0 - vader_scores["neu"] * 0.2)
                mentions_buffer.append({
                    "headline_id": uid,
                    "entity":      resolved_name,
                    "label":       resolved_label,
                    "context":     snippet,
                    "sentiment":   adj_sentiment,
                    "confidence":  ent["score"],
                })

            row["self_reference"] = int(self.check_self_reference(source, unique_resolved))

            # ── SVO GRAPH EXTRACTION ─────────────────────────────────────
            chunk_map = {token.i: chunk.text.title() for chunk in doc.noun_chunks for token in chunk}
            svo_pairs: set[tuple] = set()

            for token in doc:
                if token.pos_ != "VERB":
                    continue
                subj_text = obj_text = None
                for child in token.children:
                    if "subj" in child.dep_ and child.i in chunk_map:
                        subj_text = chunk_map[child.i]
                    if "obj" in child.dep_ and child.i in chunk_map:
                        obj_text = chunk_map[child.i]
                        
                if subj_text and obj_text and subj_text != obj_text:
                    # BLOCKLIST GUARD: Prevent SVO linking of pronouns
                    if subj_text.lower() in svo_stop_words or obj_text.lower() in svo_stop_words:
                        continue
                        
                    s_res, _ = self.resolver.resolve(subj_text)
                    o_res, _ = self.resolver.resolve(obj_text)
                    if s_res and o_res and s_res != o_res:
                        pair = tuple(sorted([s_res, o_res]))
                        svo_pairs.add(pair)
                        edges_buffer.append({
                            "source":      s_res,
                            "target":      o_res,
                            "relation":    token.lemma_,
                            "headline_id": uid,
                        })

            # Co-occurrence fallback
            sorted_ents = sorted(unique_resolved)
            for x in range(len(sorted_ents)):
                for y in range(x + 1, len(sorted_ents)):
                    pair = tuple(sorted([sorted_ents[x], sorted_ents[y]]))
                    if pair not in svo_pairs:
                        edges_buffer.append({
                            "source":      sorted_ents[x],
                            "target":      sorted_ents[y],
                            "relation":    "co-occurrence",
                            "headline_id": uid,
                        })

        self.resolver.save()

        # ── PERSIST TO LANCEDB ───────────────────────────────────────────
        print("💾 Saving analytical data to LanceDB...")
        df_articles = pl.DataFrame(pending_data)

        if "articles" in self.db.table_names():
            self.db.open_table("articles").add(df_articles)
        else:
            tbl = self.db.create_table("articles", data=df_articles)
            if len(tbl) >= 256:
                try:
                    tbl.create_index(metric="cosine", vector_column_name="vector")
                except Exception:
                    pass

        if mentions_buffer:
            df_m = pl.DataFrame(mentions_buffer)
            if "mentions" in self.db.table_names():
                self.db.open_table("mentions").add(df_m)
            else:
                self.db.create_table("mentions", data=df_m)

        if edges_buffer:
            df_e = pl.DataFrame(edges_buffer)
            if "edges" in self.db.table_names():
                self.db.open_table("edges").add(df_e)
            else:
                self.db.create_table("edges", data=df_e)

        self.mark_as_processed()
        self.export_intelligence_csvs()
        print(f"🎉 Analysis complete! {len(pending_data)} articles in {time.time() - t0:.2f}s")

    # ------------------------------------------------------------------
    # INTELLIGENCE CSV EXPORT
    # ------------------------------------------------------------------
    def export_intelligence_csvs(self):
        if "articles" not in self.db.table_names() or "mentions" not in self.db.table_names():
            return

        print("📊 Generating Intelligence CSVs...")
        articles_lf = self.db.open_table("articles").to_polars()
        mentions_lf = self.db.open_table("mentions").to_polars()

        # Articles CSV
        try:
            art_df = articles_lf.collect()
            if "vector" in art_df.columns:
                art_df = art_df.drop("vector")
            art_df.write_csv(shared_news_func_2.ARTICLES_CSV)
        except PermissionError:
            print("⚠️ Articles CSV locked. Close it in Excel to allow updates.")

        mentions_df = mentions_lf.collect()

        edges_df = (
            self.db.open_table("edges").to_polars().collect()
            if "edges" in self.db.table_names()
            else pl.DataFrame({"source": [], "target": [], "relation": []})
        )

        # Graph centrality
        if not edges_df.is_empty():
            print("🕸️  Calculating Graph Centrality...")
            G = nx.Graph()
            for edge in edges_df.to_dicts():
                w = 2.0 if edge.get("relation", "co-occurrence") != "co-occurrence" else 1.0
                G.add_edge(edge["source"], edge["target"], weight=w)
            try:
                pagerank = nx.pagerank(G, weight="weight")
                pr_df = pl.DataFrame({
                    "entity": list(pagerank.keys()),
                    "Graph_Power_Score": [round(v * 100, 2) for v in pagerank.values()],
                })
            except Exception:
                pr_df = pl.DataFrame({"entity": [], "Graph_Power_Score": []})

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
                .agg(pl.col("related").drop_nulls().unique().alias("Relations"))
                .with_columns(pl.col("Relations").list.head(5).list.join(", "))
            )
            if not pr_df.is_empty():
                rels_grouped = rels_grouped.join(pr_df, on="entity", how="left")
        else:
            rels_grouped = pl.DataFrame({"entity": [], "Relations": [], "Graph_Power_Score": []})

        # Entity aggregation
        ent_grouped = (
            mentions_df
            .group_by("entity")
            .agg(
                pl.col("label").first().alias("Type"),
                pl.col("sentiment").last().round(3).alias("Current_Sentiment"),
                pl.col("sentiment").mean().round(3).alias("Average_Sentiment"),
                pl.col("context").unique().alias("Snippets"),
            )
            .with_columns(pl.col("Snippets").list.head(2).list.join(" | "))
        )

        final_entities = (
            ent_grouped
            .join(rels_grouped, on="entity", how="left")
            .rename({"entity": "Entity_Name"})
            .sort("Average_Sentiment")
        )

        if shared_news_func_2.ENTITIES_CSV.exists():
            try:
                notes = pl.read_csv(shared_news_func_2.ENTITIES_CSV).select(["Entity_Name", "User_Notes"])
                final_entities = (
                    final_entities
                    .join(notes, on="Entity_Name", how="left")
                    .with_columns(pl.col("User_Notes").fill_null(""))
                )
            except Exception:
                final_entities = final_entities.with_columns(pl.lit("").alias("User_Notes"))
        else:
            final_entities = final_entities.with_columns(pl.lit("").alias("User_Notes"))

        try:
            final_entities.write_csv(shared_news_func_2.ENTITIES_CSV)
        except PermissionError:
            print("⚠️ Entities CSV locked. Close it in Excel to allow updates.")

        # Sources CSV
        articles_df = articles_lf.collect()
        src_grouped = (
            articles_df
            .group_by("source")
            .agg(
                pl.len().alias("Total_Articles_Found"),
                pl.col("self_reference").sum().alias("Self_Reference_Count"),
                pl.col("title").unique().alias("Sample_Headlines"),
            )
            .with_columns(pl.col("Sample_Headlines").list.head(3).list.join(" | "))
        )
        art_sent = mentions_df.group_by("headline_id").agg(pl.col("sentiment").mean().alias("avg"))
        src_sent = (
            articles_df
            .join(art_sent, left_on="id", right_on="headline_id", how="left")
            .group_by("source")
            .agg(pl.col("avg").mean().round(3).alias("Average_Sentiment"))
        )
        final_sources = (
            src_grouped
            .join(src_sent, on="source", how="left")
            .rename({"source": "Source_Domain"})
            .sort("Total_Articles_Found", descending=True)
        )
        try:
            final_sources.write_csv(shared_news_func_2.SOURCES_CSV)
        except PermissionError:
            print("⚠️ Sources CSV locked. Close it in Excel to allow updates.")

if __name__ == "__main__":
    AnalysisService().run_analysis()
