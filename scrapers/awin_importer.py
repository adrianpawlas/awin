import os
import sys
import gzip
import csv
import time
import json
import re
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import Optional

import torch
import requests
from PIL import Image
from supabase import create_client, Client
from dotenv import load_dotenv
import timm
from transformers import AutoTokenizer, AutoModel


load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

AWIN_DATAFEED_URL = (
    "https://productdata.awin.com/datafeed/download/"
    "apikey/746193b8a6735d6afaff119db9e3bcc7/language/en/"
    "cid/595,147,149,626,135,163,159,161,170,137,171,548,174,183,178,179,175,172,189,194,141,"
    "205,198,206,203,208,199,204,201,627,99,100,101,107,110,111,113,114,115,116,118,121,122,127,"
    "581,624,123,594,125/"
    "bid/63213,50605,65283,65291,65321,51631,51639,64661,64727,65447,64579,51989,65561,63279,"
    "65715,64323,65731,53433,65779,64741,65873,64759,65951,65007,66075,66115,64663,64981,63355,"
    "56141,66517,66603,64843,64991,66743,57573,66795,57757,64619,58035,64339,63407,58789,58999,"
    "67169,64673,63453,68375,63465,64915,67563,61061,64679,64987,64307,67753,64779,64325,64653,"
    "62437,68893,69271,69529,70219,71459,71737,72459,72461,72983,73233,73239,73267,73281,74369,"
    "74417,74481,74511,74591,74639,75115,75699,76245,77801,77999,80929,81949,81955,81963,81995,"
    "82019,82021,82035/"
    "columns/aw_deep_link,product_name,aw_product_id,merchant_product_id,merchant_image_url,"
    "description,merchant_category,search_price,merchant_name,merchant_id,category_name,category_id,"
    "aw_image_url,currency,store_price,delivery_cost,merchant_deep_link,language,last_updated,"
    "display_price,data_feed_id,brand_name,colour,specifications,brand_id,product_short_description,"
    "keywords,saving,savings_percent,large_image,alternate_image_two,alternate_image_four,"
    "alternate_image,Fashion%3Asize,Fashion%3Amaterial,Fashion%3Acategory/"
    "format/csv/delimiter/%2C/compression/gzip/"
)

BATCH_SIZE = 50
BATCH_SLEEP = 0.05
STALE_DELETE_BATCH_SIZE = 100
PROGRESS_EVERY = 10000
EMBED_IMAGE_TIMEOUT = 10
EMBED_THREADS = 8
SIGLIP_MODEL = "vit_base_patch16_siglip_384.webli"

WOMEN_KEYWORDS = ["women", "woman", "female", "ladies", "girl", "womenswear", "femme"]
MEN_KEYWORDS = ["men", "man", "male", "boys", "menswear", "homme"]


class SiglipEmbedder:
    _instance: Optional["SiglipEmbedder"] = None
    _lock = threading.Lock()

    def __init__(self):
        print(f"Loading SigLIP vision encoder from timm: {SIGLIP_MODEL}...")
        self.vision_model = timm.create_model(SIGLIP_MODEL, pretrained=True)
        self.vision_model.eval()
        self.data_config = timm.data.resolve_data_config(self.vision_model.pretrained_cfg)
        self.vision_transforms = timm.data.create_transform(**self.data_config, is_training=False)
        print(f"SigLIP vision encoder loaded.")

        print("Loading BERT text encoder for info embeddings...")
        self.text_model = AutoModel.from_pretrained("google/bert_uncased_L-12_H-768_A-12")
        self.text_tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-12_H-768_A-12")
        self.text_model.eval()
        print("BERT text encoder loaded.")

    @classmethod
    def get_instance(cls) -> "SiglipEmbedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def embed_image_url(self, url: str) -> Optional[list[float]]:
        try:
            resp = requests.get(url, timeout=EMBED_IMAGE_TIMEOUT)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            tensor = self.vision_transforms(img).unsqueeze(0)
            with torch.no_grad():
                emb = self.vision_model(tensor)
            if emb.ndim == 2:
                emb = emb.squeeze(0)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.tolist()
        except Exception:
            return None

    def embed_text(self, text: str) -> Optional[list[float]]:
        try:
            inputs = self.text_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                output = self.text_model(**inputs)
            emb = output.last_hidden_state[:, 0, :]
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            return emb.squeeze().tolist()
        except Exception:
            return None

    def embed_batch_images(self, items: list[tuple[str, str]]) -> dict[str, Optional[list[float]]]:
        results: dict[str, Optional[list[float]]] = {}
        with ThreadPoolExecutor(max_workers=EMBED_THREADS) as executor:
            futures = {executor.submit(self.embed_image_url, url): id_ for id_, url in items}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results


def build_gender_signal(text: str) -> str | None:
    text_lower = text.lower()
    has_women = any(kw in text_lower for kw in WOMEN_KEYWORDS)
    has_men = any(kw in text_lower for kw in MEN_KEYWORDS)
    if has_women and has_men:
        return None
    if has_women:
        return "women"
    if has_men:
        return "men"
    return None


def parse_price(value: str) -> float | None:
    if not value or not value.strip():
        return None
    cleaned = re.sub(r"[^\d.,\-]", "", value.strip())
    if not cleaned:
        return None
    cleaned = cleaned.replace(",", ".")
    try:
        num = float(cleaned)
        if num > 0:
            return num
    except ValueError:
        pass
    return None


def build_info_text(row: dict) -> str:
    parts = []
    pn = row.get("product_name", "").strip()
    if pn:
        parts.append(f"Product: {pn}")
    bn = row.get("brand_name", "").strip()
    mn = row.get("merchant_name", "").strip()
    if bn:
        parts.append(f"Brand: {bn}")
    elif mn:
        parts.append(f"Brand: {mn}")
    desc = row.get("description", "").strip()
    if desc:
        parts.append(f"Description: {desc[:500]}")
    short_desc = row.get("product_short_description", "").strip()
    if short_desc and short_desc not in desc:
        parts.append(f"Short description: {short_desc[:200]}")
    fc = row.get("Fashion:category", "").strip()
    cn = row.get("category_name", "").strip()
    mc = row.get("merchant_category", "").strip()
    cat = fc or cn or mc
    if cat:
        parts.append(f"Category: {cat}")
    size = row.get("Fashion:size", "").strip()
    if size:
        parts.append(f"Size: {size}")
    mat = row.get("Fashion:material", "").strip()
    if mat:
        parts.append(f"Material: {mat}")
    colour = row.get("colour", "").strip()
    if colour:
        parts.append(f"Colour: {colour}")
    kw = row.get("keywords", "").strip()
    if kw:
        parts.append(f"Keywords: {kw}")
    sp = row.get("search_price", "").strip()
    dp = row.get("display_price", "").strip()
    price = dp or sp
    if price:
        parts.append(f"Price: {price}")
    if row.get("savings_percent", "").strip():
        parts.append(f"Discount: {row['savings_percent']}% off")
    return " | ".join(parts)


def map_row(row: dict) -> dict | None:
    product_name = row.get("product_name", "").strip()
    aw_product_id = row.get("aw_product_id", "").strip()
    if not product_name or not aw_product_id:
        return None

    aw_image_url = row.get("aw_image_url", "").strip()
    large_image = row.get("large_image", "").strip()
    merchant_image_url = row.get("merchant_image_url", "").strip()
    image_url = aw_image_url or large_image or merchant_image_url
    if not image_url:
        return None

    display_price = row.get("display_price", "").strip()
    search_price = row.get("search_price", "").strip()
    store_price = row.get("store_price", "").strip()
    if parse_price(display_price) is None and parse_price(search_price) is None and parse_price(store_price) is None:
        return None

    aw_deep_link = row.get("aw_deep_link", "").strip()
    merchant_deep_link = row.get("merchant_deep_link", "").strip()
    if not aw_deep_link and not merchant_deep_link:
        return None

    product_id = f"awin_{aw_product_id}"

    product_url = merchant_deep_link if merchant_deep_link else None
    if not product_url:
        product_url = aw_deep_link

    affiliate_url = aw_deep_link if aw_deep_link else None

    brand_name = row.get("brand_name", "").strip()
    merchant_name = row.get("merchant_name", "").strip()
    brand = brand_name if brand_name else merchant_name

    description_raw = row.get("description", "").strip()
    product_short_description = row.get("product_short_description", "").strip()
    description = description_raw if description_raw else product_short_description
    if description:
        description = description[:2000]

    fashion_category = row.get("Fashion:category", "").strip()
    category_name = row.get("category_name", "").strip()
    merchant_category = row.get("merchant_category", "").strip()
    category = fashion_category if fashion_category else category_name
    if not category:
        category = merchant_category

    gender_text = f"{category_name} {merchant_category} {fashion_category} {row.get('keywords', '')} {product_name}".lower()
    gender = build_gender_signal(gender_text)

    colour = row.get("colour", "").strip()
    saving_str = row.get("saving", "").strip()
    savings_percent_str = row.get("savings_percent", "").strip()
    sale = None
    if savings_percent_str:
        try:
            savings_pct = float(savings_percent_str.replace(",", "."))
            if savings_pct > 0:
                sale = store_price if store_price else None
        except ValueError:
            pass

    keywords_raw = row.get("keywords", "").strip()
    material = row.get("Fashion:material", "").strip()
    tags = []
    if keywords_raw:
        for kw in keywords_raw.split(","):
            kw = kw.strip().lower()
            if kw:
                tags.append(kw)
    if colour:
        colour_lower = colour.lower().strip()
        if colour_lower not in tags:
            tags.append(colour_lower)
    if material:
        material_lower = material.lower().strip()
        if material_lower not in tags:
            tags.append(material_lower)

    merchant_id = row.get("merchant_id", "").strip()
    data_feed_id = row.get("data_feed_id", "").strip()
    language = row.get("language", "").strip()
    last_updated = row.get("last_updated", "").strip()
    specifications = row.get("specifications", "").strip()
    size = row.get("Fashion:size", "").strip()
    price_val = display_price if display_price else (search_price if search_price else store_price)

    metadata = {
        "product_name": product_name,
        "aw_product_id": aw_product_id,
        "merchant_product_id": row.get("merchant_product_id", "").strip(),
        "merchant_id": merchant_id,
        "merchant_name": merchant_name,
        "data_feed_id": data_feed_id,
        "brand_name": brand_name if brand_name else None,
        "brand_id": row.get("brand_id", "").strip(),
        "description": description,
        "product_short_description": product_short_description if product_short_description else None,
        "category_name": category_name if category_name else None,
        "category_id": row.get("category_id", "").strip(),
        "merchant_category": merchant_category if merchant_category else None,
        "fashion_category": fashion_category if fashion_category else None,
        "colour": colour if colour else None,
        "material": material if material else None,
        "size": size if size else None,
        "gender": gender,
        "keywords": keywords_raw if keywords_raw else None,
        "specifications": specifications if specifications else None,
        "price": price_val,
        "search_price": search_price if search_price else None,
        "store_price": store_price if store_price else None,
        "display_price": display_price if display_price else None,
        "currency": row.get("currency", "").strip(),
        "delivery_cost": row.get("delivery_cost", "").strip() or None,
        "saving": saving_str if saving_str else None,
        "savings_percent": savings_percent_str if savings_percent_str else None,
        "sale": sale,
        "tags": tags if tags else None,
        "language": language if language else None,
        "last_updated": last_updated if last_updated else None,
    }

    additional_images = []
    for img_field in ["alternate_image", "alternate_image_two", "alternate_image_three", "alternate_image_four"]:
        img_val = row.get(img_field, "").strip()
        if img_val:
            additional_images.append(img_val)
    additional_images_str = json.dumps(additional_images) if additional_images else None

    other = specifications if specifications else None

    return {
        "id": product_id,
        "source": "awin",
        "product_url": product_url if product_url else None,
        "affiliate_url": affiliate_url,
        "image_url": image_url,
        "brand": brand if brand else None,
        "title": product_name,
        "description": description if description else None,
        "category": category if category else None,
        "gender": gender,
        "search_tsv": None,
        "created_at": None,
        "metadata": json.dumps(metadata),
        "size": size if size else None,
        "second_hand": False,
        "image_embedding": None,
        "country": None,
        "compressed_image_url": None,
        "tags": tags if tags else None,
        "search_vector": None,
        "title_tsv": None,
        "brand_tsv": None,
        "description_tsv": None,
        "other": other,
        "price": price_val,
        "sale": sale,
        "additional_images": additional_images_str,
        "info_embedding": None,
        "_row": row,
    }





def upsert_batch(
    supabase: Client,
    batch: list[dict],
    use_embeddings: bool,
    embedder: Optional[SiglipEmbedder] = None,
) -> int:
    if not batch:
        return 0

    if use_embeddings and embedder:
        image_items = [(p["id"], p["image_url"]) for p in batch]
        image_embs = embedder.embed_batch_images(image_items)

        for p in batch:
            pid = p["id"]
            p["image_embedding"] = image_embs.get(pid)
            row = p.pop("_row")
            info_text = build_info_text(row)
            p["info_embedding"] = embedder.embed_text(info_text)

    cleaned = []
    embed_map: dict[str, dict] = {}
    for p in batch:
        row = p.pop("_row", None)
        pid = p["id"]
        embed_map[pid] = {
            "image_embedding": p.get("image_embedding"),
            "info_embedding": p.get("info_embedding"),
        }
        c = {k: v for k, v in p.items() if k not in ("image_embedding", "info_embedding")}
        if not c.get("product_url"):
            c["product_url"] = None
        if not c.get("affiliate_url"):
            c["affiliate_url"] = None
        cleaned.append(c)

    try:
        supabase.table("products").upsert(cleaned, on_conflict="id").execute()
    except Exception as e:
        print(f"Upsert error: {e}")
        return 0

    if use_embeddings and embedder and embed_map:
        ids = list(embed_map.keys())
        try:
            for pid in ids:
                emb = embed_map[pid]
                update_data = {}
                if emb["image_embedding"] is not None:
                    update_data["image_embedding"] = emb["image_embedding"]
                if emb["info_embedding"] is not None:
                    update_data["info_embedding"] = emb["info_embedding"]
                if update_data:
                    supabase.table("products").update(update_data).eq("id", pid).execute()
        except Exception as e:
            print(f"Embed update error: {e}")

    return len(cleaned)


def run(limit: Optional[int] = None, skip_stale_delete: bool = False, generate_embeddings: bool = False):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    embedder: Optional[SiglipEmbedder] = None
    if generate_embeddings:
        embedder = SiglipEmbedder.get_instance()
        print("Embedding generation enabled.")

    print("Starting Awin datafeed import...")
    print(f"Fetching: {AWIN_DATAFEED_URL}")
    if generate_embeddings:
        print("Embedding generation: ENABLED (SigLIP)")
    else:
        print("Embedding generation: DISABLED (use --embed to enable)")

    response = requests.get(AWIN_DATAFEED_URL, stream=True, timeout=120)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch datafeed: HTTP {response.status_code}")

    response.raw.decode_content = True

    batch: list[dict] = []
    processed = 0
    upserted = 0
    skipped = 0
    seen_ids: set[str] = set()
    merchant_counts: dict[str, int] = defaultdict(int)

    gz_reader = gzip.open(response.raw, mode="rt", encoding="utf-8", errors="replace")
    csv_reader = csv.DictReader(gz_reader)

    for row in csv_reader:
        processed += 1

        try:
            mapped = map_row(row)
        except Exception:
            skipped += 1
            continue

        if mapped is None:
            skipped += 1
            continue

        seen_ids.add(mapped["id"])
        merchant_name = row.get("merchant_name", "").strip()
        if merchant_name:
            merchant_counts[merchant_name] += 1

        batch.append(mapped)

        if len(batch) >= BATCH_SIZE:
            count = upsert_batch(supabase, batch, generate_embeddings, embedder)
            upserted += count
            if count < len(batch):
                skipped += len(batch) - count
            batch = []
            time.sleep(BATCH_SLEEP)

        if limit and upserted >= limit:
            print(f"\nLimit of {limit} upserted rows reached. Stopping.")
            break

        if processed % PROGRESS_EVERY == 0:
            print(f"[{processed}] Processed | Upserted: {upserted} | Skipped: {skipped}")

    if batch:
        count = upsert_batch(supabase, batch, generate_embeddings, embedder)
        upserted += count
        if count < len(batch):
            skipped += len(batch) - count

    gz_reader.close()
    response.close()

    print(f"\nFinal — Processed: {processed} | Upserted: {upserted} | Skipped: {skipped}")

    if skip_stale_delete:
        print("\nSkipping stale deletion (test mode).")
    else:
        print("\nDeleting stale Awin products...")
        total_deleted = 0
        while True:
            result = supabase.table("products").select("id").eq("source", "awin").limit(STALE_DELETE_BATCH_SIZE).execute()
            if not result.data:
                break
            stale_ids = [r["id"] for r in result.data if r["id"] not in seen_ids]
            if not stale_ids:
                break
            try:
                supabase.table("products").delete().in_("id", stale_ids).execute()
                total_deleted += len(stale_ids)
            except Exception as e:
                print(f"Error deleting stale batch: {e}")
                break
        print(f"Deleted {total_deleted} stale products")

    top_merchants = sorted(merchant_counts.items(), key=lambda x: -x[1])[:20]
    print("\nTop 20 merchants by product count:")
    for merchant, count in top_merchants:
        print(f"  {count:>8,} | {merchant}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Awin datafeed importer")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to upsert (for testing)")
    parser.add_argument("--no-stale-delete", action="store_true", help="Skip stale product deletion")
    parser.add_argument("--embed", action="store_true", help="Generate image and info embeddings with SigLIP")
    args = parser.parse_args()

    test_mode = args.limit is not None
    run(
        limit=args.limit,
        skip_stale_delete=test_mode or args.no_stale_delete,
        generate_embeddings=args.embed,
    )
