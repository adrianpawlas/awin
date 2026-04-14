import os
import sys
import gzip
import csv
import time
import json
import re
import argparse
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
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

DEFAULT_AWIN_DATAFEED_URLS = [
    "https://productdata.awin.com/datafeed/download/apikey/746193b8a6735d6afaff119db9e3bcc7/language/any/fid/56927,60563,62231,85969,89656,91998,92148,92975,95015,112087/rid/0,1,2,3,4,5,6,7,8,9,10/hasEnhancedFeeds/0/columns/aw_deep_link,product_name,aw_product_id,merchant_product_id,merchant_image_url,description,merchant_category,search_price,merchant_name,merchant_id,category_name,category_id,aw_image_url,currency,store_price,delivery_cost,merchant_deep_link,language,last_updated,display_price,data_feed_id,brand_name,colour,specifications,brand_id,product_short_description,keywords,saving,savings_percent,large_image,alternate_image_three,alternate_image_four,alternate_image_two,alternate_image,Fashion%3Asize,Fashion%3Amaterial,Fashion%3Acategory/format/csv/delimiter/%2C/compression/gzip/",
    "https://productdata.awin.com/datafeed/download/apikey/746193b8a6735d6afaff119db9e3bcc7/language/any/fid/113050/rid/0/hasEnhancedFeeds/0/columns/aw_deep_link,product_name,aw_product_id,merchant_product_id,merchant_image_url,description,merchant_category,search_price,merchant_name,merchant_id,category_name,category_id,aw_image_url,currency,store_price,delivery_cost,merchant_deep_link,language,last_updated,display_price,data_feed_id,brand_name,colour,specifications,brand_id,keywords,product_short_description,saving,savings_percent,large_image,alternate_image_three,alternate_image_four,alternate_image_two,alternate_image/format/csv/delimiter/%2C/compression/gzip/",
]

BATCH_SIZE = 50
BATCH_SLEEP = 0.05
EMBED_DELAY = 0.5
STALE_DELETE_BATCH_SIZE = 100
STALE_CONSECUTIVE_RUNS = 2
PROGRESS_EVERY = 1000
EMBED_IMAGE_TIMEOUT = 10
EMBED_THREADS = 32
SIGLIP_MODEL = "vit_base_patch16_siglip_384.webli"
MAX_RETRIES = 3
COMPRESS_IMAGES = True
COMPRESS_QUALITY = 85

WOMEN_KEYWORDS = [
    "women", "woman", "female", "ladies", "girl", "womenswear", "femme",
    "dámské", "dáma", "žena", "ženské", "kobieta", "kobiet", "mujer", "mujeres", "dama",
    "damska", "damské", "ženská",
]
MEN_KEYWORDS = [
    "men", "man", "male", "boys", "menswear", "homme",
    "pánské", "pán", "muž", "mężczyzna", "mężczyźni", "hombre", "hombres", "varón",
    "pánska", "pánske", "mužská",
]


def get_feed_urls() -> list[str]:
    env_urls = os.environ.get("AWIN_DATAFEED_URLS")
    if env_urls:
        urls = [u.strip() for u in env_urls.replace("\n", ",").split(",") if u.strip()]
        if urls:
            return urls
    return DEFAULT_AWIN_DATAFEED_URLS


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

    def compress_image_url(self, url: str) -> Optional[str]:
        if not url or not COMPRESS_IMAGES:
            return None
        try:
            api_url = f"https://api.resmush.it/ws.php?img={url}&qlty={COMPRESS_QUALITY}"
            headers = {
                "User-Agent": "AwinImporter/1.0",
                "Referer": "https://github.com/adrianpawlas/awin"
            }
            resp = requests.get(api_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("error"):
                return None
            return data.get("dest")
        except Exception:
            return None

    def compress_batch_images(self, items: list[tuple[str, str]]) -> dict[str, Optional[str]]:
        results: dict[str, Optional[str]] = {}
        total = len(items)
        done = 0
        with ThreadPoolExecutor(max_workers=EMBED_THREADS) as executor:
            futures = {executor.submit(self.compress_image_url, url): id_ for id_, url in items}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  [Compress] {done}/{total} images compressed")
        return results

    def embed_batch_images(self, items: list[tuple[str, str]], batch_idx: int = 0) -> dict[str, Optional[list[float]]]:
        results: dict[str, Optional[list[float]]] = {}
        total = len(items)
        done = 0
        with ThreadPoolExecutor(max_workers=EMBED_THREADS) as executor:
            futures = {executor.submit(self.embed_image_url, url): id_ for id_, url in items}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
                done += 1
                if done % 100 == 0 or done == total:
                    print(f"  [Embed batch {batch_idx}] {done}/{total} images embedded")
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


def format_price(value: str, currency: str) -> str:
    num = parse_price(value)
    if num is None:
        return None
    curr = currency.strip().upper() if currency else "EUR"
    return f"{num:.2f}{curr}"


TRANSLATOR_CACHE: dict[str, str] = {}
MAX_TRANSLATE_CACHE = 5000


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    if len(text) < 3:
        return text
    cache_key = text[:100]
    if cache_key in TRANSLATOR_CACHE:
        return TRANSLATOR_CACHE[cache_key]
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source="auto", target="en")
        result = translator.translate(text)
        if len(TRANSLATOR_CACHE) < MAX_TRANSLATE_CACHE:
            TRANSLATOR_CACHE[cache_key] = result or text
        return result or text
    except Exception:
        return text


def build_info_text(row: dict, language: str = "en") -> str:
    parts = []
    pn = row.get("product_name", "").strip()
    if pn:
        if language != "en":
            pn = translate_to_english(pn)
        parts.append(f"Product: {pn}")
    bn = row.get("brand_name", "").strip()
    mn = row.get("merchant_name", "").strip()
    if bn:
        parts.append(f"Brand: {bn}")
    elif mn:
        parts.append(f"Brand: {mn}")
    desc = row.get("description", "").strip()
    if desc:
        if language != "en":
            desc = translate_to_english(desc)
        parts.append(f"Description: {desc[:500]}")
    short_desc = row.get("product_short_description", "").strip()
    if short_desc and short_desc not in desc:
        if language != "en":
            short_desc = translate_to_english(short_desc)
        parts.append(f"Short description: {short_desc[:200]}")
    fc = row.get("Fashion:category", "").strip()
    cn = row.get("category_name", "").strip()
    mc = row.get("merchant_category", "").strip()
    cat = fc or cn or mc
    if cat:
        if language != "en":
            cat = translate_to_english(cat)
        parts.append(f"Category: {cat}")
    size = row.get("Fashion:size", "").strip()
    if size:
        parts.append(f"Size: {size}")
    mat = row.get("Fashion:material", "").strip()
    if mat:
        if language != "en":
            mat = translate_to_english(mat)
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

    merchant_image_url = row.get("merchant_image_url", "").strip()
    aw_image_url = row.get("aw_image_url", "").strip()
    large_image = row.get("large_image", "").strip()
    image_url = merchant_image_url or aw_image_url or large_image
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
    currency = row.get("currency", "").strip()
    price_val = display_price if display_price else (search_price if search_price else store_price)
    formatted_price = format_price(price_val, currency)

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
        "formatted_price": formatted_price,
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
        "price": formatted_price,
        "sale": sale,
        "additional_images": additional_images_str,
        "info_embedding": None,
        "_row": row,
    }




def fetch_existing_products(supabase: Client, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    try:
        result = supabase.table("products").select("id, image_url, title, brand, price, category, product_url, affiliate_url, updated_at").in_("id", ids).execute()
        return {r["id"]: r for r in result.data}
    except Exception as e:
        print(f"Error fetching existing products: {e}")
        return {}


def has_changes(existing: dict, new: dict) -> bool:
    if not existing:
        return True
    existing_title = existing.get("title") or ""
    new_title = new.get("title") or ""
    existing_brand = existing.get("brand") or ""
    new_brand = new.get("brand") or ""
    existing_price = existing.get("price") or ""
    new_price = new.get("price") or ""
    existing_category = existing.get("category") or ""
    new_category = new.get("category") or ""
    existing_product_url = existing.get("product_url") or ""
    new_product_url = new.get("product_url") or ""
    existing_affiliate_url = existing.get("affiliate_url") or ""
    new_affiliate_url = new.get("affiliate_url") or ""
    existing_image_url = existing.get("image_url") or ""
    new_image_url = new.get("image_url") or ""

    if existing_title != new_title:
        return True
    if existing_brand != new_brand:
        return True
    if existing_price != new_price:
        return True
    if existing_category != new_category:
        return True
    if existing_product_url != new_product_url:
        return True
    if existing_affiliate_url != new_affiliate_url:
        return True
    if existing_image_url != new_image_url:
        return True
    return False


def upsert_batch_with_retry(
    supabase: Client,
    batch: list[dict],
    use_embeddings: bool,
    embedder: Optional[SiglipEmbedder] = None,
    existing_products: Optional[dict[str, dict]] = None,
    has_updated_at: bool = True,
    has_created_at: bool = True,
) -> tuple[int, int, int]:
    if not batch:
        return 0, 0, 0

    now = datetime.now(timezone.utc).isoformat()

    new_count = 0
    updated_count = 0
    unchanged_count = 0

    to_insert = []
    to_update = []
    need_image_embed: list[dict] = []
    need_info_embed: list[dict] = []

    for p in batch:
        pid = p["id"]
        existing = existing_products.get(pid) if existing_products else None

        if not existing:
            new_count += 1
            to_insert.append(p)
        elif has_changes(existing, p):
            updated_count += 1
            to_update.append(p)
            if existing.get("image_url") != p.get("image_url"):
                need_image_embed.append(p)
                need_info_embed.append(p)
        else:
            unchanged_count += 1

    inserted_ids = []
    updated_ids = []

    if to_insert and use_embeddings and embedder:
        image_items = [(p["id"], p["image_url"]) for p in to_insert]
        image_embs = embedder.embed_batch_images(image_items)
        
        if COMPRESS_IMAGES:
            print(f"  Compressing {len(to_insert)} images...")
            compressed_urls = embedder.compress_batch_images(image_items)
            for p in to_insert:
                p["compressed_image_url"] = compressed_urls.get(p["id"])

        for p in to_insert:
            pid = p["id"]
            p["image_embedding"] = image_embs.get(pid)
            row = p.pop("_row")
            lang = row.get("language", "en")
            info_text = build_info_text(row, lang)
            p["info_embedding"] = embedder.embed_text(info_text)

    if to_insert:
        cleaned = []
        for p in to_insert:
            p.pop("_row", None)
            c = {k: v for k, v in p.items() if k not in ("image_embedding", "info_embedding")}
            if has_created_at:
                c["created_at"] = now
            if has_updated_at:
                c["updated_at"] = now
            if not c.get("product_url"):
                c["product_url"] = None
            if not c.get("affiliate_url"):
                c["affiliate_url"] = None
            if c.get("compressed_image_url") is None:
                c.pop("compressed_image_url", None)
            cleaned.append(c)

        try:
            supabase.table("products").upsert(cleaned, on_conflict="id").execute()
            inserted_ids = [p["id"] for p in to_insert]
        except Exception as e:
            err_str = str(e)
            if "updated_at" in err_str or "created_at" in err_str:
                cleaned_no_ts = [{k: v for k, v in c.items() if k not in ("created_at", "updated_at")} for c in cleaned]
                try:
                    supabase.table("products").upsert(cleaned_no_ts, on_conflict="id").execute()
                    inserted_ids = [p["id"] for p in to_insert]
                except Exception as e2:
                    if "duplicate key" in str(e2).lower():
                        inserted_ids = []
                    else:
                        print(f"Insert failed: {e2}")
                        inserted_ids = []
            elif "duplicate key" in err_str.lower():
                inserted_ids = []
            else:
                print(f"Insert failed: {e}")
                inserted_ids = []

    if to_update and use_embeddings and embedder:
        batch_idx = 0
        for p in need_image_embed:
            image_emb = embedder.embed_image_url(p["image_url"])
            p["image_embedding"] = image_emb
            
            if COMPRESS_IMAGES:
                compressed = embedder.compress_image_url(p["image_url"])
                p["compressed_image_url"] = compressed
            
            row = p.pop("_row")
            lang = row.get("language", "en")
            info_text = build_info_text(row, lang)
            p["info_embedding"] = embedder.embed_text(info_text)
            batch_idx += 1
            if batch_idx % 20 == 0:
                print(f"  [Embed update] {batch_idx}/{len(need_image_embed)} processed")

        batch_idx = 0
        for p in need_info_embed:
            if p.get("image_embedding") is None:
                row = p.pop("_row")
                lang = row.get("language", "en")
                info_text = build_info_text(row, lang)
                p["info_embedding"] = embedder.embed_text(info_text)
                batch_idx += 1
                if batch_idx % 20 == 0:
                    print(f"  [Embed info-only] {batch_idx}/{len(need_info_embed)} processed")

    if to_update:
        cleaned = []
        for p in to_update:
            p.pop("_row", None)
            c = {k: v for k, v in p.items() if k not in ("image_embedding", "info_embedding")}
            if has_updated_at:
                c["updated_at"] = now
            if not c.get("product_url"):
                c["product_url"] = None
            if not c.get("affiliate_url"):
                c["affiliate_url"] = None
            if c.get("compressed_image_url") is None:
                c.pop("compressed_image_url", None)
            cleaned.append(c)

        try:
            supabase.table("products").upsert(cleaned, on_conflict="id").execute()
            updated_ids = [p["id"] for p in to_update]
        except Exception as e:
            err_str = str(e)
            if "updated_at" in err_str or "created_at" in err_str:
                cleaned_no_ts = [{k: v for k, v in c.items() if k not in ("created_at", "updated_at")} for c in cleaned]
                try:
                    supabase.table("products").upsert(cleaned_no_ts, on_conflict="id").execute()
                    updated_ids = [p["id"] for p in to_update]
                except Exception as e2:
                    if "duplicate key" in str(e2).lower():
                        updated_ids = []
                    else:
                        print(f"Update failed: {e2}")
                        updated_ids = []
            elif "duplicate key" in err_str.lower():
                updated_ids = []
            else:
                print(f"Update failed: {e}")
                updated_ids = []

    embed_ids = list(set(inserted_ids + updated_ids))
    embed_map = {}
    for p in to_insert + to_update:
        pid = p["id"]
        if pid in embed_ids:
            embed_map[pid] = {
                "image_embedding": p.get("image_embedding"),
                "info_embedding": p.get("info_embedding"),
            }

    if embed_map:
        try:
            for pid, emb in embed_map.items():
                if has_updated_at:
                    update_data = {"updated_at": now}
                else:
                    update_data = {}
                if emb["image_embedding"] is not None:
                    update_data["image_embedding"] = emb["image_embedding"]
                if emb["info_embedding"] is not None:
                    update_data["info_embedding"] = emb["info_embedding"]
                if len(update_data) > 0:
                    supabase.table("products").update(update_data).eq("id", pid).execute()
        except Exception as e:
            err_str = str(e)
            if "updated_at" in err_str:
                try:
                    for pid, emb in embed_map.items():
                        update_data = {}
                        if emb["image_embedding"] is not None:
                            update_data["image_embedding"] = emb["image_embedding"]
                        if emb["info_embedding"] is not None:
                            update_data["info_embedding"] = emb["info_embedding"]
                        if len(update_data) > 0:
                            supabase.table("products").update(update_data).eq("id", pid).execute()
                except Exception as e2:
                    print(f"Embed update error: {e2}")
            else:
                print(f"Embed update error: {e}")

    return new_count, updated_count, unchanged_count


def download_feed_with_retry(url: str, max_retries: int = 3) -> Optional[str]:
    """Download feed to a temp file with retry. Returns temp file path or None."""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Download attempt {attempt}/{max_retries}...")
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            
            tmp = tempfile.NamedTemporaryFile(suffix=".gz", delete=False)
            tmp_path = tmp.name
            
            total_bytes = 0
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
                    total_bytes += len(chunk)
            tmp.close()
            resp.close()
            
            print(f"  Downloaded {total_bytes / (1024*1024):.1f} MB")
            return tmp_path
        except Exception as e:
            print(f"  Download attempt {attempt} failed: {e}")
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            if attempt == max_retries:
                return None
            time.sleep(2 * attempt)
    return None


def process_feed(
    supabase: Client,
    url: str,
    seen_ids: set[str],
    merchant_counts: dict[str, int],
    generate_embeddings: bool,
    embedder: Optional[SiglipEmbedder],
    limit: Optional[int],
    existing_products: dict[str, dict],
    has_updated_at: bool = True,
    has_created_at: bool = True,
) -> tuple[int, int, int, int, int]:
    print(f"\nFetching: {url[:80]}...")
    
    tmp_path = download_feed_with_retry(url)
    if tmp_path is None:
        print(f"Failed to fetch feed after retries: {url[:80]}...")
        return 0, 0, 0, 0, 0

    batch: list[dict] = []
    processed = 0
    new_count = 0
    updated_count = 0
    unchanged_count = 0
    skipped = 0

    try:
        gz_reader = gzip.open(tmp_path, mode="rt", encoding="utf-8", errors="replace")
        csv_reader = csv.DictReader(gz_reader)
    except Exception as e:
        print(f"Failed to open downloaded feed: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return 0, 0, 0, 0, 0

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

        if mapped["id"] in seen_ids:
            skipped += 1
            continue

        seen_ids.add(mapped["id"])
        merchant_name = row.get("merchant_name", "").strip()
        if merchant_name:
            merchant_counts[merchant_name] += 1

        batch.append(mapped)

        if len(batch) >= BATCH_SIZE:
            new, updated, unchanged = upsert_batch_with_retry(
                supabase, batch, generate_embeddings, embedder, existing_products, has_updated_at, has_created_at
            )
            new_count += new
            updated_count += updated
            unchanged_count += unchanged
            for p in batch:
                pid = p.get("id")
                if pid:
                    existing_products[pid] = {"id": pid, "image_url": p.get("image_url")}
            batch = []
            time.sleep(BATCH_SLEEP)

        current_count = new_count + updated_count + unchanged_count
        if limit and current_count >= limit:
            print(f"Limit of {limit} upserted rows reached. Stopping.")
            break

        if processed % PROGRESS_EVERY == 0:
            print(f"  [{processed}] Processed | New: {new_count} | Updated: {updated_count} | Unchanged: {unchanged_count}")

    if batch:
        new, updated, unchanged = upsert_batch_with_retry(
            supabase, batch, generate_embeddings, embedder, existing_products, has_updated_at, has_created_at
        )
        new_count += new
        updated_count += updated
        unchanged_count += unchanged

    gz_reader.close()

    if os.path.exists(tmp_path):
        os.unlink(tmp_path)

    return processed, new_count, updated_count, unchanged_count, skipped


def load_existing_awin_products(supabase: Client) -> dict[str, dict]:
    print("Loading existing Awin products from database...")
    all_products = {}
    offset = 0
    batch = 1000
    try:
        result = supabase.table("products").select("id, image_url, title, brand, price, category, product_url, affiliate_url").eq("source", "awin").range(offset, offset + batch - 1).execute()
        while result.data:
            for r in result.data:
                all_products[r["id"]] = r
            if len(result.data) < batch:
                break
            offset += batch
            result = supabase.table("products").select("id, image_url, title, brand, price, category, product_url, affiliate_url").eq("source", "awin").range(offset, offset + batch - 1).execute()
    except Exception as e:
        print(f"Error loading existing products: {e}")
    print(f"Loaded {len(all_products)} existing Awin products")
    return all_products


def check_table_columns(supabase: Client) -> tuple[bool, bool]:
    has_updated_at = False
    has_created_at = False
    
    test_id = "_test_col_check_"
    test_data = [{"id": test_id, "source": "awin", "product_url": f"https://test.com/{test_id}"}]
    try:
        supabase.table("products").upsert(test_data, on_conflict="id").execute()
        supabase.table("products").delete().eq("id", test_id).execute()
        has_updated_at = True
        has_created_at = True
    except Exception as e:
        err_str = str(e)
        if "updated_at" in err_str and "created_at" in err_str:
            has_updated_at = False
            has_created_at = False
        elif "updated_at" not in err_str:
            has_updated_at = True
        if "created_at" not in err_str:
            has_created_at = True
    
    return has_updated_at, has_created_at


def run(limit: Optional[int] = None, skip_stale_delete: bool = False, generate_embeddings: bool = False):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")

    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    embedder: Optional[SiglipEmbedder] = None
    if generate_embeddings:
        embedder = SiglipEmbedder.get_instance()
        print("Embedding generation enabled.")

    print("Checking table schema...")
    has_updated_at, has_created_at = check_table_columns(supabase)
    print(f"Schema: updated_at={has_updated_at}, created_at={has_created_at}")

    existing_products = load_existing_awin_products(supabase)

    feed_urls = get_feed_urls()
    print(f"Starting Awin datafeed import...")
    print(f"Found {len(feed_urls)} feed(s) to process")
    if generate_embeddings:
        print("Embedding generation: ENABLED (SigLIP)")
    else:
        print("Embedding generation: DISABLED (use --embed to enable)")

    seen_ids: set[str] = set()
    merchant_counts: dict[str, int] = defaultdict(int)
    total_processed = 0
    total_new = 0
    total_updated = 0
    total_unchanged = 0
    total_skipped = 0

    for i, url in enumerate(feed_urls):
        print(f"\n--- Feed {i+1}/{len(feed_urls)} ---")
        processed, new, updated, unchanged, skipped = process_feed(
            supabase, url, seen_ids, merchant_counts, generate_embeddings, embedder, limit, existing_products, has_updated_at, has_created_at
        )
        total_processed += processed
        total_new += new
        total_updated += updated
        total_unchanged += unchanged
        total_skipped += skipped
        
        current_upserted = total_new + total_updated
        if limit and current_upserted >= limit:
            print(f"\nGlobal limit of {limit} upserted rows reached. Stopping.")
            break

    print(f"\n{'='*50}")
    print(f"RUN SUMMARY")
    print(f"{'='*50}")
    print(f"Processed: {total_processed}")
    print(f"New: {total_new}")
    print(f"Updated: {total_updated}")
    print(f"Unchanged: {total_unchanged}")
    print(f"Skipped (invalid/missing data): {total_skipped}")
    print(f"Unique products seen: {len(seen_ids)}")

    if skip_stale_delete:
        print("\nSkipping stale deletion (test mode).")
    else:
        print("\nProcessing stale Awin products...")
        stale_deleted = 0
        stale_products = []

        for pid, prod in existing_products.items():
            if pid not in seen_ids:
                stale_products.append(pid)

        for pid in stale_products:
            try:
                supabase.table("products").delete().eq("id", pid).execute()
                stale_deleted += 1
            except Exception as e:
                print(f"Error deleting stale product {pid}: {e}")

        print(f"Deleted {stale_deleted} stale products")

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
    parser.add_argument("--url", type=str, default=None, help="Single feed URL to process")
    args = parser.parse_args()

    test_mode = args.limit is not None
    
    if args.url:
        os.environ["AWIN_DATAFEED_URLS"] = args.url
    
    run(
        limit=args.limit,
        skip_stale_delete=test_mode or args.no_stale_delete,
        generate_embeddings=args.embed,
    )
