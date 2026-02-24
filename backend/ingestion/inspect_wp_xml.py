import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

xml_path = Path("data/editorialdahbar.WordPress.2026-02-24.xml")
ns = {"wp": "http://wordpress.org/export/1.2/"}

root = ET.parse(xml_path).getroot()
items = root.find("channel").findall("item")
print("items", len(items))

ptype_counts = Counter()
attachment_urls = []
products = []

for item in items:
    post_type = item.findtext("wp:post_type", default="", namespaces=ns).strip()
    ptype_counts[post_type] += 1

    if post_type == "attachment":
        attachment_url = item.findtext("wp:attachment_url", default="", namespaces=ns).strip()
        if attachment_url:
            attachment_urls.append(attachment_url)

    if post_type == "product":
        post_id = item.findtext("wp:post_id", default="", namespaces=ns).strip()
        title = item.findtext("title", default="").strip()
        metas = {}
        for postmeta in item.findall("wp:postmeta", ns):
            key = postmeta.findtext("wp:meta_key", default="", namespaces=ns)
            value = postmeta.findtext("wp:meta_value", default="", namespaces=ns)
            if key in {"_thumbnail_id", "_product_image_gallery", "_sku", "_isbn", "isbn", "autor", "_autor", "_codigo"}:
                metas[key] = value
        products.append((post_id, title, metas))

print("post_types", ptype_counts.most_common(15))
print("attachments_with_url", len(attachment_urls))

domains = Counter(urlparse(url).netloc for url in attachment_urls)
print("domains", domains.most_common(10))

other_domains = [url for url in attachment_urls if "editorialdahbar.com" not in urlparse(url).netloc]
print("non_editorial_count", len(other_domains))
for url in other_domains[:10]:
    print("-", url)

print("products", len(products))
for post_id, title, metas in products[:5]:
    print("\nPID", post_id, "TITLE", title)
    for key, value in metas.items():
        print(" ", key, "=", (value or "")[:150])
