import json
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

xml_path = Path("data/editorialdahbar.WordPress.2026-02-24.xml")
out_dir = Path("data/reconciliation_reports")
out_dir.mkdir(parents=True, exist_ok=True)

ns = {"wp": "http://wordpress.org/export/1.2/"}
root = ET.parse(xml_path).getroot()
items = root.find("channel").findall("item")

attachments = []
for item in items:
    post_type = item.findtext("wp:post_type", default="", namespaces=ns).strip()
    if post_type != "attachment":
        continue

    attachments.append(
        {
            "post_id": item.findtext("wp:post_id", default="", namespaces=ns).strip(),
            "title": item.findtext("title", default="").strip(),
            "url": item.findtext("wp:attachment_url", default="", namespaces=ns).strip(),
        }
    )

probe = []
for item in attachments[:25]:
    url = item["url"]
    try:
        response = requests.get(url, timeout=20, allow_redirects=True)
        probe.append(
            {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("Content-Type", ""),
                "sg_captcha": response.headers.get("sg-captcha", ""),
                "is_image": response.headers.get("Content-Type", "").lower().startswith("image/"),
            }
        )
    except Exception as error:
        probe.append({"url": url, "error": str(error)})

(out_dir / "xml_attachment_urls.json").write_text(
    json.dumps(attachments, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
(out_dir / "xml_attachment_access_probe.json").write_text(
    json.dumps(probe, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print("attachments_in_xml", len(attachments))
print("probe_written", len(probe))
