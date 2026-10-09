# -*- coding: utf-8 -*-
import sys, json
import yt_dlp

QUERIES = [
    "UTM lecture",
    "Universiti Teknologi Malaysia lecture",
    "Universiti Malaya lecture",
    "UKM lecture english",
    "USM lecture",
    "UPM lecture english",
]

ydl_opts = {
    "skip_download": True,
    "quiet": True,
    "no_warnings": True,
    "extract_flat": False,
    "playlist_items": "1-6",
    "noplaylist": True,
}

def scrub(info):
    title = info.get("title") or "?"
    ch = info.get("channel") or info.get("uploader") or "?"
    subs = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    human = sorted(subs.keys())
    has_auto = len(auto) > 0
    # 判断是否像人工字幕: 有 en / en-orig / en-US
    en_human = [k for k in human if k.startswith("en")]
    # computed automatic captions 也可能宣称 en
    return {
        "title": title[:70],
        "channel": ch[:40],
        "url": info.get("webpage_url") or info.get("original_url") or "",
        "human_sub_langs": human,
        "en_human": en_human,
        "has_auto_captions": has_auto,
    }

out = []
for q in QUERIES:
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info("ytsearch6:" + q, download=False)
            entries = info.get("entries") or []
            for e in entries:
                if e:
                    out.append(scrub(e))
    except Exception as ex:
        out.append({"error": str(ex)[:100], "q": q})

for r in out:
    print(json.dumps(r, ensure_ascii=False))
