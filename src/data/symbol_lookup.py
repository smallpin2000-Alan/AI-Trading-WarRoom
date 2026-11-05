# -*- coding: utf-8 -*-
"""
臺灣證券與期貨商品代碼與中文名稱智慧對應與查詢模組
支援：
  1. 內建熱門臺指期貨、臺灣 50、中型 100、熱門 ETF、AI 權值股高速離線對應 (0ms)
  2. 臺灣證券交易所 (TWSE) 與櫃買中心 (TPEx) 官方即時 API 動態查表 (全市場 2,000+ 檔上市櫃股票)
  3. 本地快取自動持久化 (config/symbol_names_cache.json)
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional
import requests

CACHE_PATH = Path("config/symbol_names_cache.json")

# 常用臺灣期貨、權值股、熱門 ETF 離線高速對應表 (0ms 瞬發)
BUILTIN_SYMBOLS: Dict[str, str] = {
    # 期貨全系列
    "TXF": "大台指",
    "TXFR1": "大台指近月",
    "TXFR2": "大台指次月",
    "MTX": "小台指",
    "MTXR1": "小台指近月",
    "MTXR2": "小台指次月",
    "MXF": "小台指",
    "TMF": "微台指",
    "TMFR1": "微台指近月",
    "TMFR2": "微台指次月",
    "TE": "電子期",
    "TER1": "電子期近月",
    "TF": "金融期",
    "TFR1": "金融期近月",
    "ZEF": "半導體30期",
    "ZEFR1": "半導體30期近月",
    "XIF": "航運期",
    "XIFR1": "航運期近月",
    "TGF": "台灣生技期",

    # 熱門權值股與臺灣 50 成分股
    "2330": "台積電",
    "2317": "鴻海",
    "2454": "聯發科",
    "2308": "台達電",
    "2382": "廣達",
    "2412": "中華電",
    "2881": "富邦金",
    "2882": "國泰金",
    "2891": "中信金",
    "2303": "聯電",
    "3711": "日月光投控",
    "2886": "兆豐金",
    "3231": "緯創",
    "2603": "長榮",
    "2357": "華碩",
    "6669": "緯穎",
    "3008": "大立光",
    "2884": "玉山金",
    "2892": "第一金",
    "2885": "元大金",
    "5880": "合庫金",
    "1216": "統一",
    "2880": "華南金",
    "2002": "中鋼",
    "2883": "凱基金",
    "3045": "台灣大",
    "2887": "台新金",
    "2327": "國巨",
    "2395": "研華",
    "4904": "遠傳",
    "2345": "智邦",
    "5871": "中租-KY",
    "2912": "統一超",
    "3034": "聯詠",
    "2890": "永豐金",
    "2379": "瑞昱",
    "2801": "彰銀",
    "6505": "台塑化",
    "1301": "台塑",
    "1303": "南亞",
    "1326": "台化",
    "2609": "陽明",
    "2615": "萬海",
    "3661": "世芯-KY",
    "3443": "創意",
    "3529": "力旺",
    "2059": "川湖",
    "2356": "英業達",
    "2376": "技嘉",
    "2383": "台光電",
    "3037": "欣興",
    "2408": "南亞科",

    # 熱門 ETF
    "0050": "元大台灣50",
    "0051": "元大中型100",
    "0056": "元大高股息",
    "006208": "富邦台50",
    "00713": "元大台灣高息低波",
    "00878": "國泰永續高股息",
    "00881": "國泰台灣5G+",
    "00891": "中信關鍵半導體",
    "00919": "群益台灣精選高息",
    "00929": "復華台灣科技優息",
    "00939": "統一台灣高息動能",
    "00940": "元大台灣價值高息",

    # 熱門中小型股與 AI 概念股
    "1503": "士電",
    "1513": "中興電",
    "1514": "亞力",
    "1519": "華城",
    "2301": "光寶科",
    "2324": "仁寶",
    "2344": "華邦電",
    "2352": "佳世達",
    "2353": "宏碁",
    "2377": "微星",
    "2385": "群光",
    "2409": "友達",
    "2449": "京元電子",
    "2451": "創見",
    "2474": "可成",
    "2498": "宏達電",
    "2610": "華航",
    "2618": "長榮航",
    "3017": "奇鋐",
    "3324": "雙鴻",
    "3481": "群創",
    "3702": "大聯大",
    "4938": "和碩",
    "4966": "譜瑞-KY",
    "5269": "祥碩",
    "5483": "中美晶",
    "5876": "上海商銀",
    "6274": "台燿",
    "6415": "矽力*-KY",
    "6488": "環球晶",
    "6770": "力積電",
    "8046": "南電",
    "8069": "元太",
    "8299": "群聯",
    "8996": "高力",
    "9904": "寶成",
    "9910": "豐泰",
    "9945": "潤泰新",
}

_memory_cache: Dict[str, str] = dict(BUILTIN_SYMBOLS)
_loaded_disk_cache = False


def _load_disk_cache() -> None:
    global _loaded_disk_cache
    if _loaded_disk_cache:
        return
    _loaded_disk_cache = True
    if CACHE_PATH.exists():
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _memory_cache.update({str(k).upper(): str(v).strip() for k, v in data.items()})
        except Exception:
            pass


def _save_disk_cache() -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 只存額外擴充的非內建代碼
        custom_data = {k: v for k, v in _memory_cache.items() if k not in BUILTIN_SYMBOLS or BUILTIN_SYMBOLS[k] != v}
        CACHE_PATH.write_text(json.dumps(custom_data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_symbol_name(symbol_code: str) -> str:
    """
    根據商品代碼自動查詢繁體中文名稱：
      1. 先查記憶體/內建高速對應表 (0ms)
      2. 若無則連線 TWSE/TPEx 官方 API 動態查詢 (上市/上櫃全市場) 並持久化快取
    """
    clean = (symbol_code or "").strip().upper()
    if not clean:
        return ""

    _load_disk_cache()
    if clean in _memory_cache:
        return _memory_cache[clean]

    # 若為期貨商品但帶有特殊別名
    fut_base = clean.replace("R1", "").replace("R2", "").strip()
    if fut_base in _memory_cache:
        base_name = _memory_cache[fut_base]
        suffix = "近月" if "R1" in clean else ("次月" if "R2" in clean else "")
        resolved = f"{base_name}{suffix}" if suffix and not base_name.endswith(suffix) else base_name
        _memory_cache[clean] = resolved
        return resolved

    # 若為數字代碼（臺灣股票或 ETF），線上向 TWSE 官方查詢
    if re.match(r"^\d{4,6}[A-Z]?$", clean):
        try:
            url = f"https://mis.twse.com.tw/stock/api/getStockInfo.jsp?ex_ch=tse_{clean}.tw|otc_{clean}.tw"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                for row in data.get("msgArray", []):
                    code = str(row.get("c", "")).strip().upper()
                    name = str(row.get("n", "")).strip()
                    if code == clean and name:
                        _memory_cache[clean] = name
                        _save_disk_cache()
                        return name
        except Exception:
            pass

    return ""


def format_symbol_display(entry: str) -> str:
    """
    將代碼或自選清單項目標準化為「代碼 (商品名稱)」格式
    例如：
      - "2330" -> "2330 (台積電)"
      - "2603" -> "2603 (長榮)"
      - "TMFR1" -> "TMFR1 (微台指近月)"
      - "2330 (台積電)" -> "2330 (台積電)"
    """
    clean_entry = (entry or "").strip()
    if not clean_entry:
        return ""

    # 若已包含括號名稱，例如 "2330 (台積電)"
    if "(" in clean_entry and clean_entry.endswith(")"):
        code_part = clean_entry.split("(")[0].strip().upper()
        name_part = clean_entry[clean_entry.find("(") + 1 : -1].strip()
        # 若括號內為空，嘗試補齊
        if not name_part:
            known_name = get_symbol_name(code_part)
            return f"{code_part} ({known_name})" if known_name else code_part
        return f"{code_part} ({name_part})"

    # 純代碼形式，例如 "2603" 或 "TMFR1"
    clean_code = clean_entry.split(" ")[0].strip().upper()
    name = get_symbol_name(clean_code)
    if name:
        return f"{clean_code} ({name})"
    return clean_code
