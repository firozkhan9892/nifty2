from flask import Flask, render_template, jsonify
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from model import predict as model_predict
import threading
import time
import logging
import re
from dotenv import load_dotenv
import os

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes, CommandHandler

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

# ==================== NEWS LOGIC ====================

STOCK_KEYWORDS = {
    "JBM AUTO": ["jbm", "jbm auto"], "OLECTRA": ["olectra", "olectra greentech"],
    "TATA MOTORS": ["tata motors", "tata ev"], "TATA POWER": ["tata power"],
    "RELIANCE": ["reliance", "ril"], "TCS": ["tcs", "tata consultancy"],
    "INFOSYS": ["infosys"], "WIPRO": ["wipro"], "HDFC BANK": ["hdfc bank", "hdfc"],
    "ICICI BANK": ["icici bank", "icici"], "SBIN": ["sbi", "state bank"],
    "ADANI": ["adani"], "BAJAJ FIN": ["bajaj finance"], "MARUTI": ["maruti"],
    "MAHINDRA": ["mahindra"], "ASHOK LEYLAND": ["ashok leyland"],
    "EICHER MOTORS": ["eicher"], "TVS MOTORS": ["tvs motor"],
    "HERO MOTOCORP": ["hero motocorp", "hero"], "VEERHEALTH": ["veerhealth"],
    "SUN PHARMA": ["sun pharma"], "DR REDDYS": ["dr reddys"],
    "CIPLA": ["cipla"], "HCL TECH": ["hcl tech"], "LT": ["larsen", "l&t"],
    "BHARTI AIRTEL": ["airtel"], "ITC": ["itc"], "IRFC": ["irfc"], "RVNL": ["rvnl"],
}

POLICY_KEYWORDS = ["policy", "govt", "government", "regulation", "tax", "subsidy", "draft", "minister", "cabinet"]
EV_KEYWORDS = ["ev ", "electric vehicle", "electric 2w", "electric 3w", "electric car", "charging station", "battery", "ev policy", "renewable energy", "solar power", "wind energy"]
GEO_KEYWORDS = ["iran", "us-", "pakistan", "china", "russia", "war", "talks in islamabad", "middle east", "ukraine", "tariff", "trade war"]
ORDER_KEYWORDS = ["order", "wins order", "receives order", "order value", "contract", "bagged", "secured", "deal worth", "deal value"]
EARNINGS_KEYWORDS = ["quarterly results", "q2", "q3", "q4", "q1", "fy24", "fy25", "revenue growth", "profit rises", "net profit", "operating profit", "earnings", "result", "quarter result", "results day", "beat estimates", "miss estimates"]
FO_KEYWORDS = ["futures and options", "f&o", "derivatives", "options expiry", "monthly expiry", "weekly expiry", " rollover", "short covering", "long build-up", "open interest", "pcr ratio"]
GLOBAL_KEYWORDS = ["wall street", "dow jones", "nasdaq", "s&p 500", "ftse", "nikkei", "asian markets", "european markets", "us fed", "federal reserve", "us inflation", "us jobs data"]
RBI_KEYWORDS = ["rbi governor", "reserve bank", "monetary policy", "repo rate", "reverse repo", "cash reserve ratio", "crr", "bank rate", "rbi policy", "rbi meeting", "inflation targeting"]
MA_KEYWORDS = ["merger", "acquisition", "acquire", "merge", "takeover", "deal", "m&a", "acquisition of", "acquires", "stake sale", "divestment"]
IPO_KEYWORDS = ["ipo", "listing", "listed on bse", "listed on nse", "initial public offer", "subscription", "grey market premium", "ipo allotment", "bse sme", "nse emerge"]
COMMODITY_KEYWORDS = ["crude oil", "brent crude", "gold price", "silver price", "commodity market", "mcx", "base metals", "aluminium", "copper price", "zinc"]
CURRENCY_KEYWORDS = ["rupee", "dollar", "forex", "usd inr", "exchange rate", "currency", "fx market", "dollar index"]
BANKING_KEYWORDS = ["npa", "non performing asset", "credit growth", "deposit", "interest rate", "bank loan", "loan growth", "retail loan", "home loan", "car loan"]
AUTO_SALES_KEYWORDS = ["sales data", "monthly sales", "automobile sales", "car sales", "two wheeler sales", "passenger vehicle", "commercial vehicle", "tractor sales"]
PSU_KEYWORDS = ["psu bank", "public sector bank", "coal india", "ongc", "ntpc", "psu", "government stake", "disinvestment", "navratna"]
INFRA_KEYWORDS = ["infrastructure", "highway", "road project", "railway project", "metro", "airport", "construction", "building project", "smart city"]

def detect_affected_stocks(text):
    text_lower = text.lower()
    return list(dict.fromkeys([stock for stock, kws in STOCK_KEYWORDS.items() for kw in kws if kw in text_lower]))

CATEGORY_PRIORITY = [
    (IPO_KEYWORDS, "🏆 IPO & LISTING"),
    (MA_KEYWORDS, "🤝 M&A DEALS"),
    (EARNINGS_KEYWORDS, "📊 QSR EARNINGS"),
    (FO_KEYWORDS, "📈 F&O UPDATE"),
    (RBI_KEYWORDS, "🏦 RBI & BANKING"),
    (BANKING_KEYWORDS, "🏦 BANKING SECTOR"),
    (AUTO_SALES_KEYWORDS, "🚗 AUTO SALES"),
    (EV_KEYWORDS, "⚡️ EV & GREEN ENERGY"),
    (POLICY_KEYWORDS, "📋 POLICY UPDATE"),
    (INFRA_KEYWORDS, "🏗️ INFRA & CONSTRUCTION"),
    (PSU_KEYWORDS, "🏢 PSU & PSU BANKS"),
    (ORDER_KEYWORDS, "📦 CORPORATE ORDER"),
    (GLOBAL_KEYWORDS, "🌐 GLOBAL MARKETS"),
    (COMMODITY_KEYWORDS, "🛢️ COMMODITIES"),
    (CURRENCY_KEYWORDS, "💱 CURRENCY & FOREX"),
    (GEO_KEYWORDS, "🌍 GEOPOLITICS"),
]

def categorize_news(title, desc):
    c = (title + " " + desc).lower()
    for keywords, category in CATEGORY_PRIORITY:
        if any(kw in c for kw in keywords):
            return category
    return "📰 MARKET NEWS"

def analyze_news_sentiment(title, desc):
    score = analyzer.polarity_scores(title + " " + desc)["compound"]
    if score > 0.3: return "🟢 Positive"
    if score < -0.3: return "🔴 Negative"
    return "🟡 Neutral"

def extract_order_value(text):
    match = re.search(r'(?:order value|worth|value of|approximately|approx\.?\s*|₹)\s*₹?([\d,.]+)\s*(?:crore|lakhs|cr|lakh|million|billion)', text.lower())
    return match.group(0).strip() if match else None

_news_cache = {"data": [], "time": 0}

def fetch_market_news():
    if not FINNHUB_API_KEY: return []
    try:
        resp = requests.get(f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}", timeout=10)
        if resp.status_code != 200: return []
        articles = resp.json()
        news_list = []
        for a in articles[:50]:
            news_list.append({
                "title": a.get("headline", ""),
                "description": a.get("summary", ""),
                "url": a.get("url", ""),
                "source": a.get("source", ""),
                "image": a.get("image", "")
            })
        return news_list
    except Exception as e:
        logger.error(f"Finnhub error: {e}")
        return []

def get_cached_news():
    if time.time() - _news_cache["time"] < 300: return _news_cache["data"]
    _news_cache["data"] = fetch_market_news()
    _news_cache["time"] = time.time()
    return _news_cache["data"]

# ==================== TRADING LOGIC ====================

def get_levels(symbol):
    try:
        df = yf.Ticker(symbol).history(period="5d")
        return (round(df["Low"].min(), 2), round(df["High"].max(), 2)) if not df.empty else (0, 0)
    except: return 0, 0

def get_news_sentiment(name):
    cached_news = get_cached_news()
    if not cached_news: return 0
    scores, name_lower = [], name.lower()
    for article in cached_news:
        title = article.get("title", "")
        desc = article.get("description", "") or ""
        text = (title + " " + desc).lower()
        if name_lower in text or name_lower.split()[0] in text:
            scores.append(analyzer.polarity_scores(title + " " + desc)["compound"])
    return sum(scores) / len(scores) if scores else 0

def analyze_stock(symbol, name, horizons=[1, 3, 7]):
    try:
        test_df = yf.Ticker(symbol).history(period="1d")
        if test_df.empty:
            return {"name": name, "error": "Yahoo Finance blocked (Check Internet/VPN)"}

        sentiment = get_news_sentiment(name)
        
        predictions = {}
        confidences = {}
        for h in horizons:
            price, prediction, confidence = model_predict(symbol, sentiment=sentiment, horizon=h, use_cache=False)
            predictions[h] = prediction
            confidences[h] = confidence
            if price == 0:
                return {"name": name, "error": f"Model training failed (Not enough data)"}
        
        current_price = price
        support, resistance = get_levels(symbol)
        
        breakout = ""
        if resistance > 0 and current_price > resistance: breakout = "🚀 Breakout"
        elif support > 0 and current_price < support: breakout = "🔻 Breakdown"
        
        primary_pred = predictions[1]
        pred_diff = (primary_pred - current_price) / current_price * 100
        
        ensemble_sentiment = (sentiment + sum([1 if p > current_price else -1 for p in predictions.values()]) / len(predictions)) / 2
        
        if pred_diff > 1 and ensemble_sentiment > 0.3 and breakout: signal = "🔥 STRONG BUY"
        elif pred_diff > 0.5 and ensemble_sentiment > 0: signal = "📈 BUY"
        elif pred_diff > 0: signal = "↗️ WEAK BUY"
        elif pred_diff < -1 and ensemble_sentiment < -0.3: signal = "🔥 STRONG SELL"
        elif pred_diff < -0.5: signal = "📉 SELL"
        elif pred_diff < 0: signal = "↘️ WEAK SELL"
        else: signal = "⏸️ HOLD"
        
        return {
            "name": name,
            "price": round(current_price, 2),
            "predictions": {h: round(p, 2) for h, p in predictions.items()},
            "changes": {h: round((p - current_price) / current_price * 100, 2) for h, p in predictions.items()},
            "confidences": confidences,
            "signal": signal,
            "support": support,
            "resistance": resistance,
            "breakout": breakout,
            "sentiment": round(sentiment, 2)
        }
    except Exception as e: 
        return {"name": name, "error": str(e)}

def analyze_all():
    stocks = [("^NSEI", "Nifty 50"), ("^NSEBANK", "BankNifty"), ("RELIANCE.NS", "Reliance"), ("TCS.NS", "TCS")]
    return [analyze_stock(s, n) for s, n in stocks]

# ==================== FLASK ROUTES ====================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/debug")
def debug():
    import traceback
    res = {}
    try:
        p, pred, conf = model_predict("RELIANCE.NS", sentiment=0, horizon=1, use_cache=False)
        res["model"] = f"SUCCESS: Price={p}, Pred={pred}, Conf={conf}%" if p != 0 else "FAIL: Returning 0"
    except Exception as e:
        res["model"] = f"ERROR: {str(e)}"
    
    try:
        r = requests.get(f"https://gnews.io/api/v4/search?q=india&lang=en&max=1&apikey={GNEWS_API_KEY}", timeout=10)
        res["gnews"] = r.json() if r.status_code != 200 else "SUCCESS"
    except Exception as e:
        res["gnews"] = f"ERROR: {str(e)}"
    return jsonify(res)

@app.route("/api/signals")
def api_signals():
    return jsonify(analyze_all())

@app.route("/api/news")
@app.route("/api/news/<category>")
def api_news(category=None):
    try:
        articles = get_cached_news()
        if not articles:
            if not FINNHUB_API_KEY:
                return jsonify([{"title": "ERROR: FINNHUB_API_KEY missing in .env file", "desc": "", "category": "❌ ERROR", "sentiment": "", "stocks": [], "source": "", "url": ""}])
            return jsonify([{"title": "Fetching news from Finnhub...", "desc": "Please refresh in a moment.", "category": "ℹ️ INFO", "sentiment": "", "stocks": [], "source": "", "url": ""}])

        news_data = []
        for a in articles:
            title = a.get("title", "") or a.get("headline", "")
            desc = a.get("description", "") or a.get("summary", "") or ""
            cat = categorize_news(title, desc)
            news_data.append({"title": title, "desc": desc[:200] + "..." if len(desc) > 200 else desc, "category": cat, "sentiment": analyze_news_sentiment(title, desc), "stocks": detect_affected_stocks(title + " " + desc), "source": a.get("source", ""), "url": a.get("url", "")})
        
        if category:
            category_map = {
                "earnings": "📊 QSR EARNINGS", "earnings": "📊 QSR EARNINGS",
                "fo": "📈 F&O UPDATE", "fno": "📈 F&O UPDATE", "derivatives": "📈 F&O UPDATE",
                "rbi": "🏦 RBI & BANKING", "banking": "🏦 BANKING SECTOR",
                "auto": "🚗 AUTO SALES", "sales": "🚗 AUTO SALES",
                "ev": "⚡️ EV & GREEN ENERGY", "green": "⚡️ EV & GREEN ENERGY", "renewable": "⚡️ EV & GREEN ENERGY",
                "policy": "📋 POLICY UPDATE", "govt": "📋 POLICY UPDATE",
                "ma": "🤝 M&A DEALS", "merger": "🤝 M&A DEALS", "acquisition": "🤝 M&A DEALS",
                "ipo": "🏆 IPO & LISTING", "listing": "🏆 IPO & LISTING",
                "infra": "🏗️ INFRA & CONSTRUCTION", "construction": "🏗️ INFRA & CONSTRUCTION",
                "psu": "🏢 PSU & PSU BANKS", "psubank": "🏢 PSU & PSU BANKS",
                "order": "📦 CORPORATE ORDER", "contract": "📦 CORPORATE ORDER",
                "global": "🌐 GLOBAL MARKETS", "us": "🌐 GLOBAL MARKETS",
                "commodity": "🛢️ COMMODITIES", "crude": "🛢️ COMMODITIES", "gold": "🛢️ COMMODITIES",
                "currency": "💱 CURRENCY & FOREX", "rupee": "💱 CURRENCY & FOREX",
                "geo": "🌍 GEOPOLITICS", "geopolitics": "🌍 GEOPOLITICS",
            }
            target_cat = category_map.get(category.lower(), None)
            if target_cat:
                news_data = [n for n in news_data if n["category"] == target_cat]
        
        return jsonify(news_data[:15])
    except Exception as e:
        return jsonify([{"title": f"Network Error: {str(e)}", "desc": "Could not connect to Finnhub", "category": "❌ ERROR", "sentiment": "", "stocks": [], "source": "", "url": ""}])

@app.route("/api/orders")
def api_orders():
    order_data = []
    for a in get_cached_news():
        title = a.get("title", "")
        desc = a.get("description", "") or ""
        combined = (title + " " + desc).lower()
        if any(kw in combined for kw in ORDER_KEYWORDS):
            order_data.append({"title": title, "value": extract_order_value(combined) or "Not Disclosed", "stocks": detect_affected_stocks(combined), "sentiment": analyze_news_sentiment(title, desc)})
    return jsonify(order_data[:5])

# ==================== TELEGRAM LOGIC ====================

_sent_news_ids = set()

def send_news_to_telegram():
    if not all([TELEGRAM_BOT_TOKEN, CHAT_ID]): return
    try:
        articles = get_cached_news()
        for a in articles[:20]:
            news_id = hash(a.get("title", ""))
            if news_id in _sent_news_ids: continue
            _sent_news_ids.add(news_id)
            
            title = a.get("title", "") or a.get("headline", "")
            desc = a.get("description", "") or a.get("summary", "") or ""
            source = a.get("source", "")
            url = a.get("url", "")
            cat = categorize_news(title, desc)
            sentiment = analyze_news_sentiment(title, desc)
            
            msg = f"{cat}\n\n📰 {title}\n\n{desc[:200]}...\n\n{sentiment} | 📍 {source}"
            
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
            time.sleep(2)
    except Exception as e: 
        logger.error(f"News Telegram error: {e}")

def auto_telegram():
    while True:
        try:
            if not all([TELEGRAM_BOT_TOKEN, CHAT_ID]): time.sleep(60); continue
            
            send_news_to_telegram()
            
            data = analyze_all()
            msg = "📊 *AI TRADING SIGNALS*\n\n"
            for s in data:
                if "error" not in s: 
                    pred1 = s.get("predictions", {}).get(1, s.get("prediction", 0))
                    msg += f"*{s['name']}*\n💰 Price: ₹{s['price']}\n🎯 1D: ₹{pred1}\n📊 {s['signal']}\n\n"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        except Exception as e: logger.error(f"Signal error: {e}")
        time.sleep(1800)

async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = analyze_all()
    msg = "📊 *AI TRADING SIGNALS*\n\n"
    for s in data:
        if "error" not in s:
            preds = s.get("predictions", {})
            changes = s.get("changes", {})
            msg += f"*{s['name']}*\n💰 Price: ₹{s['price']}\n🎯 1D: ₹{preds.get(1, 0)} ({changes.get(1, 0)}%)\n🎯 3D: ₹{preds.get(3, 0)} ({changes.get(3, 0)}%)\n🎯 7D: ₹{preds.get(7, 0)} ({changes.get(7, 0)}%)\n📊 {s['signal']}\n\n"
    await update.message.reply_text(msg, parse_mode="Markdown")

async def reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    data = analyze_all()
    stock_map = {"nifty": 0, "bank": 1, "reliance": 2, "tcs": 3}
    idx = next((i for k, i in stock_map.items() if k in text), 0)
    stock = data[idx]
    if "error" in stock: await update.message.reply_text(f"❌ {stock['error']}"); return
    
    preds = stock.get("predictions", {})
    changes = stock.get("changes", {})
    
    msg = f"📊 *{stock['name']}*\n\n💰 Price: ₹{stock['price']}\n\n🎯 *Predictions:*\n1D: ₹{preds.get(1, 0)} ({changes.get(1, 0)}%)\n3D: ₹{preds.get(3, 0)} ({changes.get(3, 0)}%)\n7D: ₹{preds.get(7, 0)} ({changes.get(7, 0)}%)\n\n📊 Signal: {stock['signal']}\n\n📍 Support: ₹{stock['support']}\n📍 Resistance: ₹{stock['resistance']}\n\n{stock['breakout']}"
    await update.message.reply_text(msg, parse_mode="Markdown")

def run_reply_bot():
    if not TELEGRAM_BOT_TOKEN: return
    app_bot = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("signal", cmd_signal))
    app_bot.add_handler(MessageHandler(filters.TEXT, reply))
    app_bot.run_polling()

if __name__ == "__main__":
    threading.Thread(target=auto_telegram, daemon=True).start()
    threading.Thread(target=run_reply_bot, daemon=True).start()
    app.run(debug=False, use_reloader=False, port=5000)