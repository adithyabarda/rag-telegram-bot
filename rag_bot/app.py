from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from telegram.request import HTTPXRequest
from rag import retrieve, stream_answer
import requests
import asyncio
from dotenv import load_dotenv
import os

load_dotenv() 

TOKEN = os.getenv("TELEGRAM_TOKEN")

# =========================
# 1. USER HISTORY
# =========================
user_history = {}

# =========================
# /ask COMMAND
# =========================
async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    query = " ".join(context.args)

    if not query:
        await update.message.reply_text("❗ Provide a question.")
        return

    msg = await update.message.reply_text("⏳ Generating...")

    try:
        if user_id not in user_history:
            user_history[user_id] = []

        history = user_history[user_id][-3:]
        history_text = "\n".join(history)

        retrieved_data = retrieve(query)

        # 🔥 STREAMING
        last_text = ""

        for partial in stream_answer(query, retrieved_data, history_text):
            if partial != last_text:
                try:
                    await msg.edit_text(partial[:4000])  # Telegram limit
                    last_text = partial
                    await asyncio.sleep(0.2)
                except:
                    pass

        # Save final
        user_history[user_id].append(f"Q: {query}\nA: {last_text}")

    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)}")

# =========================
# /summarize COMMAND
# =========================
async def summarize(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    if user_id not in user_history or not user_history[user_id]:
        await update.message.reply_text("❌ No conversation to summarize.")
        return

    last_chat = "\n".join(user_history[user_id][-3:])

    prompt = f"Summarize this conversation:\n{last_chat}"

    res = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    summary = res.json().get("response", "Error")
    await update.message.reply_text(summary.strip())

# =========================
# /start COMMAND
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to Advanced RAG Bot!\n\nUse /ask to query.\nUse /summarize for summary."
    )

# =========================
# /help COMMAND
# =========================
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("/ask <question>\n/summarize\n/help")

# =========================
# MAIN
# =========================
def main():
    request = HTTPXRequest(
        connect_timeout=60.0,
        read_timeout=60.0,
        write_timeout=60.0,
        pool_timeout=60.0
    )

    app = ApplicationBuilder().token(TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("summarize", summarize))
    app.add_handler(CommandHandler("help", help_command))

    print("🚀 Advanced RAG Bot running...")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()