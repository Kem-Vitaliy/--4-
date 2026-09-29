"""
main.py — веб-рівень (FastAPI).

Відповідальність:
  - Прийняти текстове звернення клієнта.
  - Валідувати вхід (порожній / занадто довгий).
  - Делегувати роботу з моделлю модулю llm_service.
  - Обробити винятки llm_service → повернути правильний HTTP-статус.
  - Віддати HTML-сторінку з інтерфейсом.

Веб-рівень НЕ імпортує openai напряму.
Він знає тільки про llm_service.get_answer().
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.llm_service import (
    LLMAuthError,
    LLMConnectionError,
    LLMError,
    LLMPermissionError,
    LLMRateLimitError,
    LLMTimeoutError,
    get_answer,
)

# ── FastAPI app ──────────────────────────────────────────────
app = FastAPI(
    title="ТехноХвиля — Служба підтримки",
    description="AI-помічник служби підтримки інтернет-магазину",
    version="1.0.0",
)

# ── Обмеження вводу ──────────────────────────────────────────
MAX_QUERY_LENGTH = 2000  # символів


# ── Модель запиту ────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)


# ── Головна сторінка ─────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    """Повертає HTML-сторінку з UI чату підтримки."""
    return _INDEX_HTML


# ── Ендпоінт чату ────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    """Прийняти звернення, делегувати LLM, повернути JSON.

    Обробка помилок:
      - 400: порожнє або занадто довге звернення (Pydantic)
      - 401: невалідний API-ключ
      - 429: перевищено ліміт запитів
      - 503: сервіс недоступний / таймаут
      - 500: непередбачена помилка
    """
    user_message = req.message.strip()

    if not user_message:
        return JSONResponse(
            status_code=400,
            content={"error": "Будь ласка, введіть ваше звернення."},
        )

    try:
        result = get_answer(user_message)
        return {
            "answer": result["answer"],
            "model": result["model"],
            "elapsed_seconds": result["elapsed_seconds"],
        }

    except LLMAuthError as e:
        return JSONResponse(
            status_code=401,
            content={"error": str(e)},
        )

    except LLMPermissionError as e:
        return JSONResponse(
            status_code=403,
            content={"error": str(e)},
        )

    except LLMRateLimitError as e:
        return JSONResponse(
            status_code=429,
            content={"error": str(e)},
        )

    except (LLMTimeoutError, LLMConnectionError) as e:
        return JSONResponse(
            status_code=503,
            content={"error": str(e)},
        )

    except LLMError as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Внутрішня помилка сервісу. Спробуйте пізніше."},
        )

    except Exception:
        return JSONResponse(
            status_code=500,
            content={"error": "Внутрішня помилка сервісу. Спробуйте пізніше."},
        )


# ── Обробка помилок валідації Pydantic ───────────────────────
@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=400,
        content={
            "error": f"Некоректне звернення. Текст має бути від 1 до {MAX_QUERY_LENGTH} символів."
        },
    )


# ── HTML UI ──────────────────────────────────────────────────
_INDEX_HTML = """\
<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ТехноХвиля — Підтримка</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --text-muted: #94a3b8; --accent: #3b82f6;
    --accent-hover: #2563eb; --success: #22c55e; --danger: #ef4444;
    --user-bg: #1e3a5f; --bot-bg: #1e293b;
  }
  *, *::before, *::after { box-sizing: border-box; }
  body {
    margin: 0; font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg); color: var(--text);
    display: flex; justify-content: center; padding: 1rem;
    min-height: 100vh;
  }
  .container { max-width: 800px; width: 100%; display: flex; flex-direction: column; height: 95vh; }

  h1 { text-align: center; font-size: 1.6rem; margin: 0.5rem 0 0.2rem; }
  .subtitle { text-align: center; color: var(--text-muted); margin-bottom: 1rem; font-size: 0.9rem; }

  /* ── Chat area ── */
  .chat-area {
    flex: 1; overflow-y: auto; padding: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px 12px 0 0;
    display: flex; flex-direction: column; gap: 0.8rem;
  }
  .msg {
    max-width: 85%; padding: 0.8rem 1rem; border-radius: 12px;
    line-height: 1.5; font-size: 0.95rem; white-space: pre-wrap;
  }
  .msg-user {
    align-self: flex-end; background: var(--user-bg);
    border-bottom-right-radius: 4px;
  }
  .msg-bot {
    align-self: flex-start; background: var(--bot-bg);
    border: 1px solid var(--border); border-bottom-left-radius: 4px;
  }
  .msg-error {
    align-self: flex-start; background: #dc262620;
    border: 1px solid var(--danger); color: var(--danger);
    border-bottom-left-radius: 4px;
  }
  .msg-meta {
    font-size: 0.75rem; color: var(--text-muted); margin-top: 0.4rem;
  }

  /* ── Input area ── */
  .input-area {
    display: flex; gap: 0.5rem; padding: 0.8rem 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-top: none; border-radius: 0 0 12px 12px;
  }
  #msg-input {
    flex: 1; padding: 0.7rem 1rem; border: 1px solid var(--border);
    border-radius: 8px; background: var(--bg); color: var(--text);
    font-size: 0.95rem; resize: none; font-family: inherit;
    min-height: 44px; max-height: 120px;
  }
  #msg-input:focus { outline: none; border-color: var(--accent); }
  #msg-input::placeholder { color: var(--text-muted); }

  #send-btn {
    padding: 0.7rem 1.5rem; background: var(--accent); color: #fff;
    border: none; border-radius: 8px; font-size: 1rem; cursor: pointer;
    transition: background 0.2s; white-space: nowrap;
  }
  #send-btn:hover { background: var(--accent-hover); }
  #send-btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* ── Typing indicator ── */
  .typing {
    display: none; align-self: flex-start; padding: 0.6rem 1rem;
    background: var(--bot-bg); border: 1px solid var(--border);
    border-radius: 12px; color: var(--text-muted); font-size: 0.9rem;
  }
  .typing.active { display: block; }
  .typing span {
    animation: blink 1.4s infinite both;
  }
  .typing span:nth-child(2) { animation-delay: 0.2s; }
  .typing span:nth-child(3) { animation-delay: 0.4s; }
  @keyframes blink {
    0%, 80%, 100% { opacity: 0.3; }
    40% { opacity: 1; }
  }

  /* ── Scrollbar ── */
  .chat-area::-webkit-scrollbar { width: 6px; }
  .chat-area::-webkit-scrollbar-track { background: transparent; }
  .chat-area::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
</head>
<body>
<div class="container">
  <h1>🛒 ТехноХвиля — Підтримка</h1>
  <p class="subtitle">AI-помічник відповість на ваші запитання про доставку, оплату, повернення та гарантію</p>

  <div class="chat-area" id="chat-area">
    <div class="msg msg-bot">
      Вітаю! Я помічник служби підтримки інтернет-магазину «ТехноХвиля». 👋
      Запитайте мене про доставку, оплату, повернення, гарантію або графік роботи підтримки.
    </div>
    <div class="typing" id="typing">
      Помічник думає<span>.</span><span>.</span><span>.</span>
    </div>
  </div>

  <div class="input-area">
    <textarea id="msg-input" rows="1" placeholder="Напишіть ваше запитання…"></textarea>
    <button id="send-btn">Надіслати</button>
  </div>
</div>

<script>
const chatArea  = document.getElementById('chat-area');
const msgInput  = document.getElementById('msg-input');
const sendBtn   = document.getElementById('send-btn');
const typing    = document.getElementById('typing');

// Auto-resize textarea
msgInput.addEventListener('input', () => {
  msgInput.style.height = 'auto';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 120) + 'px';
});

// Send on Enter (Shift+Enter for new line)
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener('click', sendMessage);

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;

  // Show user message
  addMessage(text, 'user');
  msgInput.value = '';
  msgInput.style.height = 'auto';
  sendBtn.disabled = true;

  // Show typing indicator
  typing.classList.add('active');
  scrollToBottom();

  try {
    const resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    const data = await resp.json();

    typing.classList.remove('active');

    if (!resp.ok) {
      addMessage(data.error || `Помилка: HTTP ${resp.status}`, 'error');
    } else {
      const meta = `⏱ ${data.elapsed_seconds} с  •  🤖 ${data.model}`;
      addMessage(data.answer, 'bot', meta);
    }
  } catch (err) {
    typing.classList.remove('active');
    addMessage('Не вдалося з\\'єднатися з сервером. Перевірте інтернет.', 'error');
  } finally {
    sendBtn.disabled = false;
    msgInput.focus();
  }
}

function addMessage(text, type, meta = '') {
  const div = document.createElement('div');
  const cls = type === 'user' ? 'msg-user' : type === 'error' ? 'msg-error' : 'msg-bot';
  div.className = `msg ${cls}`;
  div.textContent = text;

  if (meta) {
    const metaDiv = document.createElement('div');
    metaDiv.className = 'msg-meta';
    metaDiv.textContent = meta;
    div.appendChild(metaDiv);
  }

  // Insert before typing indicator
  chatArea.insertBefore(div, typing);
  scrollToBottom();
}

function scrollToBottom() {
  chatArea.scrollTop = chatArea.scrollHeight;
}
</script>
</body>
</html>
"""
