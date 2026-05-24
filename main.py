"""
LM Studio Chat - FastAPI アプリケーション（Corpus2Skill 版）
"""

import asyncio
import httpx
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    from mem0 import AsyncMemory
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import stdio_client

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
from rag import Corpus2SkillManager


# ─── アプリケーション設定 ───────────────────────────────────────────────

app = FastAPI(title="LM Studio Chat (Corpus2Skill)")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

HISTORY_FILE = BASE_DIR / "chat_history.json"

# ─── Corpus2Skill 初期化 ─────────────────────────────────────────────────

RAG_DOCS_DIR    = BASE_DIR / "rag_docs"
RAG_CONFIG_FILE = BASE_DIR / "rag_config.json"
RAG_DOCS_DIR.mkdir(exist_ok=True)


def load_rag_config() -> dict:
    """rag_config.json を読み込む。存在しない場合はデフォルト値を返す"""
    defaults = {
        "llm_model": "",
        "max_top_skills": 6,
        "branching_factor": 4,
        "chunk_max_chars": 800,
        "overlap_chars": 0,
    }
    if RAG_CONFIG_FILE.exists():
        try:
            cfg = json.loads(RAG_CONFIG_FILE.read_text(encoding="utf-8"))
            return {
                "llm_model":        cfg.get("llm_model", ""),
                "max_top_skills":   int(cfg.get("max_top_skills", 6)),
                "branching_factor": int(cfg.get("branching_factor", 4)),
                "chunk_max_chars":  int(cfg.get("chunk_max_chars", 800)),
                "overlap_chars":    int(cfg.get("overlap_chars", 0)),
            }
        except Exception:
            pass
    return defaults


# ─── マルチコレクション管理 ──────────────────────────────────────────────

C2S_ROOT = BASE_DIR / "c2s_db"
C2S_ROOT.mkdir(parents=True, exist_ok=True)
COLLECTIONS_META_FILE = C2S_ROOT / "_collections.json"

rag_managers: dict[str, "Corpus2SkillManager"] = {}
active_collections: list[str] = ["default"]  # 検索に使うコレクション（複数可）
active_collection: str = "default"           # 管理対象コレクション（アップロード等）
mem0_memory: "AsyncMemory | None" = None


def get_active_managers() -> list["Corpus2SkillManager"]:
    """検索に使うマネージャ一覧を返す"""
    return [rag_managers[n] for n in active_collections if n in rag_managers]


def get_active_manager() -> "Corpus2SkillManager | None":
    """管理対象マネージャ（アップロード・削除等に使う）"""
    return rag_managers.get(active_collection)


def load_collections_meta() -> dict:
    if COLLECTIONS_META_FILE.exists():
        try:
            return json.loads(COLLECTIONS_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"active": "default", "collections": {}}


def save_collections_meta(meta: dict):
    COLLECTIONS_META_FILE.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


async def _migrate_to_collections():
    """旧単一コレクション構造をマルチコレクション構造に移行する"""
    old_index = C2S_ROOT / "_indexed_docs.json"
    default_dir = C2S_ROOT / "default"
    if old_index.exists() and not COLLECTIONS_META_FILE.exists():
        print("[Corpus2Skill] 旧データ形式を検出。'default' コレクションへ移行します...")
        default_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "_indexed_docs.json", "documents.json", "_doc_metadata.json",
            "chunk_index.json", "skill_meta.json", "_compile_info.json",
        ]:
            src = C2S_ROOT / name
            if src.exists():
                dst = default_dir / name
                if not dst.exists():
                    shutil.move(str(src), str(dst))
        for name in ["embeddings.npy"]:
            src = C2S_ROOT / name
            if src.exists():
                dst = default_dir / name
                if not dst.exists():
                    shutil.move(str(src), str(dst))
        for name in ["skills", "doc_embeddings"]:
            src = C2S_ROOT / name
            if src.exists():
                dst = default_dir / name
                if not dst.exists():
                    shutil.move(str(src), str(dst))
        # rag_docs も移行
        old_rag_docs = BASE_DIR / "rag_docs"
        new_rag_docs = old_rag_docs / "default"
        if old_rag_docs.exists() and not new_rag_docs.exists():
            new_rag_docs.mkdir(parents=True, exist_ok=True)
            for item in old_rag_docs.iterdir():
                if item.is_file():
                    shutil.move(str(item), str(new_rag_docs / item.name))
        print("[Corpus2Skill] 'default' コレクションへの移行完了")


async def _create_manager_for_collection(name: str) -> "Corpus2SkillManager":
    """コレクション用 Corpus2SkillManager を作成・初期化する"""
    cfg = load_rag_config()
    base_url = (
        f"http://{os.getenv('LM_STUDIO_HOST', '127.0.0.1')}"
        f":{os.getenv('LM_STUDIO_PORT', '1234')}/v1"
    )
    llm_model = cfg.get("llm_model", "") or os.getenv("LM_STUDIO_LLM_MODEL", "")
    manager = Corpus2SkillManager(
        working_dir=str(C2S_ROOT / name),
        lm_studio_base_url=base_url,
        llm_model=llm_model,
        embed_model=os.getenv(
            "C2S_EMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        max_top_skills=cfg.get("max_top_skills", 6),
        branching_factor=cfg.get("branching_factor", 4),
        chunk_max_chars=cfg.get("chunk_max_chars", 800),
        overlap_chars=cfg.get("overlap_chars", 0),
    )
    await manager.initialize()
    return manager


def get_collection_docs_dir(collection_name: str) -> Path:
    """コレクション別アップロードフォルダを返す（なければ作成）"""
    d = BASE_DIR / "rag_docs" / collection_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_embedding_dims(model_name: str) -> int:
    """sentence-transformers モデルの埋め込み次元数を取得する"""
    _known = {
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
    }
    short_name = model_name.split("/")[-1]
    return _known.get(short_name, _known.get(model_name, 384))


def _create_mem0_config() -> dict:
    host = os.getenv("LM_STUDIO_HOST", "127.0.0.1")
    port = os.getenv("LM_STUDIO_PORT", "1234")
    base_url = f"http://{host}:{port}"
    api_key = os.getenv("LM_STUDIO_API_KEY", "") or "lm-studio"
    cfg = load_rag_config()
    llm_model = cfg.get("llm_model") or os.getenv("LM_STUDIO_LLM_MODEL", "") or "local"
    embed_model = os.getenv(
        "C2S_EMBED_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    embedding_dims = _get_embedding_dims(embed_model)
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": llm_model,
                "api_key": api_key,
                "openai_base_url": f"{base_url}/v1",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": embed_model,
                "embedding_dims": embedding_dims,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "chat_memories",
                "path": str(BASE_DIR / "mem0_db"),
                "embedding_model_dims": embedding_dims,
            },
        },
    }


def _get_user_id(session: str) -> str:
    parts = session.split("_", 2)
    if len(parts) >= 2 and parts[0] == "user":
        return parts[1]
    return "default"


def _extract_text_content(content) -> str:
    """content が string または multimodal array の場合に text 部分のみを返す"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ).strip()
    return str(content)


def _parse_mem0_results(result) -> list[dict]:
    if isinstance(result, dict):
        return result.get("results", [])
    if isinstance(result, list):
        return result
    return []


@app.on_event("startup")
async def startup_event():
    global rag_managers, active_collections, active_collection, mem0_memory

    # 旧データ形式からの移行
    await _migrate_to_collections()

    # コレクションメタデータ読み込み
    meta = load_collections_meta()
    active_collection = meta.get("active", "default")

    # default コレクションは必ず存在させる
    collection_names: set[str] = {"default"}
    for name in meta.get("collections", {}).keys():
        collection_names.add(name)
    # c2s_db/ 内のディレクトリも読み込み対象
    for p in C2S_ROOT.iterdir():
        if p.is_dir() and not p.name.startswith("_"):
            collection_names.add(p.name)

    for name in sorted(collection_names):
        coll_dir = C2S_ROOT / name
        try:
            manager = await _create_manager_for_collection(name)
            rag_managers[name] = manager
            print(f"[Corpus2Skill] コレクション '{name}' 初期化完了 - {manager.get_status()['total_documents']} ドキュメント")
        except Exception as e:
            print(f"[Corpus2Skill] コレクション '{name}' 初期化失敗: {e}")

    if not rag_managers:
        print("[Corpus2Skill] コレクションなし（RAG無効）")

    if active_collection not in rag_managers and rag_managers:
        active_collection = next(iter(rag_managers))

    # 検索対象コレクションを設定（存在するコレクションのみ）
    stored_active = meta.get("active_collections", None)
    if stored_active is not None:
        active_collections = [n for n in stored_active if n in rag_managers]
    if not active_collections:
        active_collections = [active_collection] if active_collection in rag_managers else []

    save_collections_meta({
        "active": active_collection,
        "active_collections": active_collections,
        "collections": {name: {} for name in rag_managers},
    })

    if _MEM0_AVAILABLE:
        try:
            mem0_memory = AsyncMemory.from_config(_create_mem0_config())
            print("[mem0] 初期化完了")
        except Exception as e:
            print(f"[mem0] 初期化失敗（メモリ機能は無効）: {e}")
            mem0_memory = None


# ─── モデル定義 ─────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 8196


class LoginRequest(BaseModel):
    username: str
    password: str


class SettingsUpdate(BaseModel):
    lm_studio_host: str
    lm_studio_port: int
    app_username: str
    app_password: str
    lm_studio_api_key: str


# ─── チャット履歴管理 ───────────────────────────────────────────────────


def load_history() -> dict:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_history(history: dict) -> None:
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def get_session_history(session_id: str) -> list[dict]:
    history = load_history()
    return history.get(session_id, [])


def append_to_history(session_id: str, role: str, content: str) -> None:
    history = load_history()
    if session_id not in history:
        history[session_id] = []

    history[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

    if len(history[session_id]) > 100:
        history[session_id] = history[session_id][-100:]

    save_history(history)


def clear_session_history(session_id: str) -> None:
    history = load_history()
    if session_id in history:
        del history[session_id]
    save_history(history)


# ─── LM Studio API 通信 ─────────────────────────────────────────────────


async def fetch_models() -> list[dict]:
    try:
        api_key = config.Config.get_api_key()
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                config.Config.get_models_endpoint(),
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception as e:
        print(f"モデル取得エラー: {e}")
        return []


async def send_to_lm_studio(
    messages: list[dict],
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8196
) -> str:
    api_key = config.Config.get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                config.Config.get_api_endpoint(),
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        return "⚠️ タイムアウトしました。LM Studio サーバーが実行中か確認してください。"
    except httpx.HTTPStatusError as e:
        return f"⚠️ HTTPエラー: {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"⚠️ エラーが発生しました: {str(e)}"


# ─── 認証 ───────────────────────────────────────────────────────────────


def check_auth(request: Request) -> Optional[str]:
    session = request.cookies.get("session_token")
    if not session:
        return None
    return session


# ─── ルート ─────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session = request.cookies.get("session_token")

    if session:
        return templates.TemplateResponse(
            name="chat.html",
            context={
                "authenticated": True,
                "session_id": session,
                "history": get_session_history(session),
                "models": await fetch_models(),
                "lm_studio_url": config.Config.get_lm_studio_url(),
                "backend_type": config.Config.get_backend_type(),
            },
            request=request
        )

    return templates.TemplateResponse(
        name="login.html",
        context={
            "authenticated": False,
            "error": "ユーザー名またはパスワードが異なります。",
        },
        request=request
    )


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    username = form.get("username", "")
    password = form.get("password", "")

    if config.Config.is_authenticated(username, password):
        response = HTMLResponse(
            """<script>window.location.href='/';</script>"""
        )
        response.set_cookie(
            key="session_token",
            value=f"user_{username}_{datetime.now().timestamp()}",
            httponly=True,
            max_age=86400,
        )
        return response

    return templates.TemplateResponse("login.html", {
        "request": request,
        "authenticated": False,
        "error": "ユーザー名またはパスワードが異なります。",
    })


@app.get("/logout")
async def logout(request: Request):
    response = HTMLResponse("""<script>window.location.href='/';</script>""")
    response.delete_cookie(key="session_token")
    return response


@app.get("/api/models")
async def api_models(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    models = await fetch_models()
    return JSONResponse({"models": models})


@app.post("/api/chat")
async def api_chat(request: Request):
    """チャットリクエスト API（ストリーミング SSE + Corpus2Skill RAG + ツール対応版）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    messages    = body.get("messages", [])
    model       = body.get("model", "")
    temperature = body.get("temperature", 0.7)
    max_tokens  = body.get("max_tokens", 8192)
    tools       = body.get("tools", [])
    use_rag     = body.get("use_rag", False)
    use_memory  = body.get("use_memory", False)

    # モデル未指定時は利用可能な先頭モデルを自動選択
    # （複数モデルが起動中の場合に LM Studio がエラーを返すのを防ぐ）
    if not model:
        available = await fetch_models()
        if available:
            model = available[0].get("id", "")
            print(f"[chat] model 未指定 → 先頭モデルを使用: {model}")

    if not messages:
        return JSONResponse({"error": "メッセージが空です"}, status_code=400)

    # ユーザーの発言を先に取得（履歴保存用）
    # content が multimodal array の場合は text 部分のみ抽出
    user_content = _extract_text_content(
        next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
    )

    async def generate():
        chat_messages = list(messages)
        rag_sources: list[dict] = []
        rag_chunks: list[dict] = []
        memory_context: list[dict] = []

        # ── メモリ検索 ──────────────────────────────────────────────────
        if use_memory and mem0_memory and user_content:
            yield f"data: {json.dumps({'type': 'status', 'message': '🧠 メモリ検索中...'})}\n\n"
            try:
                user_id = _get_user_id(session)
                results = await mem0_memory.search(
                    user_content, top_k=5, filters={"user_id": user_id}
                )
                memory_context = _parse_mem0_results(results)
                if memory_context:
                    memory_text = "\n".join(
                        f"- {'[ユーザー]' if m.get('role') == 'user' else '[AI]' if m.get('role') == 'assistant' else ''} {m.get('memory', m.get('text', ''))}".strip()
                        for m in memory_context
                    )
                    chat_messages = [{
                        "role": "system",
                        "content": f"【会話の記憶】\n{memory_text}",
                    }] + chat_messages
                    # フロントエンドに使用したメモリ項目を通知
                    yield f"data: {json.dumps({'type': 'memory_done', 'items': [{'memory': m.get('memory', m.get('text', '')), 'score': round(float(m.get('score', 0)), 3), 'role': m.get('role', '')} for m in memory_context]})}\n\n"
            except Exception as e:
                print(f"[mem0] 検索エラー: {e}")

        # ── RAG 検索 ──────────────────────────────────────────────────
        active_mgrs = get_active_managers()
        if use_rag and active_mgrs and user_content:
            yield f"data: {json.dumps({'type': 'status', 'message': '📄 RAG検索中...'})}\n\n"
            all_hits: list[dict] = []
            for mgr in active_mgrs:
                hits_per = await mgr.search(user_content)
                all_hits.extend(hits_per)
            # スコア降順ソート → ソースごとに重複排除（最高スコアを優先）
            all_hits.sort(key=lambda h: h["score"], reverse=True)
            seen_sources: set[str] = set()
            hits: list[dict] = []
            for h in all_hits:
                if h["source"] not in seen_sources:
                    seen_sources.add(h["source"])
                    hits.append(h)
            if hits:
                context_text = "\n\n".join(h["content"] for h in hits)
                rag_system = {
                    "role": "system",
                    "content": (
                        "[STRICT INSTRUCTION]\n"
                        "以下に提供されたドキュメントコンテキストのみを使用して質問に答えてください。\n\n"
                        "ルール:\n"
                        "1. 以下のデータに明示されている事実のみを使用する。\n"
                        "2. 学習知識や事前情報は使用しない。\n"
                        "3. データに答えがない場合は「提供されたデータに記載がありません」と回答する。\n"
                        "4. 明示されていない関係性を推測・推論しない。\n\n"
                        "【ドキュメントコンテキスト】\n"
                        f"{context_text}"
                    ),
                }
                chat_messages = [rag_system] + chat_messages
                rag_sources = [{"source": h["source"], "score": h["score"]} for h in hits]
                rag_chunks = [c for h in hits for c in h.get("chunks", [])]
            yield f"data: {json.dumps({'type': 'rag_done', 'sources': rag_sources, 'chunks': rag_chunks})}\n\n"

        # ── ストリーミングチャット ─────────────────────────────────────
        full_reply = ""
        async for chunk_json in chat_with_tools_streaming(
            messages=chat_messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            try:
                data = json.loads(chunk_json)
                if data["type"] == "chunk":
                    full_reply += data.get("content", "")
                elif data["type"] == "done":
                    full_reply = data.get("content", full_reply)
            except Exception:
                pass
            yield f"data: {chunk_json}\n\n"

        # ── 履歴保存・メタデータ送信 ──────────────────────────────────
        if user_content:
            append_to_history(session, "user", user_content)
        if full_reply:
            append_to_history(session, "assistant", full_reply)

        # ── メモリ保存（会話後） ────────────────────────────────────────
        # infer=False: LLM抽出をスキップして直接埋め込み保存（ローカルLLM互換）
        if use_memory and mem0_memory and user_content and full_reply:
            try:
                user_id = _get_user_id(session)
                result = await mem0_memory.add(
                    [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": full_reply},
                    ],
                    user_id=user_id,
                    infer=False,
                )
                added = result.get("results", result) if isinstance(result, dict) else result
                print(f"[mem0] メモリ保存: {len(added) if isinstance(added, list) else added}")
            except Exception as e:
                print(f"[mem0] 追加エラー: {e}")

        yield f"data: {json.dumps({'type': 'meta', 'history': get_session_history(session), 'rag_sources': rag_sources, 'memory_updated': use_memory, 'memory_context': [{'memory': m.get('memory', m.get('text', '')), 'score': round(float(m.get('score', 0)), 3), 'role': m.get('role', '')} for m in memory_context]})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── MCP ツール実行 ──────────────────────────────────────────────────────

async def call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    try:
        async with stdio_client(config.Config.get_mcp_server_params()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, tool_args)
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(str(block))
                return "\n".join(parts)
    except Exception as e:
        return json.dumps({"error": f"MCPツール呼び出しエラー: {str(e)}"}, ensure_ascii=False)


async def chat_with_tools_streaming(
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_tool_iterations: int = 5,
):
    """
    LM Studio とのチャットをストリーミングで行い、JSON 文字列を yield する。

    イベント種別:
      {"type": "chunk",     "content": "..."}   テキストチャンク（リアルタイム）
      {"type": "done",      "content": "..."}   最終テキスト（ツールなし完了時）
      {"type": "tool_call", "name": "..."}      ツール実行中
      {"type": "status",    "message": "..."}   状態メッセージ
      {"type": "error",     "content": "..."}   エラー

    テキスト応答はリアルタイムにストリームされる。
    ツール呼び出しがある場合は完了まで処理し、最終テキスト応答をストリームする。
    """
    current_messages = list(messages)
    tool_definitions = list(tools) if tools else []

    api_key = config.Config.get_api_key()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    for iteration in range(max_tool_iterations):
        payload: dict = {
            "messages": current_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if model:
            payload["model"] = model
        if tool_definitions:
            payload["tools"] = tool_definitions
            payload["tool_choice"] = "auto"

        print(f"[chat_streaming] iteration={iteration}, messages={len(current_messages)}")

        full_content = ""
        tool_calls_acc: dict[int, dict] = {}
        has_tool_calls = False

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    config.Config.get_api_endpoint(),
                    headers=headers,
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})

                            # テキストチャンク → ツール呼び出しがなければリアルタイム送信
                            content = delta.get("content") or ""
                            if content and not has_tool_calls:
                                full_content += content
                                yield json.dumps({"type": "chunk", "content": content})

                            # ツールコールデルタを蓄積
                            for tc in delta.get("tool_calls") or []:
                                has_tool_calls = True
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": "", "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.get("id"):
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                if fn := tc.get("function"):
                                    tool_calls_acc[idx]["function"]["name"]      += fn.get("name", "")
                                    tool_calls_acc[idx]["function"]["arguments"] += fn.get("arguments", "")
                        except Exception:
                            pass

        except httpx.HTTPStatusError as e:
            yield json.dumps({"type": "error", "content": f"⚠️ HTTPエラー: {e.response.status_code}"})
            return
        except Exception as e:
            yield json.dumps({"type": "error", "content": f"⚠️ API通信エラー: {str(e)}"})
            return

        tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else []

        if not tool_calls:
            # テキスト応答で完了
            yield json.dumps({"type": "done", "content": full_content})
            return

        # ── ツール呼び出し処理 ──────────────────────────────────────
        current_messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            func_name = call["function"]["name"]
            raw_args  = call["function"].get("arguments", "{}")
            try:
                func_args = json.loads(raw_args)
            except json.JSONDecodeError:
                func_args = {}

            sanitized_args = {
                k: (None if v in (None, "None", "null", "") else v)
                for k, v in func_args.items()
            }

            yield json.dumps({"type": "tool_call", "name": func_name})
            print(f"[chat_streaming] MCP tool: {func_name}({sanitized_args})")
            result_content = await call_mcp_tool(func_name, sanitized_args)

            current_messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result_content,
            })

    yield json.dumps({"type": "error", "content": "⚠️ ツール呼び出しの最大反復回数を超えました。"})


# ─── mem0 メモリ API ─────────────────────────────────────────────────────


@app.get("/api/memory")
async def api_get_memory(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not mem0_memory:
        return JSONResponse({"memories": [], "available": False})
    try:
        user_id = _get_user_id(session)
        result = await mem0_memory.get_all(filters={"user_id": user_id}, top_k=50)
        memories = _parse_mem0_results(result)
        return JSONResponse({"memories": memories, "available": True})
    except Exception as e:
        return JSONResponse({"memories": [], "available": True, "error": str(e)})


@app.delete("/api/memory/{memory_id}")
async def api_delete_memory(memory_id: str, request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not mem0_memory:
        raise HTTPException(status_code=503, detail="mem0が初期化されていません")
    try:
        await mem0_memory.delete(memory_id)
        return JSONResponse({"status": "deleted"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/memory/clear")
async def api_clear_memory(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    if not mem0_memory:
        raise HTTPException(status_code=503, detail="mem0が初期化されていません")
    try:
        user_id = _get_user_id(session)
        await mem0_memory.delete_all(user_id=user_id)
        return JSONResponse({"status": "cleared"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/memory/status")
async def api_memory_status(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    return JSONResponse({"available": mem0_memory is not None})


@app.get("/api/history/{session_id}")
async def api_get_history(session_id: str, request: Request):
    _ = check_auth(request)
    return JSONResponse({"history": get_session_history(session_id)})


@app.post("/api/clear-history")
async def api_clear_history(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    clear_session_history(session)
    return JSONResponse({"status": "cleared"})


@app.get("/api/settings")
async def api_get_settings(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    backend = config.Config.get_backend_type()
    default_port = "11434" if backend == "ollama" else "1234"
    return JSONResponse({
        "backend_type":   backend,
        "lm_studio_host": os.getenv("LM_STUDIO_HOST", "127.0.0.1"),
        "lm_studio_port": os.getenv("LM_STUDIO_PORT", default_port),
        "has_api_key":    bool(os.getenv("LM_STUDIO_API_KEY", "")),
        "app_username":   os.getenv("APP_USERNAME", "admin"),
    })


@app.post("/api/update-settings")
async def api_update_settings(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    env_path = BASE_DIR / ".env"

    updates: dict[str, str] = {
        "BACKEND_TYPE":     str(body.get("backend_type", "lmstudio")),
        "LM_STUDIO_HOST":   str(body.get("lm_studio_host", "")),
        "LM_STUDIO_PORT":   str(body.get("lm_studio_port", "")),
        "LM_STUDIO_API_KEY": str(body.get("lm_studio_api_key", "")),
        "APP_USERNAME":     str(body.get("app_username", "")),
    }
    if body.get("app_password"):
        updates["APP_PASSWORD"] = str(body["app_password"])

    existing_lines: list[str] = []
    if env_path.exists():
        existing_lines = env_path.read_text(encoding="utf-8").splitlines()

    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            new_lines.append(line)
            continue
        if "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    for key, value in updates.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    load_dotenv(env_path, override=True)

    return JSONResponse({"status": "updated", "message": "設定を更新しました。サーバーを再起動してください。"})


# ─── Function Calling ツール設定 ─────────────────────────────────────────

TOOLS_CONFIG_FILE = BASE_DIR / "tools_config.json"


@app.get("/api/tools")
async def api_get_tools(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    if TOOLS_CONFIG_FILE.exists():
        tools = json.loads(TOOLS_CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        tools = []
    return JSONResponse({"tools": tools})


@app.post("/api/tools")
async def api_save_tools(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    tools = body.get("tools", [])
    TOOLS_CONFIG_FILE.write_text(
        json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return JSONResponse({"status": "saved", "count": len(tools)})


# ─── Corpus2Skill エンドポイント ─────────────────────────────────────────


@app.get("/api/rag/config")
async def rag_get_config(request: Request):
    """Corpus2Skill 設定を取得"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    cfg = load_rag_config()
    manager = get_active_manager()
    if manager:
        status = manager.get_status()
        cfg["llm_model"] = status.get("llm_model", "")
    return JSONResponse(cfg)


@app.post("/api/rag/config")
async def rag_save_config(request: Request):
    """Corpus2Skill 設定（LLM モデル・スキルツリーパラメータ）を保存"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    llm_model        = str(body.get("llm_model", "")).strip()
    max_top_skills   = int(body.get("max_top_skills", 6))
    branching_factor = int(body.get("branching_factor", 4))
    chunk_max_chars  = int(body.get("chunk_max_chars", 800))
    overlap_chars    = int(body.get("overlap_chars", 0))

    RAG_CONFIG_FILE.write_text(
        json.dumps({
            "llm_model":        llm_model,
            "max_top_skills":   max_top_skills,
            "branching_factor": branching_factor,
            "chunk_max_chars":  chunk_max_chars,
            "overlap_chars":    overlap_chars,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 全コレクションにパラメータを反映
    for m in rag_managers.values():
        current_model = m.get_status().get("llm_model", "")
        if llm_model != current_model:
            await m.set_llm_model(llm_model)
        await m.set_compile_params(max_top_skills, branching_factor, chunk_max_chars, overlap_chars)

    return JSONResponse({
        "status": "saved",
        "llm_model":        llm_model,
        "max_top_skills":   max_top_skills,
        "branching_factor": branching_factor,
        "chunk_max_chars":  chunk_max_chars,
        "overlap_chars":    overlap_chars,
    })


@app.get("/api/rag/status")
async def rag_status(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)
    status = manager.get_status()
    status["active_collection"] = active_collection
    status["active_collections"] = active_collections
    return JSONResponse(status)


@app.post("/api/rag/upload")
async def rag_upload(request: Request, file: UploadFile = File(...)):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".txt", ".pdf", ".md", ".docx"}:
        return JSONResponse({"error": f"未対応の形式: {suffix}"}, status_code=400)

    docs_dir = get_collection_docs_dir(active_collection)
    dest = docs_dir / file.filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = await manager.add_document(dest)
    return JSONResponse(result)


@app.post("/api/rag/index-dir")
async def rag_index_dir(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    docs_dir = get_collection_docs_dir(active_collection)
    results = await manager.add_directory(docs_dir)
    success = sum(1 for r in results if r.get("success"))
    return JSONResponse({
        "total": len(results),
        "success": success,
        "failed": len(results) - success,
        "details": results,
    })


@app.delete("/api/rag/document/{file_name}")
async def rag_delete_document(file_name: str, request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = await manager.delete_document(file_name)
    return JSONResponse(result)


@app.delete("/api/rag/clear")
async def rag_clear(request: Request):
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    result = await manager.clear()
    return JSONResponse(result)


@app.post("/api/rag/search")
async def rag_search(request: Request):
    """RAG 検索テスト用エンドポイント（detail=true で個別チャンク返却）"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)

    body = await request.json()
    query  = body.get("query", "")
    detail = bool(body.get("detail", False))
    top_k  = int(body.get("top_k", 10))
    if not query:
        return JSONResponse({"error": "queryが空です"}, status_code=400)

    if detail:
        chunks = await manager.search_chunks(query, top_k=top_k)
        return JSONResponse({"chunks": chunks})
    else:
        hits = await manager.search(query)
        return JSONResponse({"results": hits})


@app.get("/api/rag/compile-status")
async def rag_compile_status_endpoint(request: Request):
    """コンパイル進捗状態を返す"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"state": "error", "current_skill": 0, "total_skills": 0, "message": "RAGが初期化されていません"})
    return JSONResponse(manager.get_compile_status())


@app.post("/api/rag/recompile")
async def rag_recompile(request: Request):
    """現在のドキュメントでスキルツリーを再構築する"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    manager = get_active_manager()
    if not manager:
        return JSONResponse({"error": "RAGが初期化されていません"}, status_code=503)
    result = await manager.recompile()
    return JSONResponse(result)


# ─── コレクション管理エンドポイント ──────────────────────────────────────────


@app.get("/api/rag/collections")
async def rag_list_collections(request: Request):
    """コレクション一覧を返す"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    collections = []
    for name, mgr in sorted(rag_managers.items()):
        s = mgr.get_status()
        collections.append({
            "name": name,
            "total_documents": s["total_documents"],
            "chunk_count": s["chunk_count"],
            "is_active": name in active_collections,
            "is_management": name == active_collection,
        })
    return JSONResponse({
        "active_collections": active_collections,
        "active": active_collection,
        "collections": collections,
    })


@app.post("/api/rag/collections/active-set")
async def rag_set_active_collections(request: Request):
    """検索に使うコレクションを複数設定する"""
    global active_collections
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")
    body = await request.json()
    names = body.get("names", [])
    valid = [n for n in names if n in rag_managers]
    if not valid and rag_managers:
        valid = [next(iter(rag_managers))]
    active_collections = valid
    meta = load_collections_meta()
    meta["active_collections"] = active_collections
    save_collections_meta(meta)
    return JSONResponse({"status": "ok", "active_collections": active_collections})


@app.post("/api/rag/collections")
async def rag_create_collection(request: Request):
    """新しいコレクションを作成する"""
    global rag_managers, active_collections, active_collection
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    body = await request.json()
    name = str(body.get("name", "")).strip()
    if not name:
        return JSONResponse({"error": "コレクション名を指定してください"}, status_code=400)
    import re as _re
    if not _re.match(r'^[\w\-぀-ヿ㐀-鿿]+$', name):
        return JSONResponse({"error": "コレクション名に使えない文字が含まれています（英数字・日本語・ハイフン・アンダースコアのみ）"}, status_code=400)
    if name in rag_managers:
        return JSONResponse({"error": f"'{name}' は既に存在します"}, status_code=400)

    try:
        manager = await _create_manager_for_collection(name)
        rag_managers[name] = manager
        # 新しいコレクションを管理対象・検索対象に自動設定
        active_collection = name
        if name not in active_collections:
            active_collections.append(name)
        meta = load_collections_meta()
        meta.setdefault("collections", {})[name] = {}
        meta["active"] = active_collection
        meta["active_collections"] = active_collections
        save_collections_meta(meta)
        return JSONResponse({"status": "created", "name": name})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/api/rag/collections/{name}")
async def rag_delete_collection(name: str, request: Request):
    """コレクションを削除する"""
    global rag_managers, active_collections, active_collection
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    if name not in rag_managers:
        return JSONResponse({"error": "コレクションが見つかりません"}, status_code=404)
    if len(rag_managers) <= 1:
        return JSONResponse({"error": "最後のコレクションは削除できません"}, status_code=400)

    coll_dir = C2S_ROOT / name
    if coll_dir.exists():
        shutil.rmtree(coll_dir)
    del rag_managers[name]

    # 検索対象から除外
    if name in active_collections:
        active_collections = [n for n in active_collections if n != name]
        if not active_collections and rag_managers:
            active_collections = [next(iter(rag_managers))]

    # 管理対象から除外
    if active_collection == name:
        active_collection = active_collections[0] if active_collections else next(iter(rag_managers))

    meta = load_collections_meta()
    meta.get("collections", {}).pop(name, None)
    meta["active"] = active_collection
    meta["active_collections"] = active_collections
    save_collections_meta(meta)

    return JSONResponse({
        "status": "deleted",
        "name": name,
        "active": active_collection,
        "active_collections": active_collections,
    })


@app.post("/api/rag/collections/{name}/activate")
async def rag_activate_collection(name: str, request: Request):
    """管理対象コレクションを切り替える（アップロード・削除等の対象）"""
    global active_collection
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    if name not in rag_managers:
        return JSONResponse({"error": "コレクションが見つかりません"}, status_code=404)

    active_collection = name
    meta = load_collections_meta()
    meta["active"] = name
    save_collections_meta(meta)

    return JSONResponse({"status": "activated", "active": name})


# ─── スキルツリービューア ──────────────────────────────────────────────────

@app.get("/rag-search", response_class=HTMLResponse)
async def rag_search_page(request: Request):
    """RAG 検索テストページ"""
    session = request.cookies.get("session_token")
    if not session:
        return HTMLResponse('<script>window.location.href="/";</script>')
    return templates.TemplateResponse(name="rag_search.html", context={}, request=request)


@app.get("/skill-tree", response_class=HTMLResponse)
async def skill_tree_page(request: Request):
    """スキルツリービューア ページ"""
    session = request.cookies.get("session_token")
    if not session:
        return HTMLResponse('<script>window.location.href="/";</script>')
    return templates.TemplateResponse(name="skill_tree.html", context={}, request=request)


@app.get("/api/skill-tree")
async def api_skill_tree(request: Request):
    """スキルツリー構造 + チャンクテキストを返す"""
    session = check_auth(request)
    if not session:
        raise HTTPException(status_code=401, detail="認証が必要です")

    working_dir = C2S_ROOT / active_collection
    skill_meta_path  = working_dir / "skill_meta.json"
    chunk_index_path = working_dir / "chunk_index.json"

    if not skill_meta_path.exists():
        return JSONResponse({"skills": [], "total_chunks": 0})

    skills = json.loads(skill_meta_path.read_text(encoding="utf-8"))

    # チャンクテキストを (doc_id, chunk_idx) → text の辞書に
    chunk_map: dict[tuple[str, int], str] = {}
    if chunk_index_path.exists():
        for entry in json.loads(chunk_index_path.read_text(encoding="utf-8")):
            chunk_map[(entry["doc_id"], entry["chunk_idx"])] = entry["text"]

    # 各 sub_skill に chunk テキストを付加
    for skill in skills:
        for sub in skill.get("sub_skills", []):
            sub_dir = working_dir / sub["dir"]
            ids_path = sub_dir / "chunk_ids.json"
            sub["chunks"] = []
            if ids_path.exists():
                for ref in json.loads(ids_path.read_text(encoding="utf-8")):
                    text = chunk_map.get((ref["doc_id"], ref["chunk_idx"]), "")
                    sub["chunks"].append({
                        "doc_id":    ref["doc_id"],
                        "chunk_idx": ref["chunk_idx"],
                        "text":      text,
                    })

    total_chunks = sum(
        len(sub.get("chunks", []))
        for skill in skills
        for sub in skill.get("sub_skills", [])
    )

    return JSONResponse({"skills": skills, "total_chunks": total_chunks})


# ─── メイン ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.Config.get_app_host(),
        port=config.Config.get_app_port(),
        reload=True,
    )
