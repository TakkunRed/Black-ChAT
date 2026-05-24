"""
Corpus2Skill モジュール（LM Studio 版）

Corpus2Skill の概念 (https://github.com/dukesun99/Corpus2Skill) を
LM Studio（OpenAI 互換 API）+ SentenceTransformer で独自実装したもの。

処理フロー:
  ① [コンパイル] ドキュメント登録時（バックグラウンド実行）
       ドキュメント → チャンク分割
       → SentenceTransformer でエンベディング
       → K-means で 2 階層クラスタリング（スキル → グループ）
       → LM Studio でクラスタラベリング（トピック名・要約）
       → スキルツリーをファイルに保存 (SKILL.md / INDEX.md)

  ② [検索] チャット時
       クエリ → SentenceTransformer でエンベディング
       → 全チャンクとのコサイン類似度計算
       → 上位チャンクのテキストをコンテキストとして返す
"""

import asyncio
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".docx"}


# ── ユーティリティ ─────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """ディレクトリ名として使えるスラグに変換（日本語含む）"""
    slug = re.sub(r'[^\w぀-鿿゠-ヿ]', '_', text)
    return slug.strip('_')[:25] or "group"


def _chunk_text(text: str, max_chars: int = 800, overlap_chars: int = 0) -> list[str]:
    """
    段落（空行区切り）を優先してテキストをチャンクに分割する。

    段落が max_chars 以内 → そのまま 1 チャンク
    段落が長い → 。！？で文分割して積み重ね
    overlap_chars > 0 → 隣接チャンク間に重複テキストを付加
    """
    paragraphs = re.split(r'\n{2,}', text)
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) <= max_chars:
            if current_len + len(para) > max_chars and current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts, current_len = [], 0
            current_parts.append(para)
            current_len += len(para)
        else:
            if current_parts:
                chunks.append('\n\n'.join(current_parts))
                current_parts, current_len = [], 0
            sents = re.split(r'(?<=[。！？])\s*', para)
            buf: list[str] = []
            buf_len = 0
            for sent in sents:
                sent = sent.strip()
                if not sent:
                    continue
                if buf_len + len(sent) > max_chars and buf:
                    chunks.append(''.join(buf))
                    buf, buf_len = [], 0
                buf.append(sent)
                buf_len += len(sent)
            if buf:
                chunks.append(''.join(buf))

    if current_parts:
        chunks.append('\n\n'.join(current_parts))

    base_chunks = chunks if chunks else [text[:max_chars]]

    # オーバーラップ付加：前チャンク末尾の完全な文を次チャンクの冒頭に追加
    if overlap_chars <= 0 or len(base_chunks) <= 1:
        return base_chunks
    overlapped: list[str] = [base_chunks[0]]
    for i in range(1, len(base_chunks)):
        tail_raw = base_chunks[i - 1][-overlap_chars:]
        # 文字数で切り取ると文の途中から始まるため、
        # 最初の句点・感嘆符・疑問符の直後から使うことで文頭に揃える
        m = re.search(r'(?<=[。！？])\s*', tail_raw)
        tail = tail_raw[m.end():].strip() if m else ""
        if tail:
            overlapped.append(tail + "\n" + base_chunks[i])
        else:
            # 句点が見つからない（tail全体が1文）場合はオーバーラップなし
            overlapped.append(base_chunks[i])
    return overlapped


# ── キーワードブースト ─────────────────────────────────────────────────────

_KEYWORD_BOOST   = 2.5   # クエリ語句が完全一致した場合の乗数
_CANDIDATE_MULTI = 5     # ブースト前に取得する候補数の倍率（top_k × この値）


def _keyword_boost_factor(query: str, text: str) -> float:
    """
    クエリの語句がチャンクテキストに含まれる割合に応じてブースト係数を返す。

    - クエリ全体がそのまま含まれる → _KEYWORD_BOOST
    - 各語が部分的に含まれる      → 線形補間
    - まったく含まれない          → 1.0（ブーストなし）
    """
    q = query.strip()
    if not q:
        return 1.0
    text_l = text.lower()
    q_l    = q.lower()

    # クエリ全体が一致 → フルブースト
    if q_l in text_l:
        return _KEYWORD_BOOST

    # スペース・句読点で分割して各語を確認
    words = [w for w in re.split(r'[\s　、。,.!?！？「」『』【】]+', q_l) if len(w) >= 2]
    if not words:
        return 1.0
    hit = sum(1 for w in words if w in text_l)
    if hit == 0:
        return 1.0
    return 1.0 + (_KEYWORD_BOOST - 1.0) * hit / len(words)


# ── Corpus2SkillManager ────────────────────────────────────────────────────

class Corpus2SkillManager:
    """
    Corpus2Skill スタイルの階層スキルツリー RAG マネージャー。

    コンパイル（スキルツリー構築）はバックグラウンドで非同期実行される。
    進捗は get_compile_status() で取得可能。
    """

    def __init__(
        self,
        working_dir: str = "./c2s_db",
        lm_studio_base_url: str = "http://127.0.0.1:1234/v1",
        llm_model: str = "",
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        api_key: str = "lm-studio",
        max_top_skills: int = 6,
        branching_factor: int = 4,
        chunk_max_chars: int = 800,
        overlap_chars: int = 0,
    ):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir = self.working_dir / "skills"
        self.skills_dir.mkdir(exist_ok=True)
        self.search_mode = "corpus2skill"

        self._docs_index_file  = self.working_dir / "_indexed_docs.json"
        self._documents_file   = self.working_dir / "documents.json"
        self._chunk_index_file = self.working_dir / "chunk_index.json"
        self._embeddings_file  = self.working_dir / "embeddings.npy"
        self._skill_meta_file  = self.working_dir / "skill_meta.json"
        self._doc_metadata_file = self.working_dir / "_doc_metadata.json"
        self._compile_info_file = self.working_dir / "_compile_info.json"
        self._doc_embeddings_dir = self.working_dir / "doc_embeddings"
        self._doc_embeddings_dir.mkdir(exist_ok=True)

        self._config = {
            "lm_studio_base_url": lm_studio_base_url,
            "llm_model": llm_model or "",
            "embed_model": embed_model,
            "api_key": api_key or "lm-studio",
            "max_top_skills": max(1, max_top_skills),
            "branching_factor": max(1, branching_factor),
            "chunk_max_chars": max(100, chunk_max_chars),
            "overlap_chars": max(0, overlap_chars),
        }

        self._docs: dict[str, str] = self._load_docs_index()
        self._encoder = None

        # コンパイル進捗状態
        self._compile_status: dict = {
            "state": "idle",       # idle | compiling | done | error
            "current_skill": 0,
            "total_skills": 0,
            "message": "待機中",
        }

    # ── エンコーダー（遅延ロード） ──────────────────────────────────────

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self._config["embed_model"]
                print(f"[Corpus2Skill] エンベディングモデルをロード中: {model_name}")
                self._encoder = SentenceTransformer(model_name)
                print("[Corpus2Skill] エンベディングモデルのロード完了")
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers がインストールされていません。\n"
                    "uv add sentence-transformers を実行してください。"
                )
        return self._encoder

    # ── LLM 自動検出 ──────────────────────────────────────────────────

    async def _detect_llm_model(self) -> str:
        cfg = self._config
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{cfg['lm_studio_base_url']}/models",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                )
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    model_id = models[0]["id"]
                    print(f"[Corpus2Skill] LLM モデルを自動検出: {model_id}")
                    return model_id
        except Exception as e:
            print(f"[Corpus2Skill] LLM モデルの自動検出失敗: {e}")
        return ""

    # ── 初期化 ─────────────────────────────────────────────────────────

    async def initialize(self):
        if not self._config["llm_model"]:
            detected = await self._detect_llm_model()
            if detected:
                self._config["llm_model"] = detected
            else:
                raise RuntimeError(
                    "LM Studio に接続できないか、モデルがロードされていません。"
                )

        try:
            self._get_encoder()
        except Exception as e:
            print(f"[Corpus2Skill] エンベディングモデルのロード失敗（検索時に再試行）: {e}")

        print(
            f"[Corpus2Skill] 初期化完了 - "
            f"ドキュメント数: {len(self._docs)}, "
            f"モデル: {self._config['llm_model']}"
        )

    # ── LLM 呼び出し ──────────────────────────────────────────────────

    async def _llm_call(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        cfg = self._config
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if cfg["llm_model"]:
            payload["model"] = cfg["llm_model"]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{cfg['lm_studio_base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {cfg['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Corpus2Skill] LLM 呼び出しエラー: {e}")
            return ""

    # ── ドキュメントインデックス管理 ─────────────────────────────────

    def _load_docs_index(self) -> dict[str, str]:
        if self._docs_index_file.exists():
            try:
                return json.loads(self._docs_index_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_docs_index(self):
        self._docs_index_file.write_text(
            json.dumps(self._docs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_documents(self) -> dict[str, str]:
        if self._documents_file.exists():
            try:
                return json.loads(self._documents_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_documents(self, documents: dict[str, str]):
        self._documents_file.write_text(
            json.dumps(documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_doc_metadata(self) -> dict[str, dict]:
        if self._doc_metadata_file.exists():
            try:
                return json.loads(self._doc_metadata_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_doc_metadata(self, metadata: dict[str, dict]):
        self._doc_metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── ドキュメント別埋め込みキャッシュ ────────────────────────────

    def _get_doc_embed_paths(self, doc_name: str) -> tuple[Path, Path]:
        # with_suffix() は既存の拡張子を置換してしまうため、
        # "report.txt" と "report.pdf" が同じパスに衝突する。
        # 拡張子を含むファイル名全体をサフィックスとして付加することで衝突を回避する。
        safe = re.sub(r'[^\w\-.]', '_', doc_name)
        return (
            self._doc_embeddings_dir / (safe + ".chunks.json"),
            self._doc_embeddings_dir / (safe + ".emb.npy"),
        )

    def _load_doc_embed_cache(self, doc_name: str) -> tuple[list[dict], "np.ndarray"] | None:
        json_path, npy_path = self._get_doc_embed_paths(doc_name)
        if not json_path.exists() or not npy_path.exists():
            return None
        try:
            chunks = json.loads(json_path.read_text(encoding="utf-8"))
            embeddings = np.load(str(npy_path))
            return chunks, embeddings
        except Exception:
            return None

    def _save_doc_embed_cache(self, doc_name: str, chunks: list[dict], embeddings: "np.ndarray"):
        json_path, npy_path = self._get_doc_embed_paths(doc_name)
        json_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        np.save(str(npy_path), embeddings)

    def _delete_doc_embed_cache(self, doc_name: str):
        for path in self._get_doc_embed_paths(doc_name):
            if path.exists():
                path.unlink()

    # ── コンパイル情報（インクリメンタル用） ─────────────────────────

    def _load_compile_info(self) -> dict:
        if self._compile_info_file.exists():
            try:
                return json.loads(self._compile_info_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_compile_info(self):
        info = {
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "params": {
                "chunk_max_chars": self._config["chunk_max_chars"],
                "overlap_chars": self._config["overlap_chars"],
            },
        }
        self._compile_info_file.write_text(
            json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _need_full_recompile(self) -> bool:
        """チャンクパラメータが変更されていれば True（キャッシュ無効化が必要）"""
        info = self._load_compile_info()
        if not info:
            return False
        saved = info.get("params", {})
        current = {
            "chunk_max_chars": self._config["chunk_max_chars"],
            "overlap_chars": self._config["overlap_chars"],
        }
        return saved != current

    def _get_last_compiled_at(self) -> "datetime | None":
        info = self._load_compile_info()
        ts = info.get("compiled_at", "")
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except Exception:
                pass
        return None

    # ── ファイル読み込み ─────────────────────────────────────────────

    def _read_txt(self, file_path: Path) -> str:
        return file_path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, file_path: Path) -> str:
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            raise ValueError(f"PDF読み込みエラー: {e}") from e

    def _read_md(self, file_path: Path) -> str:
        """Markdown ファイルを読み込み、マークアップ記号を除去してプレーンテキストに変換する"""
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        # コードブロック除去
        text = re.sub(r'```[\s\S]*?```', '', text)
        # インラインコード除去
        text = re.sub(r'`[^`\n]+`', '', text)
        # 画像除去
        text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', '', text)
        # リンク → テキスト部分のみ残す
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        # 見出し記号除去
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        # 太字・斜体除去
        text = re.sub(r'\*\*([^*\n]+)\*\*', r'\1', text)
        text = re.sub(r'__([^_\n]+)__', r'\1', text)
        text = re.sub(r'\*([^*\n]+)\*', r'\1', text)
        text = re.sub(r'_([^_\n]+)_', r'\1', text)
        # 箇条書き記号除去
        text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
        # 番号付きリスト記号除去
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        # 引用記号除去
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        # テーブル行除去
        text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
        # 水平線除去
        text = re.sub(r'^[-\*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        # 連続空行を整理
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _read_docx(self, file_path: Path) -> str:
        """Word（.docx）ファイルを読み込む"""
        try:
            import docx
            doc = docx.Document(str(file_path))
            parts: list[str] = []
            for para in doc.paragraphs:
                stripped = para.text.strip()
                if stripped:
                    parts.append(stripped)
            # テーブルのテキストも取得
            for table in doc.tables:
                for row in table.rows:
                    row_texts = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                    if row_texts:
                        parts.append(" | ".join(row_texts))
            return "\n\n".join(parts)
        except Exception as e:
            raise ValueError(f"Docx読み込みエラー: {e}") from e

    def _read_file(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext == ".txt":
            return self._read_txt(file_path)
        if ext == ".pdf":
            return self._read_pdf(file_path)
        if ext == ".md":
            return self._read_md(file_path)
        if ext == ".docx":
            return self._read_docx(file_path)
        raise ValueError(f"未対応の形式: {ext}")

    # ── クラスタラベリング ────────────────────────────────────────────

    async def _label_cluster(
        self,
        texts: list[str],
        max_sample: int = 3,
    ) -> tuple[str, str]:
        sample = texts[:max_sample]
        combined = "\n\n---\n\n".join(
            f"[文書 {i + 1}]\n{t[:400]}" for i, t in enumerate(sample)
        )

        system = (
            "あなたはテキスト分析の専門家です。"
            "与えられた文書群を読んで、グループのトピック名と要約を生成してください。"
        )
        prompt = (
            "以下の文書群を読み、このグループを代表するトピック名（5〜20文字）と\n"
            "要約（1〜2文、50文字以内）を生成してください。\n\n"
            "必ずこの形式で出力してください（他のテキストは不要）:\n"
            "TOPIC: <トピック名>\n"
            "SUMMARY: <要約>\n\n"
            f"文書:\n{combined}"
        )

        result = await self._llm_call(prompt, system=system, temperature=0.1)

        topic = "未分類"
        summary = ""
        for line in result.split('\n'):
            line = line.strip()
            if line.startswith("TOPIC:"):
                topic = line[6:].strip() or "未分類"
            elif line.startswith("SUMMARY:"):
                summary = line[8:].strip()

        if not topic or topic == "未分類":
            topic = texts[0][:20].replace('\n', ' ') + "…"
        if not summary:
            summary = texts[0][:80].replace('\n', ' ')

        return topic, summary

    # ── コンパイル（スキルツリー構築） ──────────────────────────────

    async def _compile(self) -> None:
        """
        全ドキュメントからスキルツリーを構築する。
        進捗は self._compile_status に随時書き込まれる。
        """
        self._compile_status = {
            "state": "compiling",
            "current_skill": 0,
            "total_skills": 0,
            "message": "チャンク分割・エンベディング中...",
        }

        try:
            from sklearn.cluster import KMeans
        except ImportError:
            self._compile_status = {
                "state": "error", "current_skill": 0, "total_skills": 0,
                "message": "scikit-learn がインストールされていません",
            }
            raise RuntimeError("scikit-learn がインストールされていません。")

        documents = self._load_documents()
        if not documents:
            self._compile_status = {
                "state": "idle", "current_skill": 0, "total_skills": 0,
                "message": "待機中",
            }
            return

        print(f"[Corpus2Skill] コンパイル開始: {len(documents)} ドキュメント")

        # ① インクリメンタルチャンク分割 + エンベディング
        max_chars = self._config["chunk_max_chars"]
        overlap_chars = self._config.get("overlap_chars", 0)

        # チャンクパラメータ変更時はキャッシュを全削除（フルリビルド）
        if self._need_full_recompile():
            print("[Corpus2Skill] チャンクパラメータが変更されました。埋め込みキャッシュをクリアします...")
            if self._doc_embeddings_dir.exists():
                shutil.rmtree(self._doc_embeddings_dir)
            self._doc_embeddings_dir.mkdir(parents=True)

        last_compiled = self._get_last_compiled_at()
        metadata = self._load_doc_metadata()
        self._compile_status["message"] = "インクリメンタルエンベディング中..."
        encoder = self._get_encoder()

        all_chunks: list[dict] = []
        all_embeddings_list: list["np.ndarray"] = []
        cached_docs, new_docs = 0, 0

        for doc_id, text in documents.items():
            # キャッシュが有効かチェック（最終コンパイル後に変更されていないか）
            cache_hit = False
            if last_compiled:
                doc_meta = metadata.get(doc_id, {})
                updated_at_str = doc_meta.get("updated_at", "")
                if updated_at_str:
                    try:
                        updated_at = datetime.fromisoformat(updated_at_str)
                        # naive datetime（タイムゾーンなし）と aware datetime を
                        # 比較すると TypeError になるため、naive の場合は UTC に統一する
                        if updated_at.tzinfo is None:
                            updated_at = updated_at.replace(tzinfo=timezone.utc)
                        if updated_at <= last_compiled:
                            cached = self._load_doc_embed_cache(doc_id)
                            if cached:
                                doc_chunks, doc_embs = cached
                                all_chunks.extend(doc_chunks)
                                all_embeddings_list.append(doc_embs)
                                cached_docs += 1
                                cache_hit = True
                    except Exception:
                        pass

            if cache_hit:
                continue

            # 新規エンベディング
            print(f"[Corpus2Skill] {doc_id}: エンベディング中...")
            chunk_texts = _chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
            doc_chunks = [
                {"doc_id": doc_id, "chunk_idx": ci, "text": ct}
                for ci, ct in enumerate(chunk_texts)
            ]
            doc_embs = encoder.encode(
                [c["text"] for c in doc_chunks],
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32,
            )
            self._save_doc_embed_cache(doc_id, doc_chunks, doc_embs)
            all_chunks.extend(doc_chunks)
            all_embeddings_list.append(doc_embs)
            new_docs += 1

        chunks = all_chunks
        cache_info = f"(キャッシュ:{cached_docs}件, 新規エンベディング:{new_docs}件)"
        print(f"[Corpus2Skill] チャンク数: {len(chunks)} {cache_info}")

        if not chunks:
            self._compile_status = {
                "state": "error", "current_skill": 0, "total_skills": 0,
                "message": "チャンクが生成されませんでした",
            }
            return

        embeddings = np.vstack(all_embeddings_list)
        print(f"[Corpus2Skill] エンベディング完了: shape={embeddings.shape}")

        # ③ トップレベルクラスタリング
        self._compile_status["message"] = "クラスタリング中..."
        max_top = self._config["max_top_skills"]
        n_top = min(max_top, max(1, len(chunks) // 2))

        if len(chunks) <= n_top:
            top_labels = list(range(len(chunks)))
            n_top = len(chunks)
        else:
            km_top = KMeans(n_clusters=n_top, random_state=42, n_init=10)
            top_labels = km_top.fit_predict(embeddings).tolist()

        print(f"[Corpus2Skill] トップレベルクラスタ数: {n_top}")
        self._compile_status["total_skills"] = n_top

        # ④ 既存スキルツリーをクリアして再構築
        if self.skills_dir.exists():
            shutil.rmtree(self.skills_dir)
        self.skills_dir.mkdir(parents=True)

        skill_meta: list[dict] = []

        for top_idx in range(n_top):
            self._compile_status.update({
                "current_skill": top_idx + 1,
                "message": f"スキル {top_idx + 1}/{n_top} をラベリング中...",
            })

            top_chunk_indices = [i for i, lb in enumerate(top_labels) if lb == top_idx]
            if not top_chunk_indices:
                continue

            top_chunks = [chunks[i] for i in top_chunk_indices]
            top_texts = [c["text"] for c in top_chunks]
            top_embs = embeddings[top_chunk_indices]

            # ⑤ スキルラベリング
            print(f"[Corpus2Skill] スキル {top_idx + 1}/{n_top} のラベリング中...")
            topic, summary = await self._label_cluster(top_texts)

            skill_dir_name = f"skill_{top_idx:02d}_{_slugify(topic)}"
            skill_dir = self.skills_dir / skill_dir_name
            skill_dir.mkdir(parents=True)

            # ⑥ サブクラスタリング
            n_sub = self._config["branching_factor"]
            n_sub_actual = min(n_sub, max(1, len(top_chunks) // 2))

            if len(top_chunks) <= 1:
                sub_labels = [0] * len(top_chunks)
                n_sub_actual = 1
            else:
                km_sub = KMeans(n_clusters=n_sub_actual, random_state=42, n_init=5)
                sub_labels = km_sub.fit_predict(top_embs).tolist()

            sub_info: list[dict] = []

            for sub_idx in range(n_sub_actual):
                sub_chunk_indices = [i for i, lb in enumerate(sub_labels) if lb == sub_idx]
                if not sub_chunk_indices:
                    continue

                sub_chunks = [top_chunks[i] for i in sub_chunk_indices]
                sub_texts = [c["text"] for c in sub_chunks]

                if len(sub_texts) > 1:
                    sub_topic, sub_summary = await self._label_cluster(sub_texts, max_sample=2)
                else:
                    sub_topic = sub_texts[0][:20].replace('\n', ' ')
                    sub_summary = sub_texts[0][:100].replace('\n', ' ')

                sub_dir = skill_dir / f"group_{sub_idx:02d}_{_slugify(sub_topic)}"
                sub_dir.mkdir(parents=True)

                chunk_ids = [
                    {"doc_id": c["doc_id"], "chunk_idx": c["chunk_idx"]}
                    for c in sub_chunks
                ]
                (sub_dir / "chunk_ids.json").write_text(
                    json.dumps(chunk_ids, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                doc_list_md = "\n".join(
                    f"- [{c['doc_id']}] chunk {c['chunk_idx']}: {c['text'][:60].replace(chr(10), ' ')}…"
                    for c in sub_chunks
                )
                (sub_dir / "INDEX.md").write_text(
                    f"# {sub_topic}\n\n{sub_summary}\n\n## 含まれる文書\n\n{doc_list_md}\n",
                    encoding="utf-8",
                )

                sub_info.append({
                    "sub_idx": sub_idx,
                    "topic": sub_topic,
                    "summary": sub_summary,
                    "dir": str(sub_dir.relative_to(self.working_dir)),
                    "chunk_count": len(sub_chunks),
                })

            sub_list_md = "\n".join(
                f"- **{s['topic']}**: {s['summary'][:80]} （{s['chunk_count']} チャンク）"
                for s in sub_info
            )
            doc_ids = sorted({c["doc_id"] for c in top_chunks})
            (skill_dir / "SKILL.md").write_text(
                f"# {topic}\n\n"
                f"{summary}\n\n"
                f"## サブグループ\n\n{sub_list_md}\n\n"
                f"## 含まれるドキュメント\n\n"
                + "\n".join(f"- {d}" for d in doc_ids) + "\n",
                encoding="utf-8",
            )

            skill_meta.append({
                "idx": top_idx,
                "topic": topic,
                "summary": summary,
                "dir": str(skill_dir.relative_to(self.working_dir)),
                "chunk_count": len(top_chunks),
                "sub_skills": sub_info,
            })

        # ⑦ インデックス保存
        self._compile_status["message"] = "インデックスを保存中..."
        self._chunk_index_file.write_text(
            json.dumps(chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(str(self._embeddings_file), embeddings)
        self._skill_meta_file.write_text(
            json.dumps(skill_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._save_compile_info()
        self._compile_status = {
            "state": "done",
            "current_skill": n_top,
            "total_skills": n_top,
            "message": f"完了: {n_top} スキル / {len(chunks)} チャンク",
        }
        print(f"[Corpus2Skill] コンパイル完了: {n_top} スキル, {len(chunks)} チャンク")

    async def _compile_safe(self) -> None:
        """バックグラウンドコンパイルのラッパー（例外を状態に記録する）"""
        # ガードチェックと状態更新をアトミックに行う。
        # asyncio.create_task() で複数タスクがスケジュールされても、
        # 最初に実行されたタスクが即座に "compiling" にセットするため、
        # 後続タスクはここで弾かれる（await の間に他タスクが割り込めるため安全）。
        if self._compile_status["state"] == "compiling":
            print("[Corpus2Skill] コンパイルは既に実行中です。スキップします。")
            return
        self._compile_status["state"] = "compiling"  # 先行してガードを立てる
        try:
            await self._compile()
        except Exception as e:
            print(f"[Corpus2Skill] バックグラウンドコンパイルエラー: {e}")
            self._compile_status = {
                "state": "error",
                "current_skill": 0,
                "total_skills": 0,
                "message": f"エラー: {e}",
            }

    # ── コンパイル状態・パラメータ ────────────────────────────────────

    def get_compile_status(self) -> dict:
        """コンパイルの進捗状態を返す"""
        return dict(self._compile_status)

    async def set_compile_params(
        self,
        max_top_skills: int | None = None,
        branching_factor: int | None = None,
        chunk_max_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> None:
        """スキルツリー構築パラメータを更新する（再コンパイルは行わない）"""
        if max_top_skills is not None:
            self._config["max_top_skills"] = max(1, int(max_top_skills))
        if branching_factor is not None:
            self._config["branching_factor"] = max(1, int(branching_factor))
        if chunk_max_chars is not None:
            self._config["chunk_max_chars"] = max(100, int(chunk_max_chars))
        if overlap_chars is not None:
            self._config["overlap_chars"] = max(0, int(overlap_chars))
        print(
            f"[Corpus2Skill] パラメータ更新: "
            f"max_top_skills={self._config['max_top_skills']}, "
            f"branching_factor={self._config['branching_factor']}, "
            f"chunk_max_chars={self._config['chunk_max_chars']}, "
            f"overlap_chars={self._config['overlap_chars']}"
        )

    async def recompile(self) -> dict:
        """現在のドキュメントでスキルツリーをバックグラウンド再構築する"""
        if self._compile_status["state"] == "compiling":
            return {"started": False, "message": "コンパイルは既に実行中です"}
        if not self._load_documents():
            return {"started": False, "message": "ドキュメントが登録されていません"}
        asyncio.create_task(self._compile_safe())
        return {"started": True, "message": "再コンパイルを開始しました"}

    # ── インデックス操作 ─────────────────────────────────────────────

    async def add_document(self, file_path: str | Path) -> dict:
        """ドキュメントを追加（または更新）し、バックグラウンドでスキルツリーを再構築する"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"success": False, "file": file_path.name, "message": "ファイルが見つかりません"}
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return {"success": False, "file": file_path.name, "message": f"未対応の形式: {file_path.suffix}"}

        try:
            text = self._read_file(file_path)
        except ValueError as e:
            return {"success": False, "file": file_path.name, "message": str(e)}

        if not text.strip():
            return {"success": False, "file": file_path.name, "message": "テキストを抽出できませんでした"}

        is_update = file_path.name in self._docs

        documents = self._load_documents()
        documents[file_path.name] = text
        self._save_documents(documents)
        self._docs[file_path.name] = str(file_path)
        self._save_docs_index()

        # メタデータ更新
        metadata = self._load_doc_metadata()
        now = datetime.now(timezone.utc).isoformat()
        size_bytes = file_path.stat().st_size
        if is_update and file_path.name in metadata:
            metadata[file_path.name]["updated_at"] = now
            metadata[file_path.name]["size_bytes"] = size_bytes
        else:
            metadata[file_path.name] = {
                "uploaded_at": now,
                "updated_at": now,
                "size_bytes": size_bytes,
                "ext": file_path.suffix.lower(),
            }
        self._save_doc_metadata(metadata)

        asyncio.create_task(self._compile_safe())

        action = "更新しました" if is_update else "追加しました"
        return {
            "success": True,
            "file": file_path.name,
            "message": f"{action}。バックグラウンドでスキルツリーを構築中です。",
            "compiling": True,
            "replaced": is_update,
        }

    async def add_directory(self, dir_path: str | Path) -> list[dict]:
        """ディレクトリ内の全対応ファイルをまとめて追加し、1 回だけコンパイルする"""
        dir_path = Path(dir_path)
        results = []
        added: list[tuple[Path, str]] = []

        for ext in SUPPORTED_EXTENSIONS:
            for fp in dir_path.rglob(f"*{ext}"):
                try:
                    text = self._read_file(fp)
                except Exception as e:
                    results.append({"success": False, "file": fp.name, "message": str(e)})
                    continue
                if not text.strip():
                    results.append({"success": False, "file": fp.name, "message": "テキストを抽出できませんでした"})
                    continue
                added.append((fp, text))
                results.append({"success": True, "file": fp.name, "message": "受け付けました"})

        if added:
            documents = self._load_documents()
            metadata = self._load_doc_metadata()
            now = datetime.now(timezone.utc).isoformat()
            for fp, text in added:
                is_update = fp.name in self._docs
                documents[fp.name] = text
                self._docs[fp.name] = str(fp)
                size_bytes = fp.stat().st_size
                if is_update and fp.name in metadata:
                    metadata[fp.name]["updated_at"] = now
                    metadata[fp.name]["size_bytes"] = size_bytes
                else:
                    metadata[fp.name] = {
                        "uploaded_at": now,
                        "updated_at": now,
                        "size_bytes": size_bytes,
                        "ext": fp.suffix.lower(),
                    }
            self._save_documents(documents)
            self._save_docs_index()
            self._save_doc_metadata(metadata)
            asyncio.create_task(self._compile_safe())

        return results

    async def delete_document(self, file_name: str) -> dict:
        """ドキュメントを削除し、残りのドキュメントでスキルツリーを再構築する"""
        if file_name not in self._docs:
            return {"success": False, "message": "該当ファイルが見つかりません"}

        del self._docs[file_name]
        documents = self._load_documents()
        documents.pop(file_name, None)
        self._save_documents(documents)
        self._save_docs_index()
        metadata = self._load_doc_metadata()
        metadata.pop(file_name, None)
        self._save_doc_metadata(metadata)
        self._delete_doc_embed_cache(file_name)

        if self._docs:
            asyncio.create_task(self._compile_safe())
            return {
                "success": True,
                "message": f"{file_name} を削除しました。スキルツリーを再構築中です。",
                "compiling": True,
            }
        else:
            await self._clear_skill_data()
            return {"success": True, "message": f"{file_name} を削除しました。", "compiling": False}

    async def _clear_skill_data(self) -> None:
        """スキルツリーのデータをすべてクリアする（ドキュメントインデックスは保持）"""
        if self.skills_dir.exists():
            shutil.rmtree(self.skills_dir)
        self.skills_dir.mkdir(parents=True)
        for f in [self._chunk_index_file, self._embeddings_file, self._skill_meta_file]:
            if f.exists():
                f.unlink()
        self._compile_status = {
            "state": "idle", "current_skill": 0, "total_skills": 0, "message": "待機中",
        }

    async def clear(self) -> dict:
        """全ドキュメントとスキルツリーをクリアする"""
        count = len(self._docs)
        self._docs = {}
        self._save_docs_index()
        self._save_documents({})
        self._save_doc_metadata({})
        # ドキュメント別埋め込みキャッシュも全削除
        if self._doc_embeddings_dir.exists():
            shutil.rmtree(self._doc_embeddings_dir)
        self._doc_embeddings_dir.mkdir(parents=True)
        if self._compile_info_file.exists():
            self._compile_info_file.unlink()
        await self._clear_skill_data()
        return {"cleared_documents": count}

    # ── 検索 ─────────────────────────────────────────────────────────

    async def search(self, query: str, mode: str | None = None) -> list[dict]:
        """コサイン類似度でチャンクを検索し、結合コンテキストを返す（チャット用）"""
        if not self._docs:
            print("[Corpus2Skill] 検索スキップ: ドキュメントが登録されていません")
            return []

        if not self._chunk_index_file.exists() or not self._embeddings_file.exists():
            print("[Corpus2Skill] スキルツリーが未構築です（コンパイルが必要）")
            return []

        try:
            chunks: list[dict] = json.loads(
                self._chunk_index_file.read_text(encoding="utf-8")
            )
            embeddings = np.load(str(self._embeddings_file))

            encoder = self._get_encoder()
            query_emb = encoder.encode([query], normalize_embeddings=True)[0]
            sims = embeddings @ query_emb

            # ── ハイブリッド検索: セマンティック類似度 + キーワードブースト ──
            # 1) コサイン類似度だけで広めに候補を取得
            display_k   = 12
            candidate_k = min(display_k * _CANDIDATE_MULTI, len(chunks))
            candidate_indices = np.argsort(sims)[::-1][:candidate_k]

            # 2) 各候補にキーワードブーストを適用して再ランキング
            boosted: list[tuple[float, float, int]] = []  # (boosted, raw, idx)
            for idx in candidate_indices:
                raw   = float(sims[idx])
                boost = _keyword_boost_factor(query, chunks[int(idx)]["text"])
                boosted.append((raw * boost, raw, int(idx)))
            boosted.sort(key=lambda x: x[0], reverse=True)

            skill_overview = ""
            if self._skill_meta_file.exists():
                try:
                    skill_meta: list[dict] = json.loads(
                        self._skill_meta_file.read_text(encoding="utf-8")
                    )
                    lines = ["【スキルツリー概要】"]
                    for sm in skill_meta:
                        lines.append(f"・{sm['topic']}: {sm['summary'][:80]}")
                    skill_overview = "\n".join(lines) + "\n\n"
                except Exception:
                    pass

            seen: set[str] = set()
            context_parts: list[str] = ["【関連ドキュメント】"]
            hit_chunks: list[dict] = []

            for boosted_score, raw_score, raw_idx in boosted[:display_k]:
                if boosted_score < 0.15:
                    continue
                chunk = chunks[raw_idx]
                key = f"{chunk['doc_id']}_{chunk['chunk_idx']}"
                if key in seen:
                    continue
                seen.add(key)
                context_parts.append(
                    f"\n[出典: {chunk['doc_id']} | スコア: {boosted_score:.2f}]\n{chunk['text']}"
                )
                hit_chunks.append({
                    "doc_id":    chunk["doc_id"],
                    "chunk_idx": chunk["chunk_idx"],
                    "score":     round(boosted_score, 3),
                    "text":      chunk["text"][:300],   # UI プレビュー用
                })

            if len(context_parts) <= 1:
                print("[Corpus2Skill] 関連コンテキストが見つかりませんでした")
                return []

            context = skill_overview + "\n".join(context_parts)
            print(
                f"[Corpus2Skill] 検索成功: "
                f"{len(context_parts) - 1} チャンク, {len(context)} 文字"
            )
            # 最高チャンクのコサイン類似度をトップレベルスコアとして使う
            best_score = hit_chunks[0]["score"] if hit_chunks else 0.0
            return [{"content": context, "source": "Corpus2Skill", "score": best_score, "chunks": hit_chunks}]

        except Exception as e:
            print(f"[Corpus2Skill] 検索エラー: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def search_chunks(self, query: str, top_k: int = 10) -> list[dict]:
        """
        個別チャンクの検索結果を返す（検索テスト・デバッグ用）。

        Returns:
            [{"doc_id": str, "chunk_idx": int, "text": str, "score": float}]
        """
        if not self._chunk_index_file.exists() or not self._embeddings_file.exists():
            return []

        try:
            chunks: list[dict] = json.loads(
                self._chunk_index_file.read_text(encoding="utf-8")
            )
            embeddings = np.load(str(self._embeddings_file))

            encoder = self._get_encoder()
            query_emb = encoder.encode([query], normalize_embeddings=True)[0]
            sims = embeddings @ query_emb

            # ── ハイブリッド検索: セマンティック類似度 + キーワードブースト ──
            candidate_k = min(top_k * _CANDIDATE_MULTI, len(chunks))
            candidate_indices = np.argsort(sims)[::-1][:candidate_k]

            boosted: list[tuple[float, float, int]] = []
            for idx in candidate_indices:
                raw   = float(sims[idx])
                boost = _keyword_boost_factor(query, chunks[int(idx)]["text"])
                boosted.append((raw * boost, raw, int(idx)))
            boosted.sort(key=lambda x: x[0], reverse=True)

            results = []
            for boosted_score, raw_score, raw_idx in boosted[:top_k]:
                if boosted_score < 0.15:
                    continue
                chunk = chunks[raw_idx]
                results.append({
                    "doc_id":    chunk["doc_id"],
                    "chunk_idx": chunk["chunk_idx"],
                    "text":      chunk["text"],
                    "score":     round(boosted_score, 4),
                    "raw_score": round(raw_score, 4),  # デバッグ用: ブースト前スコア
                })
            return results

        except Exception as e:
            print(f"[Corpus2Skill] search_chunks エラー: {e}")
            return []

    # ── 状態・設定 ────────────────────────────────────────────────────

    def get_status(self) -> dict:
        skill_count = 0
        if self.skills_dir.exists():
            skill_count = sum(1 for p in self.skills_dir.iterdir() if p.is_dir())

        chunk_count = 0
        if self._chunk_index_file.exists():
            try:
                chunk_count = len(json.loads(
                    self._chunk_index_file.read_text(encoding="utf-8")
                ))
            except Exception:
                pass

        # ドキュメントメタデータ付きリスト
        metadata = self._load_doc_metadata()
        document_list: list[dict] = []
        for name in sorted(self._docs.keys()):
            meta = metadata.get(name, {})
            document_list.append({
                "name": name,
                "uploaded_at": meta.get("uploaded_at", ""),
                "updated_at": meta.get("updated_at", ""),
                "size_bytes": meta.get("size_bytes", 0),
                "ext": meta.get("ext", Path(name).suffix.lower()),
            })

        return {
            "total_documents": len(self._docs),
            "documents": sorted(self._docs.keys()),
            "document_list": document_list,
            "mode": "Corpus2Skill",
            "llm_model": self._config.get("llm_model", ""),
            "skill_count": skill_count,
            "chunk_count": chunk_count,
            "embed_model": self._config.get("embed_model", ""),
            "max_top_skills": self._config.get("max_top_skills", 6),
            "branching_factor": self._config.get("branching_factor", 4),
            "chunk_max_chars": self._config.get("chunk_max_chars", 800),
            "overlap_chars": self._config.get("overlap_chars", 0),
        }

    async def set_llm_model(self, model: str) -> None:
        self._config["llm_model"] = model
        label = model if model else "（自動検出）"
        print(f"[Corpus2Skill] LLM モデルを変更しました: {label}")
