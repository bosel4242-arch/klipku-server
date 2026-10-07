"""KlipKu backend: ambil video, potong, ambil foto, dan fitur AI (OpenAI)."""
import asyncio
import base64
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
import yt_dlp
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---------- Konfigurasi (bisa diubah lewat environment variable) ----------
DATA = Path(os.getenv("KLIPKU_DATA", "/tmp/klipku"))
DATA.mkdir(parents=True, exist_ok=True)
TTL = int(os.getenv("KLIPKU_TTL_HOURS", "24")) * 3600
MAX_MINUTES = int(os.getenv("KLIPKU_MAX_MINUTES", "15"))
MAX_MB = int(os.getenv("KLIPKU_MAX_MB", "100"))
MAX_IMG_MB = 15
MAX_PER_HOUR = int(os.getenv("KLIPKU_MAX_PER_HOUR", "40"))  # batas permintaan berat per IP
COOKIES = os.getenv("KLIPKU_COOKIES", "")  # opsional: path cookies.txt untuk yt-dlp

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
STT_MODEL = os.getenv("OPENAI_STT_MODEL", "whisper-1")
TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
IMG_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
IMG_QUALITY = os.getenv("OPENAI_IMAGE_QUALITY", "medium")

ALLOWED_HOSTS = ("youtube.com", "youtu.be", "tiktok.com", "facebook.com", "fb.watch", "fb.com")
ID_RE = re.compile(r"^[a-f0-9]{32}$")
NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

app = FastAPI(title="KlipKu")

ORIGINS = [o.strip() for o in os.getenv("KLIPKU_ORIGINS", "https://youdown.my.id,http://localhost:8000").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["*"], expose_headers=["Content-Disposition"], max_age=600)


@app.middleware("http")
async def private_network(request: Request, call_next):
    resp = await call_next(request)
    if request.method == "OPTIONS" and request.headers.get("access-control-request-private-network"):
        resp.headers["Access-Control-Allow-Private-Network"] = "true"
    return resp

heavy = asyncio.Semaphore(int(os.getenv("KLIPKU_CONCURRENCY", "2")))
hits: dict[str, list[float]] = {}


# ---------- Util ----------
def rate_limit(request: Request):
    ip = request.client.host if request.client else "x"
    now = time.time()
    h = [t for t in hits.get(ip, []) if now - t < 3600]
    if len(h) >= MAX_PER_HOUR:
        raise HTTPException(429, "Terlalu banyak permintaan. Coba lagi sebentar lagi.")
    h.append(now)
    hits[ip] = h


def new_dir() -> tuple[str, Path]:
    i = uuid.uuid4().hex
    d = DATA / i
    d.mkdir(parents=True)
    return i, d


def get_dir(i: str) -> Path:
    if not ID_RE.match(i):
        raise HTTPException(400, "ID tidak valid.")
    d = DATA / i
    if not d.is_dir():
        raise HTTPException(404, "File tidak ditemukan atau sudah kedaluwarsa (24 jam).")
    return d


def meta_write(d: Path, **kw):
    (d / "meta.json").write_text(json.dumps(kw))


def meta_read(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text())
    except Exception:
        return {}


def run(cmd: list[str], timeout=600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-400:])
    return p.stdout


def probe(path: Path) -> dict:
    out = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    return json.loads(out)


def ensure_mp4(src: Path) -> Path:
    """Pastikan H.264 + AAC dalam MP4 supaya bisa diputar di semua browser."""
    info = probe(src)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if not v:
        raise RuntimeError("Tidak ada trek video.")
    ok = src.suffix == ".mp4" and v.get("codec_name") == "h264" and (not a or a.get("codec_name") == "aac")
    out = src.with_name("src.mp4")
    if ok:
        if src.name != "src.mp4":
            src.rename(out)
        return out
    tmp = src.with_name("conv.mp4")
    run(["ffmpeg", "-y", "-i", str(src), "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "24", "-c:a", "aac", "-movflags", "+faststart", str(tmp)], timeout=900)
    src.unlink(missing_ok=True)
    tmp.rename(out)
    return out


def host_ok(url: str) -> bool:
    try:
        u = urlparse(url)
        h = (u.hostname or "").lower()
        return u.scheme in ("http", "https") and any(h == d or h.endswith("." + d) for d in ALLOWED_HOSTS)
    except Exception:
        return False


def safe_public_url(url: str):
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise HTTPException(400, "Tautan tidak valid.")
    try:
        for fam, _, _, _, addr in socket.getaddrinfo(u.hostname, None):
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise HTTPException(400, "Tautan tidak diizinkan.")
    except socket.gaierror:
        raise HTTPException(400, "Alamat tautan tidak ditemukan.")


def clean_name(s: str, default="klip") -> str:
    s = re.sub(r"[^\w\- ]+", "", s or "", flags=re.UNICODE).strip().replace(" ", "-")
    return (s[:60] or default)


# ---------- Pembersihan otomatis ----------
async def janitor():
    while True:
        now = time.time()
        for d in DATA.iterdir():
            try:
                if d.is_dir() and now - d.stat().st_mtime > TTL:
                    shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass
        await asyncio.sleep(600)


@app.on_event("startup")
async def _start():
    asyncio.create_task(janitor())


# ---------- Model request ----------
class UrlIn(BaseModel):
    url: str


class CutIn(BaseModel):
    id: str
    start: float = 0
    end: float = 0
    full: bool = False


class IdIn(BaseModel):
    id: str


class CaptionIn(BaseModel):
    idea: str
    tone: str = "Santai"


class TextIn(BaseModel):
    text: str


class TtsIn(BaseModel):
    text: str
    voice: str = "alloy"
    style: str = ""


class ImgIn(BaseModel):
    prompt: str


# ---------- Kesehatan ----------
@app.get("/api/health")
def health():
    return {"ok": True, "ai": bool(OPENAI_KEY), "ffmpeg": bool(shutil.which("ffmpeg")),
            "maxMinutes": MAX_MINUTES, "maxMB": MAX_MB}


# ---------- Video ----------
def _fetch_video(url: str) -> dict:
    i, d = new_dir()
    base = {"quiet": True, "no_warnings": True, "noplaylist": True, "socket_timeout": 20, "retries": 2}
    if COOKIES and Path(COOKIES).is_file():
        base["cookiefile"] = COOKIES
    try:
        with yt_dlp.YoutubeDL(base) as y:
            info = y.extract_info(url, download=False)
        dur = info.get("duration") or 0
        if dur > MAX_MINUTES * 60:
            raise ValueError(f"Video terlalu panjang (maks. {MAX_MINUTES} menit).")
        opts = dict(base)
        opts.update({
            "format": "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][vcodec^=avc1]/bv*[height<=720]+ba/b[height<=720]/b",
            "merge_output_format": "mp4",
            "outtmpl": str(d / "dl.%(ext)s"),
            "max_filesize": MAX_MB * 1024 * 1024,
        })
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([url])
        files = [f for f in d.glob("dl.*") if f.suffix not in (".part", ".ytdl")]
        if not files:
            raise ValueError("Video gagal diunduh atau melebihi batas ukuran.")
        src = files[0]
        if src.stat().st_size > MAX_MB * 1024 * 1024 * 1.2:
            raise ValueError(f"Ukuran video melebihi {MAX_MB} MB.")
        out = ensure_mp4(src)
        real_dur = float(probe(out)["format"].get("duration", dur))
        meta_write(d, title=info.get("title") or "video", duration=real_dur, kind="video",
                   thumb=info.get("thumbnail"), source=url)
        return {"id": i, "title": info.get("title") or "video", "duration": real_dur,
                "thumbnail": info.get("thumbnail"), "media": f"/api/media/{i}/src.mp4"}
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        msg = str(e)
        if "Sign in to confirm" in msg or "not a bot" in msg:
            msg = "YouTube meminta verifikasi login dari server ini. Lihat README bagian cookies."
        elif isinstance(e, ValueError):
            pass
        else:
            msg = "Gagal mengambil video. Tautan mungkin privat, dibatasi, atau tidak didukung."
        raise HTTPException(400, msg)


@app.post("/api/video/fetch")
async def video_fetch(body: UrlIn, request: Request):
    rate_limit(request)
    url = body.url.strip()
    if not host_ok(url):
        raise HTTPException(400, "Hanya tautan YouTube, TikTok, atau Facebook yang didukung.")
    async with heavy:
        return await asyncio.to_thread(_fetch_video, url)


@app.post("/api/video/upload")
async def video_upload(request: Request, file: UploadFile = File(...)):
    rate_limit(request)
    i, d = new_dir()
    ext = Path(file.filename or "v.mp4").suffix.lower()
    if ext not in (".mp4", ".mov", ".webm", ".mkv", ".m4v", ".3gp", ".mp3", ".m4a", ".wav"):
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Format file tidak didukung.")
    dst = d / f"up{ext}"
    size = 0
    with dst.open("wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(400, f"Ukuran file melebihi {MAX_MB} MB.")
            f.write(chunk)
    try:
        async with heavy:
            info = await asyncio.to_thread(probe, dst)
            has_v = any(s["codec_type"] == "video" for s in info["streams"])
            if has_v:
                out = await asyncio.to_thread(ensure_mp4, dst)
            else:
                out = dst
            dur = float(info["format"].get("duration", 0))
        if dur > MAX_MINUTES * 60:
            raise ValueError(f"Durasi melebihi {MAX_MINUTES} menit.")
        title = Path(file.filename or "video").stem
        meta_write(d, title=title, duration=dur, kind="video" if has_v else "audio")
        return {"id": i, "title": title, "duration": dur, "media": f"/api/media/{i}/{out.name}", "hasVideo": has_v}
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, str(e) if isinstance(e, ValueError) else "File tidak bisa dibaca.")


def _cut(i: str, start: float, end: float, full: bool) -> dict:
    d = get_dir(i)
    src = d / "src.mp4"
    if not src.exists():
        raise HTTPException(400, "Video sumber tidak ditemukan.")
    m = meta_read(d)
    dur = float(m.get("duration", 0))
    name = f"clip-{int(time.time())}.mp4"
    out = d / name
    if full:
        shutil.copy(src, out)
    else:
        start = max(0.0, start)
        end = min(end, dur) if dur else end
        if end - start < 0.3:
            raise HTTPException(400, "Rentang potongan terlalu pendek.")
        try:
            run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{end - start:.2f}",
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac",
                 "-movflags", "+faststart", str(out)], timeout=600)
        except Exception:
            raise HTTPException(500, "Gagal memotong video.")
    return {"file": name, "size": out.stat().st_size, "name": clean_name(m.get("title", "video")) + ("" if full else "-klip") + ".mp4",
            "url": f"/api/file/{i}/{name}"}


@app.post("/api/video/cut")
async def video_cut(body: CutIn, request: Request):
    rate_limit(request)
    async with heavy:
        return await asyncio.to_thread(_cut, body.id, body.start, body.end, body.full)


# ---------- Foto ----------
async def _download_image(url: str) -> dict:
    cur = url
    async with httpx.AsyncClient(timeout=20, follow_redirects=False,
                                 headers={"User-Agent": "Mozilla/5.0 KlipKu"}) as c:
        for _ in range(4):
            safe_public_url(cur)
            async with c.stream("GET", cur) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    cur = str(httpx.URL(cur).join(r.headers["location"]))
                    continue
                if r.status_code != 200:
                    raise HTTPException(400, f"Gambar tidak bisa diambil (kode {r.status_code}).")
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}.get(ctype)
                if not ext:
                    raise HTTPException(400, "Hanya JPG, PNG, dan WEBP yang didukung.")
                i, d = new_dir()
                size = 0
                f = (d / f"image{ext}").open("wb")
                try:
                    async for chunk in r.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_IMG_MB * 1024 * 1024:
                            raise HTTPException(400, f"Ukuran gambar melebihi {MAX_IMG_MB} MB.")
                        f.write(chunk)
                except Exception:
                    f.close()
                    shutil.rmtree(d, ignore_errors=True)
                    raise
                f.close()
                meta_write(d, title="gambar", kind="image")
                return {"id": i, "file": f"image{ext}", "type": ctype, "size": size,
                        "url": f"/api/file/{i}/image{ext}", "media": f"/api/media/{i}/image{ext}"}
    raise HTTPException(400, "Terlalu banyak pengalihan tautan.")


@app.post("/api/image/fetch")
async def image_fetch(body: UrlIn, request: Request):
    rate_limit(request)
    return await _download_image(body.url.strip())


@app.post("/api/image/thumbnail")
async def image_thumb(body: UrlIn, request: Request):
    rate_limit(request)
    url = body.url.strip()
    if not host_ok(url):
        raise HTTPException(400, "Hanya tautan YouTube, TikTok, atau Facebook yang didukung.")

    def get():
        opts = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True}
        if COOKIES and Path(COOKIES).is_file():
            opts["cookiefile"] = COOKIES
        with yt_dlp.YoutubeDL(opts) as y:
            return y.extract_info(url, download=False)

    try:
        info = await asyncio.to_thread(get)
    except Exception:
        raise HTTPException(400, "Gagal membaca video dari tautan itu.")
    th = info.get("thumbnail")
    if not th:
        raise HTTPException(400, "Video ini tidak punya sampul.")
    return await _download_image(th)


# ---------- Berkas ----------
def _path(i: str, name: str) -> Path:
    d = get_dir(i)
    if not NAME_RE.match(name):
        raise HTTPException(400, "Nama file tidak valid.")
    p = d / name
    if not p.is_file():
        raise HTTPException(404, "File tidak ditemukan.")
    return p


@app.get("/api/media/{i}/{name}")
def media(i: str, name: str):
    return FileResponse(_path(i, name))  # mendukung Range untuk pemutar video


@app.get("/api/file/{i}/{name}")
def file_dl(i: str, name: str, dl: str = ""):
    p = _path(i, name)
    fname = clean_name(dl.rsplit(".", 1)[0], p.stem) + p.suffix if dl else p.name
    return FileResponse(p, filename=fname)


# ---------- AI (OpenAI) ----------
def need_ai():
    if not OPENAI_KEY:
        raise HTTPException(503, "Fitur AI belum aktif: isi OPENAI_API_KEY di server.")


async def openai_json(path: str, payload: dict, timeout=120):
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post("https://api.openai.com/v1" + path, json=payload,
                         headers={"Authorization": f"Bearer {OPENAI_KEY}"})
    if r.status_code != 200:
        try:
            msg = r.json()["error"]["message"]
        except Exception:
            msg = r.text[:200]
        raise HTTPException(502, f"OpenAI: {msg}")
    return r


async def chat(system: str, user: str) -> str:
    r = await openai_json("/chat/completions", {
        "model": CHAT_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0.8,
    })
    return r.json()["choices"][0]["message"]["content"].strip()


@app.post("/api/ai/caption")
async def ai_caption(body: CaptionIn, request: Request):
    need_ai()
    rate_limit(request)
    if not body.idea.strip():
        raise HTTPException(400, "Ceritakan videomu dulu.")
    text = await chat(
        "Kamu penulis caption media sosial berbahasa Indonesia. Balas hanya daftar caption.",
        f"Buat 3 pilihan caption bergaya {body.tone[:30]} untuk video ini: {body.idea[:1000]}\n"
        "Tiap caption maksimal 150 karakter dan diakhiri 3-5 hashtag. Format: tiga baris bernomor 1-3.")
    return {"text": text}


@app.post("/api/ai/tidy")
async def ai_tidy(body: TextIn, request: Request):
    need_ai()
    rate_limit(request)
    text = await chat(
        "Kamu editor transkrip. Pertahankan bahasa asli dan jangan menambah isi baru.",
        "Rapikan teks berikut (tanda baca, huruf kapital, paragraf). Lalu tulis 'Ringkasan:' dengan 3 poin singkat.\n\n"
        + body.text[:12000])
    return {"text": text}


def _audio_for(i: str) -> Path:
    d = get_dir(i)
    out = d / "audio.mp3"
    if not out.exists():
        src = next((p for p in d.iterdir() if p.name.startswith(("src.", "up."))), None)
        if not src:
            raise HTTPException(400, "Sumber audio tidak ditemukan.")
        run(["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(out)])
    return out


@app.post("/api/ai/transcribe")
async def ai_transcribe(body: IdIn, request: Request):
    need_ai()
    rate_limit(request)
    async with heavy:
        audio = await asyncio.to_thread(_audio_for, body.id)
    if audio.stat().st_size > 24 * 1024 * 1024:
        raise HTTPException(400, "Audio terlalu besar untuk ditranskripsi.")
    async with httpx.AsyncClient(timeout=300) as c:
        r = await c.post("https://api.openai.com/v1/audio/transcriptions",
                         headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                         data={"model": STT_MODEL},
                         files={"file": ("audio.mp3", audio.read_bytes(), "audio/mpeg")})
    if r.status_code != 200:
        raise HTTPException(502, "Transkripsi gagal: " + r.text[:200])
    return {"text": r.json().get("text", "")}


@app.post("/api/ai/tts")
async def ai_tts(body: TtsIn, request: Request):
    need_ai()
    rate_limit(request)
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Tulis teksnya dulu.")
    voice = body.voice if body.voice in {"alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"} else "alloy"
    payload = {"model": TTS_MODEL, "input": text[:4000], "voice": voice, "response_format": "mp3"}
    if body.style.strip() and "tts-1" not in TTS_MODEL:
        payload["instructions"] = body.style.strip()[:200]
    r = await openai_json("/audio/speech", payload, timeout=180)
    i, d = new_dir()
    (d / "speech.mp3").write_bytes(r.content)
    meta_write(d, title="sulih-suara", kind="audio")
    return {"id": i, "media": f"/api/media/{i}/speech.mp3", "url": f"/api/file/{i}/speech.mp3"}


@app.post("/api/ai/image")
async def ai_image(body: ImgIn, request: Request):
    need_ai()
    rate_limit(request)
    if not body.prompt.strip():
        raise HTTPException(400, "Tulis deskripsi gambarnya dulu.")
    payload = {"model": IMG_MODEL, "prompt": body.prompt[:3000], "size": "1024x1024", "n": 1}
    if IMG_MODEL.startswith("gpt-image"):
        payload["quality"] = IMG_QUALITY
    r = await openai_json("/images/generations", payload, timeout=240)
    item = r.json()["data"][0]
    i, d = new_dir()
    if item.get("b64_json"):
        (d / "ai.png").write_bytes(base64.b64decode(item["b64_json"]))
    elif item.get("url"):
        async with httpx.AsyncClient(timeout=60) as c:
            (d / "ai.png").write_bytes((await c.get(item["url"])).content)
    else:
        raise HTTPException(502, "OpenAI tidak mengembalikan gambar.")
    meta_write(d, title="gambar-ai", kind="image")
    return {"id": i, "media": f"/api/media/{i}/ai.png", "url": f"/api/file/{i}/ai.png"}


# ---------- Frontend ----------
STATIC = Path(__file__).parent / "static"
if STATIC.is_dir():
    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
