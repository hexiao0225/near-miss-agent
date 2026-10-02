"""Near-Miss Agent: finds people-vs-vehicle near misses across every camera in a VSS archive.

Pipeline: VSS hybrid search (VAST DB + Cosmos Embed) -> captions (Cosmos Reason) and
YOLO detections -> W&B serverless LLM scores each moment -> ranked board + safety briefing.

Stdlib only so it runs on python:3.12-slim from a ConfigMap (see deploy-app-no-registry).
Routes are served at / (the Ingress strips /app).

Env:
  VSS_URL / VSS_USERNAME / VSS_PASSWORD     team VSS backend (falls back to INGRESS_URL/USERNAME/PASSWORD)
  WANDB_API_KEY, WANDB_TEAM, WANDB_PROJECT  W&B serverless inference
  WANDB_MODEL                               default meta-llama/Llama-3.1-8B-Instruct
  MOCK=1                                    serve fixtures, no network (local UI testing)
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
PORT = int(os.environ.get("PORT", "8080"))
MOCK = os.environ.get("MOCK") == "1"

VSS_URL = (os.environ.get("VSS_URL") or os.environ.get("INGRESS_URL") or "").rstrip("/")
VSS_USER = os.environ.get("VSS_USERNAME") or os.environ.get("USERNAME", "")
VSS_PASS = os.environ.get("VSS_PASSWORD") or os.environ.get("PASSWORD", "")

WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_TEAM = os.environ.get("WANDB_TEAM", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "near-miss-agent")
WANDB_MODEL = os.environ.get("WANDB_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
WANDB_BASE = "https://api.inference.wandb.ai/v1"

# Several phrasings widen recall: captions describe the same event in different words.
NEAR_MISS_QUERIES = [
    "person close to a moving vehicle",
    "pedestrian crossing in front of a moving car",
    "person stepping back to avoid a vehicle",
    "forklift approaching a person in an aisle",
    "cyclist close to a car or truck",
]

RISK_ORDER = {"high": 0, "medium": 1, "low": 2, "none": 3}


# ---------------------------------------------------------------- VSS client

_token = {"value": None}
_token_lock = threading.Lock()


def _http(method, url, body=None, headers=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read() or b"null")


def vss_token(refresh=False):
    with _token_lock:
        if refresh or not _token["value"]:
            res = _http("POST", f"{VSS_URL}/api/v1/auth/login",
                        {"username": VSS_USER, "password": VSS_PASS})
            _token["value"] = res["access_token"]
        return _token["value"]


def vss(method, path, body=None):
    for attempt in (0, 1):
        try:
            return _http(method, f"{VSS_URL}{path}", body,
                         {"Authorization": f"Bearer {vss_token(refresh=attempt == 1)}"})
        except urllib.error.HTTPError as e:
            if e.code != 401 or attempt == 1:
                raise


# ---------------------------------------------------------------- normalising hits

def _pick(d, *keys, default=None):
    """First present key, also looking one level down in metadata-ish dicts."""
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    for nest in ("metadata", "extra_metadata", "meta"):
        sub = d.get(nest)
        if isinstance(sub, dict):
            for k in keys:
                if sub.get(k) not in (None, ""):
                    return sub[k]
    return default


def _objects(hit):
    """YOLO object counts as {'person': 3, 'car': 2}, from whatever shape the row has."""
    counts = {}
    perception = hit.get("perception_json")
    if isinstance(perception, str):
        try:
            perception = json.loads(perception)
        except ValueError:
            perception = None
    if isinstance(perception, dict):
        for k, v in (perception.get("object_counts") or perception.get("counts") or {}).items():
            if isinstance(v, (int, float)):
                counts[k] = int(v)
    for tag in hit.get("tags") or []:
        m = re.match(r"^\s*([a-z ]+?)\s*(\d+)?\s*$", str(tag), re.I)
        if m:
            counts.setdefault(m.group(1).lower(), int(m.group(2) or 1))
    return counts


def normalise(search_res, query):
    """One hit per parent video (chunk_results), enriched with its best segment row."""
    rows_by_source = {r.get("source"): r for r in search_res.get("results") or []}
    hits = []
    chunks = search_res.get("chunk_results") or []
    if not chunks:  # segment rows only
        chunks = search_res.get("results") or []
    for c in chunks:
        source = c.get("preview_source") or c.get("source")
        row = rows_by_source.get(source) or c
        hits.append({
            "id": c.get("original_video") or source,
            "source": source,
            "original_video": c.get("original_video"),
            "camera_id": _pick(row, "camera_id") or _pick(c, "camera_id") or "unknown",
            "location": _pick(row, "location") or _pick(c, "location") or "",
            "start": c.get("best_match_start_sec", _pick(row, "start_sec", "segment_start_sec", default=0)),
            "end": c.get("best_match_end_sec", _pick(row, "end_sec", "segment_end_sec", default=0)),
            "score": c.get("similarity_score") or row.get("similarity_score") or 0,
            "caption": row.get("reasoning_content") or c.get("reasoning_content") or "",
            "objects": _objects(row),
            "query": query,
        })
    return hits


def search_near_misses(queries, top_k, min_sim, camera=None):
    def one(q):
        body = {"query": q, "top_k": top_k, "llm_top_n": 0, "min_similarity": min_sim,
                "include_public": True}
        if camera:
            body["metadata_filters"] = {"camera_id": camera}
        return normalise(vss("POST", "/api/v1/search", body), q)

    seen = {}
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        for hits in pool.map(one, queries):
            for h in hits:
                if h["id"] not in seen or h["score"] > seen[h["id"]]["score"]:
                    seen[h["id"]] = h
    return list(seen.values())


# ---------------------------------------------------------------- W&B scoring

SCORER_PROMPT = """You are a safety analyst reviewing CCTV, dashcam and warehouse footage.
A vision model (NVIDIA Cosmos Reason) described a 5-second clip, and YOLO counted objects.
Decide whether it shows a NEAR MISS: a person (pedestrian, cyclist, worker) and a moving
vehicle (car, truck, bus, forklift) coming close enough that one had to react or could have been hit.

Return ONLY JSON:
{"near_miss": true|false,
 "risk": "high"|"medium"|"low"|"none",
 "actors": "who and what, e.g. 'pedestrian vs turning pickup'",
 "what_happened": "one plain sentence",
 "action": "one concrete recommended action for the site/fleet owner"}

high = contact or an evasive move (stepping back, braking hard, swerving)
medium = person in the vehicle's path or within about a car length while it moves
low = person and moving vehicle in frame but separated
none = vehicles parked/stopped, or no person"""


def llm(messages, max_tokens=300, json_mode=True):
    body = {"model": WANDB_MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.1}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {WANDB_KEY}"}
    if WANDB_TEAM:
        headers["OpenAI-Project"] = f"{WANDB_TEAM}/{WANDB_PROJECT}"
    res = _http("POST", f"{WANDB_BASE}/chat/completions", body, headers, timeout=45)
    return res["choices"][0]["message"]["content"]


def _parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def heuristic(hit):
    """Keyword fallback so the board still renders if inference is unavailable."""
    text = re.sub(r"\bno (pedestrians?|people|persons?|one)\b", "", hit["caption"].lower())
    person =hit["objects"].get("person") or re.search(r"pedestrian|person|people|worker|cyclist", text)
    moving = re.search(r"moving|turning|approach|driv|pass|reversing", text)
    evasive = re.search(r"step(s|ped)? back|swerv|brak|stop(s|ped)? abruptly|jump|narrowly|close call", text)
    risk = "high" if person and evasive else "medium" if person and moving else "low" if person else "none"
    return {"near_miss": risk in ("high", "medium"), "risk": risk,
            "actors": "person vs vehicle" if person else "vehicles only",
            "what_happened": hit["caption"][:160], "action": "Review clip", "scored_by": "heuristic"}


def score(hit):
    if MOCK or not WANDB_KEY:
        return heuristic(hit)
    user = (f"Camera: {hit['camera_id']} ({hit['location']})\n"
            f"YOLO objects: {json.dumps(hit['objects'])}\n"
            f"Cosmos description: {hit['caption']}")
    try:
        out = _parse_json(llm([{"role": "system", "content": SCORER_PROMPT},
                               {"role": "user", "content": user}]))
        out["risk"] = str(out.get("risk", "none")).lower()
        if out["risk"] not in RISK_ORDER:
            out["risk"] = "none"
        out["scored_by"] = WANDB_MODEL
        return out
    except Exception as e:  # noqa: BLE001 - one bad call must not sink the scan
        fallback = heuristic(hit)
        fallback["error"] = str(e)[:200]
        return fallback


def scan(params):
    t0 = time.time()
    queries = [q for q in params.get("q", []) if q.strip()] or NEAR_MISS_QUERIES
    top_k = int(params.get("top_k", ["12"])[0])
    min_sim = float(params.get("min_sim", ["0.2"])[0])
    camera = (params.get("camera") or [None])[0] or None

    hits = mock_hits() if MOCK else search_near_misses(queries, top_k, min_sim, camera)
    t_search = time.time() - t0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for hit, verdict in zip(hits, pool.map(score, hits)):
            hit["verdict"] = verdict
    hits.sort(key=lambda h: (RISK_ORDER[h["verdict"]["risk"]], -h["score"]))

    cameras = {}
    for h in hits:
        c = cameras.setdefault(h["camera_id"], {"camera_id": h["camera_id"], "location": h["location"],
                                                "high": 0, "medium": 0, "low": 0, "none": 0})
        c[h["verdict"]["risk"]] += 1
    return {"hits": hits, "cameras": sorted(cameras.values(), key=lambda c: (-c["high"], -c["medium"])),
            "queries": queries, "model": WANDB_MODEL if WANDB_KEY and not MOCK else "heuristic",
            "timing": {"search_s": round(t_search, 2), "total_s": round(time.time() - t0, 2)}}


def briefing(scan_result):
    flagged = [h for h in scan_result.get("hits", []) if h["verdict"]["risk"] in ("high", "medium")][:15]
    lines = [f"- [{h['verdict']['risk']}] {h['camera_id']} ({h['location']}) "
             f"{h['start']}-{h['end']}s: {h['verdict'].get('what_happened', '')}" for h in flagged]
    if MOCK or not WANDB_KEY:
        return "Near-miss briefing (offline mode)\n" + "\n".join(lines)
    prompt = ("Write a short safety briefing for an operations manager from these near-miss detections "
              "across several cameras. Sections: Headline (one sentence), Hotspots (which cameras, why), "
              "Top 3 incidents, Recommended actions (3 bullets). Plain text, under 200 words.\n\n"
              + "\n".join(lines))
    return llm([{"role": "user", "content": prompt}], max_tokens=500, json_mode=False)


def mock_hits():
    return json.loads((HERE / "fixtures_hits.json").read_text())


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s %s" % (self.command, fmt % args), flush=True)

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (
            json.dumps(body).encode() if ctype == "application/json" else body.encode())
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path.removeprefix("/app") or "/"
        params = urllib.parse.parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            if path == "/health":
                return self._send(200, {"ok": True, "mock": MOCK, "model": WANDB_MODEL,
                                        "wandb": bool(WANDB_KEY), "vss": bool(VSS_URL)})
            if path == "/api/scan":
                return self._send(200, scan(params))
            if path == "/api/stream":
                return self._stream(params["source"][0])
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path.removeprefix("/app")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if path == "/api/briefing":
                return self._send(200, {"briefing": briefing(body)})
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": str(e)})

    def _stream(self, source):
        """Proxy the clip so the browser never sees the VSS token; forwards Range for seeking."""
        if MOCK:
            return self._send(404, {"error": "no video in mock mode"})
        q = urllib.parse.urlencode({"source": source, "token": vss_token()})
        req = urllib.request.Request(f"{VSS_URL}/api/v1/videos/stream?{q}")
        if self.headers.get("Range"):
            req.add_header("Range", self.headers["Range"])
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            resp = e
        self.send_response(resp.status)
        for h in ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges"):
            if resp.headers.get(h):
                self.send_header(h, resp.headers[h])
        self.end_headers()
        while chunk := resp.read(64 * 1024):
            try:
                self.wfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break


if __name__ == "__main__":
    print(f"near-miss-agent on :{PORT} mock={MOCK} vss={VSS_URL or '-'} model={WANDB_MODEL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
