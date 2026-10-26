"""Web demo: watch the trained PPO agent stream next to classic ABR algorithms.

    python demo/server.py                         # opens http://127.0.0.1:8000
    python demo/server.py --model models/ppo_asym.pt --port 8000 --no-browser

The page lets you pick a real/stress-test network trace or draw your own bandwidth curve. The server
simulates every player on that network (same deterministic simulator as evaluate.py) and the page replays
the result: picture sharpness = bitrate, spinner = stall, live buffer and QoE.
Uses only the Python standard library + this project's packages. Listens on localhost only.
"""

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from abrsim import ABREnv, Trace, load_dataset  # noqa: E402
from abrsim import config as C  # noqa: E402
from agents import PPOAgent, RLPolicy  # noqa: E402
from baselines import MPC, Bola, BufferBased, OfflineOptimal  # noqa: E402
from evaluate import run_policy  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASETS = {"Stress-test scenarios": "scenarios", "Norway HSDPA 3G (test)": "hsdpa/test", "FCC broadband (test)": "fcc/test"}
ROW_KEYS = ("chunk", "wall_time_s", "delay_s", "sleep_s", "bitrate_kbps", "buffer_s", "rebuffer_s", "reward",
            "throughput_mbps")


class Demo:
    def __init__(self, model_path):
        self.agent = PPOAgent.load(model_path)          # loaded once; only the actor is used
        self.traces = {}                       # "dataset/name" -> Trace
        self.catalog = {}
        for label, spec in DATASETS.items():
            ts = load_dataset(spec)
            self.catalog[label] = [f"{spec}::{t.name}" for t in ts]
            self.traces |= {f"{spec}::{t.name}": t for t in ts}
        self.lock = threading.Lock()

    def policies(self, oracle):
        ps = [RLPolicy(self.agent, "PPO"), MPC(robust=True), Bola(), BufferBased()]
        return ps + [OfflineOptimal()] if oracle else ps

    def simulate(self, req):
        if req.get("bandwidth"):
            bw = np.clip(np.asarray(req["bandwidth"], dtype=float), 0.0, 20.0)
            if len(bw) < 10:
                raise ValueError("draw at least 10 seconds of bandwidth")
            trace = Trace("your network", np.arange(len(bw), dtype=float), bw)
        else:
            trace = self.traces[req["trace"]]
        out = {"trace": trace.name, "chunk_len_s": C.CHUNK_LEN_S, "n_chunks": C.TOTAL_CHUNKS,
               "bitrates_kbps": C.BITRATES_KBPS, "players": []}
        end = 0.0
        with self.lock:
            for policy in self.policies(bool(req.get("oracle", True))):
                env = ABREnv([trace], noise=False, random_start=False)
                rows, per_trace = run_policy(policy, env, 1)
                end = max(end, rows[-1]["wall_time_s"])
                out["players"].append({
                    "name": "Offline optimal" if policy.name == "Oracle" else policy.name,
                    "rows": [{k: round(float(r[k]), 4) for k in ROW_KEYS} for r in rows],
                    "summary": {k: round(float(v), 3) for k, v in per_trace[0].items() if k != "trace"},
                })
        # bandwidth the players actually experienced (trace looped if shorter than the session)
        t0, times, bw = trace.times[0], [], []
        loops = 0
        while not times or times[-1] < end + 5:
            times += list(trace.times[1:] - t0 + loops * (trace.times[-1] - t0))
            bw += list(trace.bw_mbps[1:])
            loops += 1
        out["bandwidth"] = {"t": [round(x, 3) for x in times], "mbps": [round(x, 3) for x in bw]}
        return out


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            data = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/traces":
                self._send(200, json.dumps(demo.catalog), "application/json")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/api/simulate":
                return self._send(404, "not found", "text/plain")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, json.dumps(demo.simulate(req)), "application/json")
            except (KeyError, ValueError) as e:
                self._send(400, json.dumps({"error": str(e)}), "application/json")

        def log_message(self, fmt, *args):            # keep the terminal quiet
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(ROOT / "models" / "ppo_asym.pt"))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    demo = Demo(args.model)
    demo.simulate({"trace": "scenarios::metro_tunnel_0", "oracle": False})     # warm up torch
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(demo))
    url = f"http://127.0.0.1:{args.port}"
    print(f"ABR demo running at {url}  (model: {args.model})  — Ctrl+C to stop")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
