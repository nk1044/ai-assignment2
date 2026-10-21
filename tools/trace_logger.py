"""Record your own bandwidth trace (README §12 U1).

Continuously downloads from a speed-test server and logs the achieved downlink throughput once per
``--interval`` seconds, in the same ``<time_s> <bandwidth_Mbps>`` format as the FCC/HSDPA traces.

Use it on a laptop tethered to the phone's hotspot (or on Android via Termux: ``pkg install python``),
while travelling — metro, bus, auto, train, campus Wi-Fi:

    python tools/trace_logger.py --name delhi_metro_yellow_line --duration 600
    -> data/traces/india/delhi_metro_yellow_line   (+ a .meta.json with notes / network info)

WARNING: this consumes mobile data. The download is capped at ``--cap-mbps`` (default 8 Mbps ≈ 60 MB
per minute at most) and ``--max-mb`` stops the recording once that much data has been used.
"""

import argparse
import json
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "traces" / "india"
DEFAULT_URL = "https://speed.cloudflare.com/__down?bytes={bytes}"
BLOCK = 64 * 1024


class Downloader(threading.Thread):
    """Keeps a download running back-to-back and counts received bytes."""

    def __init__(self, url_template, request_bytes, cap_mbps=None, timeout=10.0):
        super().__init__(daemon=True)
        self.cap_bytes_per_s = cap_mbps * 1e6 / 8 if cap_mbps else None
        self.url = url_template.format(bytes=int(request_bytes))
        self.timeout = timeout
        self.bytes = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": "abr-trace-logger/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout) as stream:
                    while not self.stop.is_set():
                        data = stream.read(BLOCK)
                        if not data:
                            break                  # request finished: start the next one
                        self.bytes += len(data)
                        self._throttle()
            except OSError:                        # no signal / dropped connection: retry
                self.stop.wait(0.5)

    def _throttle(self):
        """Never read faster than the cap (saves mobile data; the ABR ladder tops out at 4.3 Mbps)."""
        if self.cap_bytes_per_s is None:
            return
        if not hasattr(self, "_t0"):
            self._t0, self._b0 = time.monotonic(), self.bytes
        ahead = (self.bytes - self._b0) / self.cap_bytes_per_s - (time.monotonic() - self._t0)
        if ahead > 0:
            time.sleep(ahead)
        elif ahead < -1.0:                         # link was slower than the cap: don't bank credit
            self._t0, self._b0 = time.monotonic(), self.bytes


def record(args):
    dl = Downloader(args.url, args.request_mb * 1e6, args.cap_mbps or None)
    dl.start()
    samples, start = [], time.monotonic()
    last_t, last_bytes = start, 0
    try:
        while last_t - start < args.duration and dl.bytes < args.max_mb * 1e6:
            next_t = last_t + args.interval
            time.sleep(max(0.0, next_t - time.monotonic()))
            now, now_bytes = time.monotonic(), dl.bytes
            mbps = (now_bytes - last_bytes) * 8 / 1e6 / (now - last_t)
            samples.append((round(last_t - start, 3), round(mbps, 4)))
            print(f"\r{samples[-1][0]:7.1f}s  {mbps:6.2f} Mbps   used {now_bytes / 1e6:7.1f} MB   ",
                  end="", flush=True)
            last_t, last_bytes = now, now_bytes
    finally:
        dl.stop.set()
        print()
    return samples, dl.bytes


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="trace name, e.g. bangalore_bus_500d")
    p.add_argument("--duration", type=float, default=300, help="seconds to record")
    p.add_argument("--interval", type=float, default=1.0, help="seconds per sample")
    p.add_argument("--max-mb", type=float, default=400, help="stop after using this much data")
    p.add_argument("--cap-mbps", type=float, default=8.0,
                   help="max measured rate (saves data; 0 = uncapped). Anything above ~8 Mbps does not change "
                        "ABR decisions for our 300–4300 kbps ladder")
    p.add_argument("--request-mb", type=float, default=50, help="size of each download request")
    p.add_argument("--url", default=DEFAULT_URL, help="download URL template with {bytes}")
    p.add_argument("--network", default="", help="free-text, e.g. 'Jio 4G', 'Airtel 5G', 'campus Wi-Fi'")
    p.add_argument("--notes", default="", help="free-text, e.g. 'Rajiv Chowk -> Huda City Centre'")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"recording '{args.name}' for {args.duration:.0f}s (max {args.max_mb:.0f} MB) — Ctrl+C to stop early")
    started = datetime.now().isoformat(timespec="seconds")
    try:
        samples, used = record(args)
    except KeyboardInterrupt:
        print("stopped — nothing saved")
        return
    path = out / args.name
    path.write_text("".join(f"{t}\t{bw}\n" for t, bw in samples))
    mean = sum(bw for _, bw in samples) / max(len(samples), 1)
    (out / f"{args.name}.meta.json").write_text(json.dumps({
        "name": args.name, "started": started, "samples": len(samples), "mean_mbps": round(mean, 3),
        "data_used_mb": round(used / 1e6, 1), "network": args.network, "notes": args.notes,
        "interval_s": args.interval, "cap_mbps": args.cap_mbps}, indent=2))
    print(f"wrote {path} ({len(samples)} samples, mean {mean:.2f} Mbps, used {used / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
