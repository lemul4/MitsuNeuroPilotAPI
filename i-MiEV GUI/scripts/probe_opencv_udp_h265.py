from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hardware.opencv_udp_camera import OpenCvUdpH265CameraSpec, write_temp_sdp

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe OpenCV RTP/H265 receiver on Windows.")
    parser.add_argument("--port", type=int, default=5601)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--rate", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--ffmpeg-testsrc", action="store_true", help="Start local ffmpeg testsrc sender.")
    parser.add_argument("--input", default="", help="Optional MKV input for ffmpeg RTP sender.")
    args = parser.parse_args()

    ffmpeg = None
    spec = OpenCvUdpH265CameraSpec("wide_90", int(args.port), width=int(args.width), height=int(args.height))
    sdp_path = write_temp_sdp(spec)
    width = int(spec.width or 1280)
    height = int(spec.height or 720)
    frame_size = width * height * 3
    receiver = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-protocol_whitelist",
            "file,udp,rtp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-i",
            sdp_path,
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_size * 2,
    )
    print(f"receiver_started=True sdp={sdp_path}")

    if args.ffmpeg_testsrc:
        time.sleep(0.5)
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={int(args.width)}x{int(args.height)}:rate={int(args.rate)}",
                "-c:v",
                "libx265",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                "1",
                "-x265-params",
                "repeat-headers=1:bframes=0",
                "-f",
                "rtp",
                f"rtp://127.0.0.1:{int(args.port)}",
            ]
        )
    elif args.input:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-stream_loop",
                "-1",
                "-i",
                str(args.input),
                "-c:v",
                "copy",
                "-bsf:v",
                "hevc_mp4toannexb",
                "-f",
                "rtp",
                f"rtp://127.0.0.1:{int(args.port)}",
            ]
        )

    result = {"ok": False, "shape": None}

    def _read_one_frame():
        if receiver.stdout is None:
            return
        raw = receiver.stdout.read(frame_size)
        if len(raw) == frame_size:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            result["ok"] = True
            result["shape"] = tuple(frame.shape)

    reader = threading.Thread(target=_read_one_frame, daemon=True)
    reader.start()
    try:
        reader.join(timeout=float(args.timeout))
        print(f"frame={result['ok']} shape={result['shape']}")
        if not result["ok"]:
            try:
                receiver.terminate()
                _, err = receiver.communicate(timeout=2.0)
                if err:
                    print(err.decode(errors="ignore")[:1000])
            except Exception:
                pass
        return 0 if result["ok"] else 3
    finally:
        receiver.terminate()
        try:
            receiver.wait(timeout=2.0)
        except Exception:
            receiver.kill()
        if ffmpeg is not None:
            ffmpeg.terminate()
            try:
                ffmpeg.wait(timeout=2.0)
            except Exception:
                ffmpeg.kill()


if __name__ == "__main__":
    raise SystemExit(main())
