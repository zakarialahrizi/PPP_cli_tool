import argparse, os, sys
from pathlib import Path
from .converter import file_to_image
from .inference import predict

BANNER = r"""
 ███╗   ███╗ █████╗ ██╗    ██╗   ██╗██╗███████╗
 ████╗ ████║██╔══██╗██║    ██║   ██║██║██╔════╝
 ██╔████╔██║███████║██║    ██║   ██║██║███████╗
 ██║╚██╔╝██║██╔══██║██║    ╚██╗ ██╔╝██║╚════██║
 ██║ ╚═╝ ██║██║  ██║███████╗╚████╔╝ ██║███████║
 ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝ ╚═══╝  ╚═╝╚══════╝
   CNN-based Malware Scanner  |  v0.1.0
"""

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(
        prog="malvis",
        description="CNN-based malware scanner — converts binaries to images and classifies them",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
examples:
  malvis --path suspicious.exe
  malvis --scan-dir
  malvis --path sample.bin --verbose
        """
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--path",     type=Path, help="Scan a single file")
    group.add_argument("--scan-dir", action="store_true", help="Scan all files in current directory")
    parser.add_argument("--verbose", action="store_true", help="Show extra details per file")
    args = parser.parse_args()

    targets = list(Path(".").iterdir()) if args.scan_dir else [args.path]
    targets = [f for f in targets if f.is_file()]

    print(f"  Scanning {len(targets)} file(s)...\n")

    clean, malware, errors = 0, 0, 0

    for f in targets:
        try:
            img = file_to_image(f)
            label, conf = predict(img)
            if label != "Benign":
                status = "⚠  MALWARE"
                malware += 1
            else:
                status = "✓  Clean  "
                clean += 1
            print(f"  {status}  [{conf*100:.1f}%]  {label:<20}  {f}")
            if args.verbose:
                print(f"           size: {f.stat().st_size} bytes")
        except Exception as e:
            print(f"  [ERROR]  {f}: {e}", file=sys.stderr)
            errors += 1

    print(f"""
  ─────────────────────────────
  ✓  Clean   : {clean}
  ⚠  Malware : {malware}
  ✗  Errors  : {errors}
  ─────────────────────────────
    """)

if __name__ == "__main__":
    main()
