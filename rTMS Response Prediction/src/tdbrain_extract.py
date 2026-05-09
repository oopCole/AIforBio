"""
extract password-protected TDBRAIN zip archives (e.g. derivatives split).

the Brainclinics / Synapse distribution uses an encrypted zip; the password
is provided with your data use agreement or download instructions — not in
this repo. set environment variable TDBRAIN_ZIP_PASSWORD or pass --password.
"""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path


def _password_bytes(password: str | None) -> bytes:
    if not password:
        raise SystemExit(
            "missing zip password: set env TDBRAIN_ZIP_PASSWORD or use --password "
            "(see README; password comes from your DUA / download page)"
        )
    return password.encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="extract encrypted TDBRAIN zip")
    ap.add_argument("--zip", type=Path, required=True, help="path to .zip file")
    ap.add_argument("--out", type=Path, required=True, help="output directory (created)")
    ap.add_argument(
        "--password",
        default=None,
        help="zip password (prefer env TDBRAIN_ZIP_PASSWORD to avoid shell history)",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="only test password on dataset_description.json then exit",
    )
    args = ap.parse_args()

    pwd = args.password or os.environ.get("TDBRAIN_ZIP_PASSWORD")
    pwd_b = _password_bytes(pwd)

    if not args.zip.is_file():
        raise SystemExit(f"zip not found: {args.zip}")

    args.out.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip, "r") as zf:
        zf.setpassword(pwd_b)
        try:
            zf.read("dataset_description.json")
        except KeyError:
            # some archives use a prefix folder
            names = zf.namelist()
            jsons = [n for n in names if n.endswith("dataset_description.json")]
            if not jsons:
                raise SystemExit("no dataset_description.json in archive (unexpected layout)")
            zf.read(jsons[0])
        print("password ok: read dataset_description.json")

        if args.verify_only:
            return

        print(f"extracting to {args.out} (this can take a long time for ~40GB)...")
        zf.extractall(args.out)
        print("done.")


if __name__ == "__main__":
    main()

