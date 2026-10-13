"""Create the Space on Hugging Face and upload a folder built by build_space.py (a Docker Space) or build_static_space.py (a static one).

    hf auth login                                   # once, in your own terminal: the token is typed there, never into a command line or a chat
    python deploy/huggingface/upload_space.py OUTDIR --space-id Hastika06/substrate-mission-control

It uses the login `hf auth login` stored (or HF_TOKEN in the environment); this file never reads, accepts or prints a token. The Space is created **private**; pass
--public to make it visible to everyone (a separate, deliberate step, and one that search engines and copies can outlive).

Before anything is sent it checks the folder: a card that says `sdk: docker` (with a Dockerfile) or `sdk: static` (with an index.html), and no file that looks like a secret.
A Docker Space needs a Hugging Face PRO subscription on the free hardware; a static Space does not.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SECRET_NAMES = re.compile(r"(^|/)(\.env[^/]*|id_rsa[^/]*|id_ed25519[^/]*|[^/]*\.pem|[^/]*\.key|[^/]*\.p12|[^/]*\.pfx|credentials[^/]*|[^/]*token[^/]*|\.netrc|\.git(/.*)?)$", re.I)
MAX_BYTES = 200 * 1024 * 1024
NEEDED = {"docker": "Dockerfile", "static": "index.html"}              # the file each kind of Space cannot do without, besides its card


class UploadError(Exception):
    """The folder or the account is not ready. The message says what to do."""


def space_sdk(folder: Path) -> str:
    """`docker` or `static`, as the folder's card declares it, or UploadError."""
    card_path = folder / "README.md"
    if not card_path.is_file():
        raise UploadError(f"{folder} has no README.md")
    card = card_path.read_text(encoding="utf-8")
    found = re.search(r"(?m)^sdk:\s*(docker|static)\s*$", card.split("---")[1] if card.startswith("---") else "")
    if not found:
        raise UploadError("the README.md card does not declare `sdk: docker` or `sdk: static`")
    return found.group(1)


def check_folder(folder: Path) -> list[str]:
    """The files that would be uploaded, or UploadError. A Space folder has its card and the file its kind needs, and nothing that looks like a secret."""
    if not folder.is_dir():
        raise UploadError(f"{folder} is not a folder (build one with build_space.py or build_static_space.py)")
    files = sorted(str(p.relative_to(folder)).replace("\\", "/") for p in folder.rglob("*") if p.is_file())
    if "README.md" not in files:
        raise UploadError(f"{folder} has no README.md")
    needed = NEEDED[space_sdk(folder)]
    if needed not in files:
        raise UploadError(f"{folder} has no {needed}")
    suspicious = [f for f in files if SECRET_NAMES.search(f)]
    if suspicious:
        raise UploadError(f"these files look like secrets and will not be uploaded: {', '.join(suspicious[:5])}")
    if sum((folder / f).stat().st_size for f in files) > MAX_BYTES:
        raise UploadError(f"the folder is larger than {MAX_BYTES // 2**20} MiB: a demo should not be")
    return files


def main(argv=None, api=None) -> int:
    parser = argparse.ArgumentParser(description="Upload a Space folder to Hugging Face (private unless --public).")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--space-id", required=True, help="owner/name, e.g. Hastika06/substrate-mission-control")
    parser.add_argument("--public", action="store_true", help="make the Space public (default: it is created private)")
    args = parser.parse_args(argv)
    try:
        files = check_folder(args.folder)
        sdk = space_sdk(args.folder)
    except UploadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    try:
        account = api.whoami()
    except Exception:                                                  # noqa: BLE001 (whatever the library raises, the advice is the same, and no token is echoed)
        print("error: not logged in to Hugging Face. Run `hf auth login` in your own terminal (or set HF_TOKEN), then run this again.", file=sys.stderr)
        return 1
    owners = {account["name"]} | {o["name"] for o in account.get("orgs", [])}
    if args.space_id.split("/")[0] not in owners:
        print(f"error: you are logged in as {account['name']}; {args.space_id.split('/')[0]} is not you or one of your organizations", file=sys.stderr)
        return 1
    commit = (args.folder / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()[:7] if (args.folder / "SOURCE_COMMIT").is_file() else "unknown"
    try:
        api.create_repo(args.space_id, repo_type="space", space_sdk=sdk, private=not args.public, exist_ok=True)
        api.upload_folder(folder_path=str(args.folder), repo_id=args.space_id, repo_type="space", commit_message=f"Mission control demo, {sdk} (source commit {commit})")
        if args.public:
            api.update_repo_settings(repo_id=args.space_id, repo_type="space", private=False)
    except Exception as error:                                         # noqa: BLE001 (a refusal from the Hub is an answer, not a crash: show what it said)
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        reason = str(error).split("\n", 1)[-1].strip() if status == 402 else type(error).__name__
        print(f"error: Hugging Face refused or failed ({status or type(error).__name__}): {reason}. Nothing was made public.", file=sys.stderr)
        return 1
    print(f"uploaded {len(files)} files to https://huggingface.co/spaces/{args.space_id} ({'public' if args.public else 'private'}); it builds there now")
    return 0


if __name__ == "__main__":
    sys.exit(main())
