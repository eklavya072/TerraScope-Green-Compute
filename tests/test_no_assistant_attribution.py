"""Nothing published may attribute this work to an AI assistant.

The repository owner's instruction is that no commit, file or metadata record
credits an assistant. It has been breached twice: ten commits carried a
Co-Authored-By trailer and needed a history rewrite to remove, and one more
appeared a single message after the rule was restated.

This matters more now than it did. A Zenodo DOI archives a snapshot
permanently, and the creators list in that record cannot be quietly corrected
the way a commit message can -- a wrong one is citable forever.

Scope: tracked file CONTENT, the two metadata files Zenodo reads, and the commit
messages reachable from HEAD. Nothing here can police a future commit message
before it is written, so it is a backstop rather than a lock.
"""

import json
import os
import re
import subprocess

import pytest
import yaml

# Attribution phrasings, not bare vendor names: ".claude/" appears in
# .gitignore as a tool directory to exclude, which is the opposite of a credit.
PATTERNS = [
    r"Co-Authored-By:\s*Claude",
    r"Claude\s+(Opus|Sonnet|Haiku)",
    r"noreply@anthropic\.com",
    r"Generated with \[?Claude",
    r"🤖",
]


def _tracked_files():
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout
    return [f for f in out.split("\n") if f and os.path.isfile(f)]


def test_no_tracked_file_credits_an_assistant():
    offenders = []
    for path in _tracked_files():
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except (UnicodeDecodeError, OSError):
            continue          # binary: models, images, the video
        for pat in PATTERNS:
            if re.search(pat, text):
                offenders.append(f"{path}: {pat}")
    assert not offenders, "assistant attribution found in:\n  " + "\n  ".join(offenders)


def test_no_commit_message_reachable_from_head_credits_an_assistant():
    log = subprocess.run(["git", "log", "--format=%B", "HEAD"],
                         capture_output=True, text=True).stdout
    if not log.strip():
        pytest.skip("no commit history available")
    found = [p for p in PATTERNS if re.search(p, log)]
    assert not found, (
        "assistant attribution in commit messages reachable from HEAD: "
        + ", ".join(found))


def test_zenodo_and_citation_creators_are_the_owner_only():
    """The field that becomes the DOI record's author list."""
    with open(".zenodo.json") as fh:
        zenodo = json.load(fh)
    with open("CITATION.cff") as fh:
        cff = yaml.safe_load(fh)

    names = [c["name"] for c in zenodo["creators"]]
    assert names == ["Singh, Eklavya"], f"unexpected Zenodo creators: {names}"

    authors = [f"{a['family-names']}, {a['given-names']}" for a in cff["authors"]]
    assert authors == ["Singh, Eklavya"], f"unexpected CITATION authors: {authors}"
