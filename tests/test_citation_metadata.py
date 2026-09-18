"""CITATION.cff and .zenodo.json must agree with each other and with the repo.

A DOI pins a snapshot permanently. If the archived metadata names the wrong
repository, the wrong licence or the wrong version, that error is citable
forever and cannot be corrected in place -- only superseded by another release.

This is not hypothetical. The repository was renamed once already, and both
files carried the old URL until they were edited by hand. Nothing checked them,
because nothing read them: they exist for Zenodo and GitHub, neither of which
runs in CI.
"""

import json
import os
import re
import subprocess

import pytest
import yaml


@pytest.fixture(scope="module")
def cff():
    with open("CITATION.cff") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def zenodo():
    with open(".zenodo.json") as fh:
        return json.load(fh)


def test_both_files_parse_and_carry_the_required_fields(cff, zenodo):
    for key in ("cff-version", "message", "title", "authors", "version",
                "date-released", "license", "repository-code"):
        assert key in cff, f"CITATION.cff is missing {key}"
    for key in ("title", "description", "upload_type", "license", "creators",
                "version"):
        assert key in zenodo, f".zenodo.json is missing {key}"


def test_the_two_files_do_not_contradict_each_other(cff, zenodo):
    assert cff["title"] == zenodo["title"], "title differs between the two files"
    assert cff["license"] == zenodo["license"], "licence differs"
    assert str(cff["version"]) == str(zenodo["version"]), "version differs"


def test_the_repository_url_is_this_repository(cff):
    """The failure a rename causes, caught before it is archived."""
    remote = subprocess.run(["git", "remote", "get-url", "origin"],
                            capture_output=True, text=True).stdout.strip()
    if not remote:
        pytest.skip("no git remote configured")
    slug = re.sub(r"\.git$", "", remote).split("github.com/")[-1].strip("/")
    assert slug, f"could not read a repository slug from {remote!r}"
    assert slug in cff["repository-code"], (
        f"CITATION.cff points at {cff['repository-code']}, but origin is {slug}")

    readme = open("README.md").read()
    assert slug in readme, "the README does not reference this repository either"


def test_the_eurosat_reference_is_followable(cff):
    """A reference with no year and no venue is not a citation."""
    ref = cff["references"][0]
    assert "EuroSAT" in ref["title"]
    for key in ("year", "journal", "doi"):
        assert ref.get(key), f"the EuroSAT reference has no {key}"


def test_date_released_is_a_real_iso_date(cff):
    value = str(cff["date-released"])
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), (
        f"date-released must be ISO yyyy-mm-dd, got {value!r}")
