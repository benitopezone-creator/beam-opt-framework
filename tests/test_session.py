"""Tests for MapdlSession (launcher replaced by a FakeMapdl)."""

from __future__ import annotations

import pytest
from fakes import FakeMapdl

import beam_opt.session as session_mod
from beam_opt.session import MapdlSession


@pytest.fixture
def launches(monkeypatch):
    calls = []

    def fake_launch(**kwargs):
        calls.append(kwargs)
        return FakeMapdl()

    monkeypatch.setattr(session_mod, "launch_mapdl", fake_launch)
    return calls


def test_constructor_does_not_launch(launches, tmp_path):
    MapdlSession(run_location=str(tmp_path))
    assert launches == []


def test_context_manager_launches_once_and_exits(launches, tmp_path):
    session = MapdlSession(run_location=str(tmp_path / "run"), nproc=2, jobname="job")
    with session as mapdl:
        assert session.start() is mapdl
        assert ("filname", ("job",)) in mapdl.calls
    assert len(launches) == 1
    assert launches[0]["nproc"] == 2 and launches[0]["jobname"] == "job"
    assert ("exit", ()) in mapdl.calls
    assert session.mapdl is None
    assert (tmp_path / "run").is_dir()


def test_close_on_exception(launches, tmp_path):
    session = MapdlSession(run_location=str(tmp_path))
    with pytest.raises(KeyError), session as mapdl:
        raise KeyError("boom")
    assert ("exit", ()) in mapdl.calls
