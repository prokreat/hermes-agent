"""Tests for telemetry opt-out enforcement across anon_auth, consent, and models_dev."""

from unittest.mock import patch

from agent import models_dev
from hermes_cli import anon_auth
from hermes_cli.observability import shared_metrics_consent


def test_guest_allowed_defaults_to_false_without_explicit_opt_in():
    with patch("hermes_cli.config.load_config_readonly", return_value={}):
        assert anon_auth.guest_allowed() is False


def test_guest_allowed_blocked_when_telemetry_disabled():
    cfg = {
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        "nous": {"guest": True},
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert anon_auth.guest_allowed() is False


def test_guest_allowed_unconditionally_disabled():
    cfg = {
        "telemetry": {"shared_metrics": {"enabled": True, "send": True}},
        "nous": {"guest": True},
    }
    with patch("hermes_cli.config.load_config_readonly", return_value=cfg):
        assert anon_auth.guest_allowed() is False


def test_set_answer_opt_out_disables_guest_and_models_dev():
    target = {}
    shared_metrics_consent.set_answer(target, False, False)
    assert target["telemetry"]["shared_metrics"]["enabled"] is False
    assert target["telemetry"]["shared_metrics"]["send"] is False
    assert target["nous"]["guest"] is False
    assert target["models_dev"]["enabled"] is False


def test_models_dev_background_refresh_is_noop():
    assert models_dev._start_background_refresh_models_dev() is None
