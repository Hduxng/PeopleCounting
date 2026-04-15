from types import SimpleNamespace

import main


def test_resolve_run_options_uses_config_defaults():
    cfg = {
        "video": {"source": "test2.mp4"},
        "runtime": {"display": True, "debug": True},
        "output": {"save": False, "path": None},
    }
    args = SimpleNamespace(save=None, no_save=None, display=None, debug=None)

    save_path, display, debug = main.resolve_run_options(cfg, args)

    assert save_path is None
    assert display is True
    assert debug is True


def test_resolve_run_options_uses_configured_output_path():
    cfg = {
        "video": {"source": "test2.mp4"},
        "runtime": {"display": False, "debug": False},
        "output": {"save": True, "path": "output/test2_result.mp4"},
    }
    args = SimpleNamespace(save=None, no_save=None, display=None, debug=None)

    save_path, display, debug = main.resolve_run_options(cfg, args)

    assert save_path == "output/test2_result.mp4"
    assert display is False
    assert debug is False


def test_resolve_run_options_cli_overrides_config():
    cfg = {
        "video": {"source": "test2.mp4"},
        "runtime": {"display": False, "debug": False},
        "output": {"save": True, "path": "output/from_config.mp4"},
    }
    args = SimpleNamespace(save="output/from_cli.mp4", no_save=True, display=True, debug=True)

    save_path, display, debug = main.resolve_run_options(cfg, args)

    assert save_path == "output/from_cli.mp4"
    assert display is True
    assert debug is True
