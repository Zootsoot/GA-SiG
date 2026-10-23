import json

import pytest

from gasig.config import ConfigError, load_config, schema_rows, template_config, validate_config


def minimal(**extra):
    cfg = {"mechanism": "mech.yaml", "idt": {"enabled": True, "source": "precomputed",
                                             "precomputed": {"time": "t.npy", "T": "T.npy", "P": "P.npy",
                                                             "Y": "Y.npy", "index": "i.npy"}}}
    cfg.update(extra)
    return cfg


def test_template_validates(tmp_path):
    cfg = validate_config(template_config(), tmp_path)
    assert cfg["idt"]["enabled"] is True
    assert cfg["sensitivity"]["rate_law"] == "original"
    assert cfg["sensitivity"]["substep_tol"] is None
    assert cfg["aggregation"]["abs_S"] is True
    assert cfg["aggregation"]["normalization"] == "S_AB_row_sum"
    assert cfg["fs"]["flame_tip_T_fraction"] == pytest.approx(0.99)


def test_defaults_and_paths_resolve(tmp_path):
    cfg = validate_config(minimal(), tmp_path)
    assert cfg["mechanism"] == str((tmp_path / "mech.yaml").resolve())
    assert cfg["output_dir"] == str((tmp_path / "results").resolve())
    assert cfg["idt"]["precomputed"]["Y"] == str((tmp_path / "Y.npy").resolve())
    assert cfg["device"] == "auto"
    assert cfg["ranking"]["combine"] == "auto"
    assert cfg["_config_dir"] == str(tmp_path.resolve())


def test_unknown_key_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="sovler"):
        validate_config({"mechanism": "m.yaml", "idt": {"enabled": True, "sovler": {}}}, tmp_path)


def test_wrong_type_and_choice(tmp_path):
    with pytest.raises(ConfigError, match="device"):
        validate_config(minimal(device=3), tmp_path)
    with pytest.raises(ConfigError, match="dtype"):
        validate_config(minimal(dtype="float16"), tmp_path)
    with pytest.raises(ConfigError, match="boolean"):
        validate_config(minimal(sensitivity={"max_substeps": True}), tmp_path)


def test_mode_requirements(tmp_path):
    with pytest.raises(ConfigError, match="idt.solver"):
        validate_config({"mechanism": "m.yaml", "idt": {"enabled": True}}, tmp_path)
    with pytest.raises(ConfigError, match="idt.precomputed"):
        validate_config({"mechanism": "m.yaml", "idt": {"enabled": True, "source": "precomputed"}}, tmp_path)
    with pytest.raises(ConfigError, match="fs.precomputed"):
        validate_config({"mechanism": "m.yaml", "fs": {"enabled": True, "source": "precomputed"}}, tmp_path)
    with pytest.raises(ConfigError, match="at least one mode"):
        validate_config({"mechanism": "m.yaml"}, tmp_path)


def test_new_validations(tmp_path):
    with pytest.raises(ConfigError, match="max_substeps"):
        validate_config(minimal(sensitivity={"substep_tol": 1e-2, "max_substeps": 1}), tmp_path)
    validate_config(minimal(sensitivity={"substep_tol": None, "max_substeps": 1}), tmp_path)
    with pytest.raises(ConfigError, match="substep_tol"):
        validate_config(minimal(sensitivity={"substep_tol": 0.0}), tmp_path)
    with pytest.raises(ConfigError, match="top_n_print"):
        validate_config(minimal(ranking={"top_n_print": -1}), tmp_path)
    fs = {"enabled": True, "solver": {"fuel": {"CH4": 1.0}, "oxidizer": {"O2": 1.0}, "P_atm": [1], "phi": [1.0],
                                      "transport_model": "mixture"}}
    with pytest.raises(ConfigError, match="transport_model"):
        validate_config({"mechanism": "m.yaml", "fs": fs}, tmp_path)
    fs["solver"]["transport_model"] = "multicomponent"
    fs["solver"]["flame_tip_T_fraction"] = "abc"
    with pytest.raises(ConfigError, match="flame_tip_T_fraction"):
        validate_config({"mechanism": "m.yaml", "fs": fs}, tmp_path)


def test_load_config_accepts_bom_and_utf16(tmp_path):
    from gasig.config import public_config

    text = json.dumps(minimal(output_dir="out"))
    (tmp_path / "bom.json").write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    (tmp_path / "u16.json").write_bytes(text.encode("utf-16"))
    for name in ("bom.json", "u16.json"):
        cfg = load_config(tmp_path / name)
        assert cfg["output_dir"] == str((tmp_path / "out").resolve())
        assert "_config_dir" not in public_config(cfg)
    (tmp_path / "list.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(tmp_path / "list.json")


def test_flame_tip_alias_and_range(tmp_path):
    base = {"mechanism": "m.yaml", "fs": {"enabled": True, "solver": {
        "fuel": {"CH4": 1.0}, "oxidizer": {"O2": 0.21, "N2": 0.79}, "P_atm": [1], "phi": [1.0],
        "flame_tip_T_fraction": 0.95}}}
    cfg = validate_config(base, tmp_path)
    assert cfg["fs"]["flame_tip_T_fraction"] == pytest.approx(0.95)
    assert cfg["fs"]["solver"]["flame_tip_T_fraction"] == pytest.approx(0.95)
    base["fs"]["solver"]["flame_tip_T_fraction"] = None
    cfg = validate_config(base, tmp_path)
    assert cfg["fs"]["flame_tip_T_fraction"] is None
    base["fs"]["solver"]["flame_tip_T_fraction"] = 1.5
    with pytest.raises(ConfigError, match="flame_tip_T_fraction"):
        validate_config(base, tmp_path)


def test_load_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(minimal(output_dir="out")), encoding="utf-8")
    cfg = load_config(path)
    assert cfg["output_dir"] == str((tmp_path / "out").resolve())
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(tmp_path / "bad.json")
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.json")


def test_schema_rows_cover_every_leaf():
    keys = {row[0] for row in schema_rows()}
    for expected in ("mechanism", "sensitivity.rate_law", "idt.solver.fuel", "fs.precomputed.velocity",
                     "aggregation.normalization", "ranking.hybrid_weight_idt"):
        assert expected in keys
