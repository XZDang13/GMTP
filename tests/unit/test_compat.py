import sys

from gmtp.integrations.ref2act import compat


def test_import_module_uses_ref2act_src_fallback(tmp_path, monkeypatch):
    package_root = tmp_path / "src"
    ref2act_dir = package_root / "ref2act"
    ref2act_dir.mkdir(parents=True)
    (ref2act_dir / "__init__.py").write_text("VALUE = 123\n", encoding="utf-8")

    monkeypatch.setattr(compat, "DEFAULT_REF2ACT_SRC", package_root)
    if "ref2act" in sys.modules:
        del sys.modules["ref2act"]
    sys.path[:] = [path for path in sys.path if str(package_root) != path]

    monkeypatch.setattr(compat, "_module_exists", lambda module_name: str(package_root) in sys.path)
    compat._ensure_ref2act_on_path()
    assert str(package_root) in sys.path
