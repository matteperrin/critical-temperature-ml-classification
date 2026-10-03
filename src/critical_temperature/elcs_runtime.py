"""Load only the bundled eLCS, with isolated corrected/unmodified variants.

Absolute upstream imports are relocated into a unique private namespace in a
throwaway copy. Neither installed skeLCS modules nor bundled files are mutated.
"""
import hashlib
import importlib.util
from email.parser import Parser
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import uuid

BUNDLE = Path(__file__).resolve().parents[2] / "third_party" / "scikit-eLCS"
VARIANTS = ("corrected", "unmodified")
CORRECTED = b"        self.majorityClass = max(self.classCount, key=self.classCount.get)"
UNMODIFIED = b"        self.majorityClass = max(self.classCount)"


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def effective_sources(sources, variant):
    """Undo only the known local repair, never guess at an unknown source."""
    if variant not in VARIANTS:
        raise ValueError("Unknown eLCS library variant.")
    result = dict(sources)
    data = result.get("DataManagement.py", b"")
    if data.splitlines().count(CORRECTED) != 1 or UNMODIFIED in data.splitlines():
        raise ValueError("Bundled eLCS majority repair is not the exact known patch.")
    if variant == "unmodified":
        result["DataManagement.py"] = data.replace(CORRECTED, UNMODIFIED, 1)
    return result


def snapshot(variant):
    """Return effective source bytes plus reproducible source/metadata provenance."""
    package = BUNDLE / "skeLCS"
    sources = {str(path.relative_to(package)).replace("\\", "/"): path.read_bytes()
               for path in sorted(package.rglob("*.py"))}
    effective = effective_sources(sources, variant)
    metadata_paths = ("setup.py", "setup.cfg", "scikit_eLCS.egg-info/PKG-INFO")
    metadata = {name: (BUNDLE / name).read_bytes() for name in metadata_paths}
    info = Parser().parsestr(metadata[metadata_paths[-1]].decode("utf-8"))
    if info["Name"] != "scikit-eLCS" or info["Version"] != "1.2.4":
        raise ValueError("Unexpected bundled eLCS package metadata.")
    provenance = {
        "loader": "isolated-bundled-namespace-v1",
        "source_root": str(BUNDLE.resolve()),
        "package_name": info["Name"], "package_version": info["Version"],
        "library_variant": variant,
        "source_sha256": {name: sha256(data) for name, data in sources.items()},
        "effective_source_sha256": {name: sha256(data) for name, data in effective.items()},
        "metadata_sha256": {name: sha256(data) for name, data in metadata.items()},
        "majority_repair": {
            "file": "skeLCS/DataManagement.py",
            "upstream_line": UNMODIFIED.decode().strip(),
            "corrected_line": CORRECTED.decode().strip(),
            "applied": variant == "corrected",
        },
    }
    return effective, provenance


def provenance(variant="corrected"):
    return snapshot(variant)[1]


def validate_configuration(config):
    """Historical configs without source provenance cannot unlock final testing."""
    if "runtime" not in config or "library_variant" not in config:
        raise ValueError("Missing frozen eLCS runtime provenance; rerun development CV in a new directory.")
    current = provenance(config["library_variant"])
    if config["runtime"] != current:
        raise ValueError("Bundled eLCS runtime source or metadata changed from frozen provenance; rerun development CV.")


def create_model(parameters, variant="corrected", expected=None):
    sources, current = snapshot(variant)
    if expected is not None and expected != current:
        raise ValueError("Bundled eLCS runtime changed from frozen provenance.")
    namespace = "_holdout_elcs_" + uuid.uuid4().hex
    try:
        with TemporaryDirectory(prefix="holdout-elcs-") as directory:
            package = Path(directory) / namespace
            package.mkdir()
            for name, content in sources.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                # Upstream uses absolute intra-package imports; relocate only those.
                path.write_bytes(content.replace(b"from skeLCS.", f"from {namespace}.".encode()))
            spec = importlib.util.spec_from_file_location(
                namespace, package / "__init__.py", submodule_search_locations=[str(package)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = module
            spec.loader.exec_module(module)
            for name, loaded in list(sys.modules.items()):
                if name == namespace or name.startswith(namespace + "."):
                    if not Path(loaded.__file__).resolve().is_relative_to(package.resolve()):
                        raise ValueError("eLCS imported from outside its isolated bundled copy.")
            return module.eLCS(**parameters)
    finally:
        # Classes retain module globals. There are no deferred intra-package imports;
        # remove registrations so repeated fits do not retain entire module trees.
        for name in list(sys.modules):
            if name == namespace or name.startswith(namespace + "."):
                del sys.modules[name]
