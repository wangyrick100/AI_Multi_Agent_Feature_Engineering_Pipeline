"""
Feature Store Utility
──────────────────────
Persists, retrieves, and exports engineered feature definitions and
computed feature DataFrames.

Layout on disk:
  store_root/
    registry.json          — JSON list of FeatureDefinition dicts
    dataframes/
      <run_id>.parquet     — computed feature matrix
    pipeline_states/
      <run_id>.pkl         — serialised PipelineState
    python_modules/
      <module_name>.py     — exported standalone Python transform module
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    File-based feature store for caching and exporting engineered features.

    Parameters
    ----------
    store_root   Root directory for the store (created if absent).
    """

    REGISTRY_FILE = "registry.json"
    DF_DIR = "dataframes"
    STATE_DIR = "pipeline_states"
    MODULES_DIR = "python_modules"

    def __init__(self, store_root: str = "feature_store") -> None:
        self.root = Path(store_root)
        self._init_dirs()
        self._registry: List[Dict] = self._load_registry()

    # ── Directory helpers ────────────────────────────────────────────────────

    def _init_dirs(self) -> None:
        for sub in [self.DF_DIR, self.STATE_DIR, self.MODULES_DIR]:
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def _registry_path(self) -> Path:
        return self.root / self.REGISTRY_FILE

    def _load_registry(self) -> List[Dict]:
        p = self._registry_path()
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_registry(self) -> None:
        with open(self._registry_path(), "w", encoding="utf-8") as f:
            json.dump(self._registry, f, indent=2, default=str)

    # ── Public API ───────────────────────────────────────────────────────────

    def register_feature(self, feature_def: Any) -> None:
        """Add or update a feature definition in the registry."""
        existing_ids = {r["feature_id"] for r in self._registry}
        data = feature_def.model_dump() if hasattr(feature_def, "model_dump") else vars(feature_def)
        if data["feature_id"] not in existing_ids:
            self._registry.append(data)
        else:
            # update in-place
            for i, r in enumerate(self._registry):
                if r["feature_id"] == data["feature_id"]:
                    self._registry[i] = data
                    break
        self._save_registry()
        logger.debug("Registered feature %s", data["feature_id"])

    def register_many(self, feature_defs: list) -> None:
        for fd in feature_defs:
            self.register_feature(fd)

    def get_feature_definitions(self, ids: Optional[List[str]] = None) -> List[Dict]:
        """Return all (or filtered) feature definition dicts."""
        if ids is None:
            return self._registry
        id_set = set(ids)
        return [r for r in self._registry if r["feature_id"] in id_set]

    def list_feature_ids(self) -> List[str]:
        return [r["feature_id"] for r in self._registry]

    # ── DataFrame persistence ────────────────────────────────────────────────

    def save_dataframe(
        self, df: pd.DataFrame, run_id: str, compression: str = "snappy"
    ) -> Path:
        """Persist a feature DataFrame as Parquet and return the file path."""
        out = self.root / self.DF_DIR / f"{run_id}.parquet"
        df.to_parquet(out, compression=compression, index=True)
        logger.info("Saved feature DataFrame → %s  shape=%s", out, df.shape)
        return out

    def load_dataframe(self, run_id: str) -> pd.DataFrame:
        path = self.root / self.DF_DIR / f"{run_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No saved DataFrame for run_id='{run_id}'")
        return pd.read_parquet(path)

    def list_dataframe_runs(self) -> List[str]:
        return [p.stem for p in (self.root / self.DF_DIR).glob("*.parquet")]

    # ── Pipeline state persistence ───────────────────────────────────────────

    def save_state(self, state: object, run_id: str) -> Path:
        """Pickle the PipelineState object (excluding large DataFrames)."""
        out = self.root / self.STATE_DIR / f"{run_id}.pkl"
        # Remove constructed_df from __dict__ before pickling to keep it small
        constructed_df = state.__dict__.pop("constructed_df", None)
        with open(out, "wb") as f:
            pickle.dump(state, f, protocol=5)
        if constructed_df is not None:
            state.__dict__["constructed_df"] = constructed_df
            self.save_dataframe(constructed_df, run_id=f"{run_id}_features")
        logger.info("Saved pipeline state → %s", out)
        return out

    def load_state(self, run_id: str) -> object:
        path = self.root / self.STATE_DIR / f"{run_id}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"No saved state for run_id='{run_id}'")
        with open(path, "rb") as f:
            state = pickle.load(f)
        # Restore df if available
        feat_path = self.root / self.DF_DIR / f"{run_id}_features.parquet"
        if feat_path.exists():
            state.__dict__["constructed_df"] = pd.read_parquet(feat_path)
        return state

    # ── Python module export ─────────────────────────────────────────────────

    def export_to_python_module(
        self,
        selected_ids: List[str],
        output_path: str = "features_transform.py",
        module_name: str = "FeaturesTransformer",
    ) -> str:
        """
        Generate a standalone Python transform module containing a class
        with one method per selected feature, plus a transform(df) method
        that applies all of them.

        Returns the generated code as a string (and writes it to output_path).
        """
        defs = self.get_feature_definitions(ids=selected_ids)
        if not defs:
            raise ValueError("No feature definitions found for the given IDs.")

        lines = [
            '"""Auto-generated feature transform module."""',
            "from __future__ import annotations",
            "",
            "import numpy as np",
            "import pandas as pd",
            "",
            "",
            f"class {module_name}:",
            '    """Applies selected engineered features to a raw DataFrame."""',
            "",
            "    def transform(self, df: pd.DataFrame) -> pd.DataFrame:",
            '        """Return df with all engineered features appended."""',
            "        out = df.copy()",
        ]
        for defn in defs:
            fid = defn.get("feature_id", "unknown")
            fname = defn.get("name", fid)
            code = defn.get("python_code", "")
            lines.append(f"        # {fname}")
            if code and code.strip():
                # indent and sanitise
                for code_line in code.strip().splitlines():
                    lines.append(f"        {code_line}")
            else:
                lines.append(f'        out["{fname}"] = np.nan  # code unavailable')
        lines.append("        return out")
        lines.append("")

        source = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(source)
        logger.info("Exported %d features → %s", len(defs), output_path)
        return source

    # ── Utilities ────────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "registered_features": len(self._registry),
            "saved_dataframes": len(self.list_dataframe_runs()),
            "categories": list(
                {r.get("category", "unknown") for r in self._registry}
            ),
        }
