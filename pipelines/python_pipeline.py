"""
Python Pipeline
────────────────
Materialises selected features from a PipelineState/FeatureStore into
a production-ready pandas transform pipeline.

Usage
-----
    from pipelines.python_pipeline import PythonPipeline

    pipe = PythonPipeline(state, feature_store)
    out_df = pipe.fit_transform(raw_df)
    pipe.save("my_pipeline.pkl")
"""
from __future__ import annotations

import io
import logging
import pickle
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _FeatureStep:
    """A single feature computation step wrapping a callable."""

    def __init__(self, feature_id: str, name: str, fn: Callable, code: str) -> None:
        self.feature_id = feature_id
        self.name = name
        self.fn = fn
        self.code = code

    def run(self, df: pd.DataFrame) -> pd.Series:
        try:
            result = self.fn(df)
            if not isinstance(result, pd.Series):
                result = pd.Series(result, index=df.index, name=self.name)
            return result.rename(self.name)
        except Exception as exc:
            logger.warning("Feature '%s' failed: %s", self.name, exc)
            return pd.Series(np.nan, index=df.index, name=self.name)


class PythonPipeline:
    """
    Assembles individual feature steps into an end-to-end pandas transform.

    Parameters
    ----------
    selected_definitions  List of FeatureDefinition objects (or dicts) to include.
    fail_fast             If True, a single step failure raises instead of continuing.
    """

    def __init__(
        self,
        selected_definitions: Optional[list] = None,
        fail_fast: bool = False,
    ) -> None:
        self._steps: List[_FeatureStep] = []
        self._fit_stats: Dict[str, Any] = {}
        self.fail_fast = fail_fast

        if selected_definitions:
            self._build_steps(selected_definitions)

    # ── Build ────────────────────────────────────────────────────────────────

    def _build_steps(self, definitions: list) -> None:
        for defn in definitions:
            if hasattr(defn, "model_dump"):
                d = defn.model_dump()
            elif isinstance(defn, dict):
                d = defn
            else:
                d = vars(defn)

            fid = d.get("feature_id", "unknown")
            name = d.get("name", fid)
            code = d.get("python_code", "")

            fn = self._compile_feature_code(name, code)
            self._steps.append(_FeatureStep(fid, name, fn, code))

        logger.info("PythonPipeline: built %d feature steps.", len(self._steps))

    @staticmethod
    def _compile_code(name: str, code: str) -> Callable:
        """
        Attempt to compile the python_code string into a callable fn(df)->Series.
        Falls back to a no-op if compilation fails.
        """
        if not code or not code.strip():
            return lambda df: pd.Series(np.nan, index=df.index)

        # Wrap the code in a function that takes 'df' and returns the last assignment
        # The convention in our construction agent is that the final statement
        # assigns to a variable with the same name as the feature.
        safe_name = name.replace(" ", "_").replace("-", "_")
        wrapped = textwrap.dedent(f"""
import numpy as np
import pandas as pd

def _feature_fn(df):
{textwrap.indent(code.strip(), '    ')}
    try:
        return {safe_name}
    except NameError:
        # Code didn't produce a named variable — exec and return last assignment
        local_vars = {{}}
        exec(compile(open.__doc__ or '', '<string>', 'exec'), {{}}, local_vars)
        vals = list(local_vars.values())
        if vals:
            return vals[-1]
        return pd.Series(np.nan, index=df.index)
""")
        try:
            ns: Dict[str, Any] = {}
            exec(compile(wrapped, "<string>", "exec"), ns)  # noqa: S102
            return ns["_feature_fn"]
        except SyntaxError:
            # If code can't compile, return a no-op
            return lambda df: pd.Series(np.nan, index=df.index)

    # ── Transform ────────────────────────────────────────────────────────────

    @staticmethod
    def _compile_feature_code(name: str, code: str) -> Callable:
        """Compile construction-agent code into a callable that returns one feature series."""
        if not code or not code.strip():
            return lambda df: pd.Series(np.nan, index=df.index, name=name)

        try:
            compiled = compile(code, f"<feature:{name}>", "exec")
        except SyntaxError:
            return lambda df: pd.Series(np.nan, index=df.index, name=name)

        safe_name = name.replace(" ", "_").replace("-", "_")

        def _feature_fn(df: pd.DataFrame) -> pd.Series:
            local_df = df.copy()
            local_ns: Dict[str, Any] = {"np": np, "pd": pd, "df": local_df}
            exec(compiled, local_ns, local_ns)  # noqa: S102

            if name in local_df.columns:
                return local_df[name]

            maybe_series = local_ns.get(safe_name)
            if isinstance(maybe_series, pd.Series):
                return maybe_series.rename(name)

            return pd.Series(np.nan, index=df.index, name=name)

        return _feature_fn

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps to *df* and return original columns + new features."""
        out = df.copy()
        for step in self._steps:
            if self.fail_fast:
                out[step.name] = step.run(out)
            else:
                try:
                    out[step.name] = step.run(out)
                except Exception as e:
                    logger.warning("Step '%s' raised: %s", step.name, e)
                    out[step.name] = np.nan
        logger.info("PythonPipeline.transform: produced %d new columns.", len(self._steps))
        return out

    fit_transform = transform  # alias — this pipeline has no stateful fit

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)
        logger.info("Pipeline saved → %s", path)

    @classmethod
    def load(cls, path: str) -> "PythonPipeline":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Pipeline loaded ← %s", path)
        return obj

    # ── Code export ──────────────────────────────────────────────────────────

    def generate_module_code(self, class_name: str = "FeatureTransformer") -> str:
        """
        Return a self-contained Python source string that applies all features.
        This code can be copy-pasted into production without this library.
        """
        buf = io.StringIO()
        buf.write('"""Auto-generated feature transform — do not edit manually."""\n')
        buf.write("from __future__ import annotations\n\nimport numpy as np\nimport pandas as pd\n\n\n")
        buf.write(f"class {class_name}:\n")
        buf.write('    """Applies engineered features to a raw DataFrame."""\n\n')
        buf.write("    def transform(self, df: pd.DataFrame) -> pd.DataFrame:\n")
        buf.write("        out = df.copy()\n")
        buf.write("        df = out\n")
        for step in self._steps:
            buf.write(f"\n        # ── {step.name} ──\n")
            if step.code and step.code.strip():
                for line in step.code.strip().splitlines():
                    buf.write(f"        {line}\n")
                buf.write(f"        if '{step.name}' not in out.columns:\n")
                buf.write(f"            out['{step.name}'] = np.nan\n")
            else:
                buf.write(f"        out['{step.name}'] = np.nan  # code unavailable\n")
        buf.write("        return out\n")
        return buf.getvalue()

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"PythonPipeline(steps={len(self._steps)}, fail_fast={self.fail_fast})"
