from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_REPO_ROOT = Path(__file__).resolve().parents[1]


_FORTRAN_COMMENT_PATTERN = re.compile(r"!.*$")
_RE_REPEATED_VALUE = re.compile(r"^\s*(\d+)\s*\*\s*(.+)$")
_RE_STRING_QUOTE = re.compile(r'^([\'"])(.*)\1$')


def _strip_inline_comment(line: str) -> str:
    """Remove Fortran comments respecting that ``!`` starts a comment."""
    return _FORTRAN_COMMENT_PATTERN.sub("", line)


def _merge_lines_with_continuations(lines: Iterable[str]) -> list[str]:
    """Collapse Fortran continuation lines into single logical statements."""
    statements: list[str] = []
    buffer: list[str] = []
    for raw in lines:
        no_comment = _strip_inline_comment(raw).rstrip()
        if not no_comment.strip():
            continue
        segment = no_comment.strip()
        trailing_cont = segment.endswith("&")
        if trailing_cont:
            segment = segment[:-1].rstrip()
        if buffer:
            buffer.append(segment)
        else:
            buffer = [segment]
        if not trailing_cont:
            statements.append(" ".join(buffer).strip())
            buffer = []
    if buffer:
        statements.append(" ".join(buffer).strip())
    return statements


def _extract_assignments(statements: Sequence[str]) -> dict[str, str]:
    """Map variable names (lowercase) to their default Fortran literal."""
    assignments: dict[str, str] = {}
    for stmt in statements:
        if "::" not in stmt or "=" not in stmt:
            continue
        _, remainder = stmt.split("::", 1)
        if "=" not in remainder:
            continue
        var_part, value_part = remainder.split("=", 1)
        var_name = var_part.strip()
        if "(" in var_name:
            var_name = var_name.split("(", 1)[0].strip()
        var_key = var_name.lower()
        assignments[var_key] = value_part.strip()
    return assignments


def _extract_namelists(statements: Sequence[str]) -> dict[str, list[str]]:
    """Parse ``namelist`` statements and return mapping of name -> variables."""
    namelists: dict[str, list[str]] = {}
    for stmt in statements:
        if not stmt.lower().startswith("namelist"):
            continue
        parts = stmt.split("/", 2)
        if len(parts) < 3:
            continue
        name = parts[1].strip().lower()
        body = parts[2]
        variables = [item.strip() for item in body.split(",") if item.strip()]
        namelists[name] = variables
    return namelists


def _split_fortran_list(value: str) -> list[str]:
    """Split a comma-separated list while respecting quoted substrings."""
    tokens: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    i = 0
    while i < len(value):
        ch = value[i]
        if in_quote:
            current.append(ch)
            if ch == in_quote:
                if i + 1 < len(value) and value[i + 1] == in_quote:
                    current.append(in_quote)
                    i += 1
                else:
                    in_quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = ch
            current.append(ch)
        elif ch == ",":
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
        else:
            current.append(ch)
        i += 1
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _parse_fortran_scalar(token: str) -> Any:
    """Convert a single Fortran literal token into a Python value."""
    token = token.strip()
    if not token:
        return None

    repeated = _RE_REPEATED_VALUE.match(token)
    if repeated:
        count = int(repeated.group(1))
        value = _parse_fortran_scalar(repeated.group(2))
        return [value for _ in range(count)]

    token_lower = token.lower()
    if token_lower in {".true.", "t"}:
        return True
    if token_lower in {".false.", "f"}:
        return False

    match = _RE_STRING_QUOTE.match(token.strip())
    if match:
        body = match.group(2)
        quote = match.group(1)
        escaped = body.replace(quote * 2, quote)
        return escaped

    cleaned = token.replace("d", "e").replace("D", "e")
    try:
        if any(sym in cleaned.lower() for sym in ("e", ".")):
            return float(cleaned)
        return int(cleaned, 10)
    except ValueError:
        return cleaned.strip()


def _parse_fortran_value(value: str) -> Any:
    """Parse a Fortran literal or array constructor into Python types."""
    literal = value.strip()
    if literal.endswith(","):
        literal = literal[:-1].rstrip()
    if literal.startswith("(/") and literal.endswith("/)"):
        inner = literal[2:-2].strip()
        values = []
        for token in _split_fortran_list(inner):
            parsed = _parse_fortran_scalar(token)
            if isinstance(parsed, list):
                values.extend(parsed)
            elif parsed is not None:
                values.append(parsed)
        return values

    tokens = _split_fortran_list(literal)
    parsed_values: list[Any] = []
    for token in tokens:
        parsed = _parse_fortran_scalar(token)
        if isinstance(parsed, list):
            parsed_values.extend(parsed)
        elif parsed is not None:
            parsed_values.append(parsed)
    if not parsed_values:
        return None
    if len(parsed_values) == 1:
        return parsed_values[0]
    return parsed_values


def _load_common_nml_defaults(base_dir: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Extract namelist variables and defaults from ``common_nml.f90``."""
    module_path = base_dir / "scale" / "scale-letkf" / "scale" / "common" / "common_nml.f90"
    statements = _merge_lines_with_continuations(module_path.read_text().splitlines())
    assignments = _extract_assignments(statements)
    namelists = _extract_namelists(statements)

    defaults: dict[str, dict[str, Any]] = {}
    for group, variables in namelists.items():
        defaults[group] = {}
        for var in variables:
            key = var.lower()
            if key not in assignments:
                continue
            defaults[group][key] = _parse_fortran_value(assignments[key])
    return namelists, defaults


def _load_common_scale_metadata(base_dir: Path) -> dict[str, list[str]]:
    """Fetch variable names for 3D/2D state variables from ``common_scale.f90``."""
    module_path = base_dir / "scale" / "scale-letkf" / "scale" / "common" / "common_scale.f90"
    statements = _merge_lines_with_continuations(module_path.read_text().splitlines())
    assignments = _extract_assignments(statements)

    metadata: dict[str, list[str]] = {}
    for key in ("v3d_name", "v2d_name", "v3dd_name", "v2dd_name"):
        if key in assignments:
            values = _parse_fortran_value(assignments[key])
            metadata[key] = [str(v).strip() for v in values]
    return metadata


def get_state_variable_metadata() -> dict[str, list[str]]:
    """Return canonical SCALE state variable names keyed by dimension."""
    return _load_common_scale_metadata(_REPO_ROOT)




def _parse_namelist_file(path: Path) -> dict[str, dict[str, Any]]:
    """Parse a Fortran namelist configuration file into nested dictionaries."""
    groups: dict[str, dict[str, Any]] = {}

    current_group: str | None = None
    buffer: list[str] = []

    def flush_buffer() -> None:
        if current_group is None or not buffer:
            return
        statement = " ".join(buffer).strip()
        buffer.clear()
        if "=" not in statement:
            return
        var_part, value_part = statement.split("=", 1)
        var_name = var_part.strip().lower()
        value = _parse_fortran_value(value_part.strip())
        groups.setdefault(current_group, {})[var_name] = value

    for raw_line in path.read_text().splitlines():
        stripped = _strip_inline_comment(raw_line).strip()
        if not stripped:
            continue
        if stripped.startswith("&"):
            flush_buffer()
            current_group = stripped[1:].split()[0].lower()
            groups.setdefault(current_group, {})
            continue
        if stripped.startswith("/"):
            flush_buffer()
            current_group = None
            continue
        if current_group is None:
            continue
        has_assignment = "=" in stripped
        if has_assignment and buffer:
            flush_buffer()
        buffer.append(stripped.rstrip(","))
    flush_buffer()
    return groups


def _replace_domain_token(template: str, domain_id: int, width: int = 2) -> str:
    """Replace ``<domain>`` placeholder with zero-padded integer."""
    return template.replace("<domain>", f"{domain_id:0{width}d}")


def _gather_group_defaults(
    namelists: Mapping[str, Sequence[str]],
    defaults: Mapping[str, Mapping[str, Any]],
    groups: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Select defaults for the requested groups."""
    selected: dict[str, dict[str, Any]] = {}
    for group in groups:
        lower = group.lower()
        group_defaults = defaults.get(lower, {})
        selected[lower] = {k: v for k, v in group_defaults.items()}
    return selected


def _merge_group_values(
    base_groups: dict[str, dict[str, Any]],
    override: Mapping[str, Mapping[str, Any]],
) -> None:
    """Update ``base_groups`` with values from the override mapping."""
    for group, values in override.items():
        lower = group.lower()
        target = base_groups.setdefault(lower, {})
        for key, value in values.items():
            target[key.lower()] = value


@dataclass(frozen=True)
class GeneralSettings:
    log: Mapping[str, Any]
    model: Mapping[str, Any]
    ensemble: Mapping[str, Any]
    process: Mapping[str, Any]


@dataclass(frozen=True)
class ObservationSettings:
    inputs: Mapping[str, Any]
    errors: Mapping[str, Any]


@dataclass(frozen=True)
class LocalizationSettings:
    observation: Mapping[str, Any]
    variable: Mapping[str, Any]


@dataclass(frozen=True)
class RadarSettings:
    core: Mapping[str, Any]


@dataclass(frozen=True)
class LetkfParameters:
    general: GeneralSettings
    observation: ObservationSettings
    letkf: Mapping[str, Any]
    localization: LocalizationSettings
    monitoring: Mapping[str, Any]
    radar: RadarSettings
    source_files: tuple[Path, ...]
    var_metadata: Mapping[str, list[str]]
    raw_groups: Mapping[str, Mapping[str, Any]]


_RELEVANT_GROUPS = [
    "param_log",
    "param_model",
    "param_ensemble",
    "param_process",
    "param_obsope",
    "param_letkf",
    "param_letkf_obs",
    "param_letkf_var_local",
    "param_letkf_monitor",
    "param_letkf_radar",
    "param_obs_error",
]


def load_letkf_parameters(
    config_path: str | Path,
    *,
    domain: int | None = None,
) -> LetkfParameters:
    """Load LETKF-related configuration parameters into structured mappings."""
    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_file} does not exist")

    repo_root = _REPO_ROOT
    namelist_defs, default_values = _load_common_nml_defaults(repo_root)
    state_metadata = _load_common_scale_metadata(repo_root)

    defaults = _gather_group_defaults(namelist_defs, default_values, _RELEVANT_GROUPS)
    groups = {name: values.copy() for name, values in defaults.items()}

    parsed_main = _parse_namelist_file(config_file)
    _merge_group_values(groups, parsed_main)

    general = groups.get("param_ensemble", {})
    process = groups.get("param_process", {})
    conf_template = general.get("conf_files") or defaults.get("param_ensemble", {}).get("conf_files")
    num_domain = int(process.get("num_domain") or defaults.get("param_process", {}).get("num_domain", 1))

    source_files = [config_file]

    if conf_template:
        if isinstance(conf_template, (tuple, list)):
            conf_template = conf_template[0]
        chosen_domains: Iterable[int]
        if domain is None:
            chosen_domains = range(1, num_domain + 1)
        else:
            chosen_domains = (domain,)
        for dom_id in chosen_domains:
            domain_path = _replace_domain_token(str(conf_template), dom_id)
            domain_file = Path(domain_path)
            if not domain_file.is_absolute():
                domain_file = config_file.parent / domain_file
            if not domain_file.exists():
                continue
            parsed_domain = _parse_namelist_file(domain_file)
            if parsed_domain:
                _merge_group_values(groups, parsed_domain)
                source_files.append(domain_file.resolve())
            if domain is not None:
                break

    general_settings = GeneralSettings(
        log=groups.get("param_log", {}),
        model=groups.get("param_model", {}),
        ensemble=groups.get("param_ensemble", {}),
        process=groups.get("param_process", {}),
    )
    observation_settings = ObservationSettings(
        inputs=groups.get("param_obsope", {}),
        errors=groups.get("param_obs_error", {}),
    )
    localization_settings = LocalizationSettings(
        observation=groups.get("param_letkf_obs", {}),
        variable=groups.get("param_letkf_var_local", {}),
    )
    radar_settings = RadarSettings(
        core=groups.get("param_letkf_radar", {}),
    )

    return LetkfParameters(
        general=general_settings,
        observation=observation_settings,
        letkf=groups.get("param_letkf", {}),
        localization=localization_settings,
        monitoring=groups.get("param_letkf_monitor", {}),
        radar=radar_settings,
        source_files=tuple(source_files),
        var_metadata={
            key: value
            for key, value in state_metadata.items()
            if key in {"v3d_name", "v2d_name", "v3dd_name", "v2dd_name"}
        },
        raw_groups=groups,
    )


__all__ = [
    "GeneralSettings",
    "LocalizationSettings",
    "ObservationSettings",
    "RadarSettings",
    "LetkfParameters",
    "load_letkf_parameters",
    "get_state_variable_metadata",
]
