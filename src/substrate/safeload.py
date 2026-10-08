"""Reading experiment files (YAML or JSON) without trusting them.

`yaml.safe_load` cannot construct arbitrary Python objects, but it still expands anchors and aliases, so a few hundred bytes can describe a
document of billions of nodes ("billion laughs"), and a deeply nested document exhausts the stack. Experiments need neither anchors nor deep
nesting, so both are refused, along with files above the size limit. JSON accepts `NaN` and `Infinity` by default; those are refused too. Every
failure is a ValidationError that names the file, so the command line and the GUI report it like any other bad input.
"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import ValidationError
from .limits import require

MAX_DEPTH = 40                 # nesting levels of mappings and lists; the deepest shipped experiment uses 5
MAX_EVENTS = 200_000           # YAML syntax events; the largest shipped file has a few hundred


def _check_yaml(text: str, source: str) -> None:
    """Scan the YAML event stream (nothing is constructed, aliases are not expanded) for aliases, excessive depth and excessive length."""
    import yaml
    depth = events = 0
    try:
        for event in yaml.parse(text, Loader=yaml.SafeLoader):
            events += 1
            if events > MAX_EVENTS:
                raise ValidationError(f"{source}: more than {MAX_EVENTS:,} YAML events")
            if isinstance(event, yaml.AliasEvent):
                raise ValidationError(f"{source}: YAML aliases (*name) are not accepted in experiment files")
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
                if depth > MAX_DEPTH:
                    raise ValidationError(f"{source}: nested more than {MAX_DEPTH} levels deep")
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
    except yaml.YAMLError as error:
        raise ValidationError(f"{source}: not valid YAML ({error})") from None


def load_yaml(text: str, source: str = "YAML text"):
    """The parsed document, after `_check_yaml`."""
    import yaml
    _check_yaml(text, source)
    return yaml.safe_load(text)


def _refuse_constant(name: str):
    raise ValueError(f"'{name}' is not a number an experiment can use")


def load_json(text: str, source: str = "JSON text"):
    try:
        return json.loads(text, parse_constant=_refuse_constant)
    except RecursionError:
        raise ValidationError(f"{source}: nested too deeply") from None
    except ValueError as error:                       # a JSONDecodeError is a ValueError
        raise ValidationError(f"{source}: not valid JSON ({error})") from None


def read_text_file(path: str | Path) -> str:
    """The file's text, refusing one above the size limit before it is read."""
    path = Path(path)
    require("max_spec_bytes", path.stat().st_size, f"{path.name}: file size in bytes")
    return path.read_text(encoding="utf-8")


def read_spec_file(path: str | Path):
    """An experiment file as a plain object: YAML for .yaml/.yml, JSON otherwise."""
    path = Path(path)
    text = read_text_file(path)
    return load_yaml(text, path.name) if path.suffix.lower() in (".yaml", ".yml") else load_json(text, path.name)


__all__ = ["load_yaml", "load_json", "read_text_file", "read_spec_file"]
