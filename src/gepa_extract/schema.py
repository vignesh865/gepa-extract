"""The frozen skeleton and the mutable description layer.

The central invariant of this project: **only field descriptions evolve.**
Keys, types, nesting, required-ness, the task prompt and any organisational
policy text must be structurally impossible for the optimiser to reach.

That is enforced by construction rather than by validation. The optimiser's
candidate is a flat ``{field_path: description}`` mapping, and the schema is
reassembled at evaluation time by dropping those descriptions into a skeleton
the optimiser never sees. There is no code path by which a mutation could
alter structure, because structure is not part of what is mutated.
``fingerprint()`` exists to prove that in a test, not to defend it at runtime.

Field paths use ``.`` for object properties and ``[]`` for array descent::

    invoice_number
    vendor.name
    line_items[].quantity
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

__all__ = ["ExtractionSchema", "FieldSpec", "resolve_path"]


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One leaf field of the schema -- one independently-evolved component."""

    path: str
    json_type: str
    required: bool
    in_array: bool
    seed_description: str

    @property
    def component_name(self) -> str:
        """The name GEPA knows this field by. Field path and component name are
        deliberately identical, so traces are readable without a lookup table."""
        return self.path


class ExtractionSchema:
    """A JSON extraction schema split into a frozen skeleton and seed descriptions.

    Args:
        schema: A JSON Schema object. Leaf ``description`` values are lifted out
            as the seed candidate; everything else becomes the frozen skeleton.
        task_prompt: Instructions sent alongside the schema on every extraction
            call. Frozen -- never part of the candidate.
        policy_text: Organisational/safety text appended to the task prompt.
            Frozen, and kept separate from ``task_prompt`` so it is obvious in
            review that no code path mutates it.
    """

    def __init__(self, schema: dict[str, Any], *, task_prompt: str, policy_text: str = "") -> None:
        if schema.get("type") != "object":
            raise ValueError("Top-level extraction schema must be of type 'object'.")
        self.task_prompt = task_prompt
        self.policy_text = policy_text

        fields: list[FieldSpec] = []
        self._skeleton = copy.deepcopy(schema)
        _collect(self._skeleton, path="", required=True, in_array=False, out=fields)
        if not fields:
            raise ValueError("Extraction schema contains no leaf fields to optimise.")

        self.fields: tuple[FieldSpec, ...] = tuple(fields)
        self._by_path = {f.path: f for f in self.fields}
        if len(self._by_path) != len(self.fields):
            raise ValueError("Duplicate field paths in schema.")

    @property
    def field_paths(self) -> tuple[str, ...]:
        return tuple(f.path for f in self.fields)

    def field(self, path: str) -> FieldSpec:
        return self._by_path[path]

    def seed_candidate(self) -> dict[str, str]:
        """The starting candidate: the hand-written descriptions we intend to beat."""
        return {f.path: f.seed_description for f in self.fields}

    def bind(self, candidate: dict[str, str]) -> dict[str, Any]:
        """Reassemble a full JSON Schema from the frozen skeleton + candidate.

        Unknown keys are rejected rather than ignored: a candidate key that does
        not name a field means the optimiser and the schema have drifted apart,
        and silently dropping it would hide that.
        """
        unknown = set(candidate) - set(self._by_path)
        if unknown:
            raise KeyError(f"Candidate contains keys that are not schema fields: {sorted(unknown)}")

        bound = copy.deepcopy(self._skeleton)
        for path, description in candidate.items():
            node = _schema_node(bound, path)
            node["description"] = description
        return bound

    def fingerprint(self) -> str:
        """Canonical form of the structure, with every description removed.

        Two schemas share a fingerprint iff they are structurally identical.
        Used by tests to assert that optimisation cannot move structure.
        """
        return _fingerprint(self._skeleton)

    def structural_fingerprint_of(self, schema: dict[str, Any]) -> str:
        return _fingerprint(schema)

    def flatten(self, data: Any) -> dict[str, Any]:
        """Project an extracted or gold document onto ``{field_path: value}``.

        Array fields yield a list of per-item values, so that
        ``line_items[].quantity`` is a list aligned with the item order.
        """
        return {f.path: resolve_path(data, f.path) for f in self.fields}


def resolve_path(data: Any, path: str) -> Any:
    """Read ``path`` out of a decoded JSON document. Missing -> None."""
    return _resolve(data, _tokens(path))


def _tokens(path: str) -> list[str]:
    return path.split(".") if path else []


def _resolve(node: Any, tokens: list[str]) -> Any:
    if not tokens:
        return node
    token, rest = tokens[0], tokens[1:]
    is_array = token.endswith("[]")
    key = token[:-2] if is_array else token

    if not isinstance(node, dict):
        return None
    child = node.get(key)
    if is_array:
        if not isinstance(child, list):
            return None
        return [_resolve(item, rest) for item in child]
    return _resolve(child, rest)


def _schema_node(schema: dict[str, Any], path: str) -> dict[str, Any]:
    node = schema
    for token in _tokens(path):
        is_array = token.endswith("[]")
        key = token[:-2] if is_array else token
        node = node["properties"][key]
        if is_array:
            node = node["items"]
    return node


def _collect(
    node: dict[str, Any],
    *,
    path: str,
    required: bool,
    in_array: bool,
    out: list[FieldSpec],
) -> None:
    """Walk the schema, lifting leaf descriptions out into ``out``.

    Mutates ``node`` in place, stripping the descriptions it lifts -- the caller
    owns a deepcopy, and stripping is what makes the remainder a *skeleton*.
    """
    node_type = node.get("type")

    if node_type == "object":
        required_keys = set(node.get("required", []))
        for key, child in node.get("properties", {}).items():
            _collect(
                child,
                path=f"{path}.{key}" if path else key,
                required=key in required_keys,
                in_array=in_array,
                out=out,
            )
        return

    if node_type == "array":
        items = node.get("items")
        if isinstance(items, dict):
            _collect(items, path=f"{path}[]", required=required, in_array=True, out=out)
        return

    out.append(
        FieldSpec(
            path=path,
            json_type=str(node_type or "string"),
            required=required,
            in_array=in_array,
            seed_description=node.pop("description", "").strip(),
        )
    )


def _fingerprint(schema: Any) -> str:
    return json.dumps(_strip_descriptions(schema), sort_keys=True, separators=(",", ":"))


def _strip_descriptions(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_descriptions(v) for k, v in node.items() if k != "description"}
    if isinstance(node, list):
        return [_strip_descriptions(v) for v in node]
    return node
