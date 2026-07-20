import json
import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from graphtool.graph.taxonomy import UNCLASSIFIED_NODE_TYPE, normalize_type_name
from graphtool.graph.types import Node
from graphtool.llm.base import LLMClient
from graphtool.llm.types import LLMMessage

ENTITY_RESOLUTION_SYSTEM_PROMPT = (
    "You decide whether a new knowledge graph node refers to the same real-world "
    "entity as one of the provided candidate nodes. Merge only when they are the "
    "same entity, not when they are merely related. Keep organizations, products, "
    "services, APIs, and deployments distinct unless the names clearly refer to "
    "the same entity."
)


class EntityResolutionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["merge", "new"]
    target_node_id: str | None = None
    confidence: float = 0.0
    canonical_label: str | None = None
    aliases_to_add: list[str] = Field(default_factory=list)


def find_normalized_match(
    node: Node,
    canonical_nodes: Sequence[Node],
) -> Node | None:
    incoming_names = _normalized_names(node)
    if not incoming_names:
        return None

    for candidate in canonical_nodes:
        if comparable_node_types(node, candidate):
            if incoming_names & _normalized_names(candidate):
                return candidate
    return None


def same_type_candidates(
    node: Node,
    canonical_nodes: Sequence[Node],
) -> list[Node]:
    return [
        candidate
        for candidate in canonical_nodes
        if comparable_node_types(node, candidate)
    ]


def judge_same_entity(
    llm: LLMClient,
    node: Node,
    candidates: Sequence[tuple[Node, float]],
) -> EntityResolutionDecision:
    payload = {
        "incoming": _node_payload(node),
        "candidates": [
            {
                **_node_payload(candidate),
                "similarity": round(score, 6),
            }
            for candidate, score in candidates
        ],
        "rules": [
            "Return merge only if the incoming node and target candidate are the "
            "same entity.",
            "Do not merge an organization with its product, API, service, "
            "deployment, or partner.",
            "When uncertain, return new.",
        ],
    }
    messages = [
        LLMMessage(role="system", content=ENTITY_RESOLUTION_SYSTEM_PROMPT),
        LLMMessage(
            role="user",
            content=(
                "Resolve the incoming node against the candidates. "
                "Return the structured decision only.\n\n"
                f"{json.dumps(payload, indent=2, sort_keys=True)}"
            ),
        ),
    ]
    return llm.generate_structured(messages, EntityResolutionDecision)


def accepted_target_id(
    decision: EntityResolutionDecision,
    candidate_ids: set[str],
    merge_confidence_threshold: float,
) -> str | None:
    if decision.decision != "merge":
        return None
    if decision.target_node_id not in candidate_ids:
        return None
    if decision.confidence < merge_confidence_threshold:
        return None
    return decision.target_node_id


def comparable_node_types(left: Node, right: Node) -> bool:
    left_type = normalize_type_name(left.type)
    right_type = normalize_type_name(right.type)
    return (
        left_type == right_type
        or left_type == UNCLASSIFIED_NODE_TYPE
        or right_type == UNCLASSIFIED_NODE_TYPE
    )


def _normalized_names(node: Node) -> set[str]:
    return {
        normalized
        for normalized in (
            _normalize_name(name) for name in [node.label, *node.aliases]
        )
        if normalized
    }


def _normalize_name(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold())
    return " ".join(_singularize_word(word) for word in normalized.split())


def _singularize_word(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if (
        len(word) > 4
        and word.endswith(("ches", "shes", "sses", "xes", "zes"))
    ):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _node_payload(node: Node) -> dict[str, object]:
    return {
        "id": node.id,
        "label": node.label,
        "type": node.type,
        "suggested_type": node.suggested_type,
        "aliases": node.aliases,
        "properties": node.properties,
    }
