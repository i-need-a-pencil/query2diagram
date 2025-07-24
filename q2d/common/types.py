from typing import Any, List, Literal, Optional

from pydantic import BaseModel

VISIBILITY_TYPES = Literal["private", "protected", "package private", "package_private", "internal", "public"]
NODE_TYPES = Literal["class", "variable", "function", "entity", "method", "field"]
QUERY_VERSIONS = Literal["minimal", "medium", "full"]


class SubscriptableBaseModel(BaseModel):
    def __getitem__(self, key: Any) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: Any, value: Any) -> None:
        return setattr(self, key, value)

    def get(self, key: Any, default: Any) -> Any:
        return getattr(self, key) if hasattr(self, key) else default

    def __str__(self):
        return str(self.model_dump(mode="json"))


class Node(SubscriptableBaseModel):
    type: NODE_TYPES
    name: str
    node_id: str
    description: Optional[str]
    visibility: VISIBILITY_TYPES
    return_type: Optional[str] = None
    params: Optional[str] = None
    source_class_id: Optional[str] = None


class Edge(SubscriptableBaseModel):
    node_id_from: str
    node_id_to: str
    description: Optional[str] = None


class Package(SubscriptableBaseModel):
    package_id: str
    children: List[str]
    description: Optional[str] = None


class Graph(SubscriptableBaseModel):
    nodes: List[Node]
    edges: List[Edge]
    packages: List[Package]


class GeneratedOutput(SubscriptableBaseModel):
    raw_output: Optional[str] = None
    extracted_output: Optional[str] = None
    total_tokens: Optional[int] = None

    def is_empty(self) -> bool:
        return self.raw_output is None or len(self.raw_output) == 0

    def get_output(self) -> Optional[str]:
        return self.extracted_output if self.extracted_output is not None else self.raw_output
