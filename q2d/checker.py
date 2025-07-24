import re
from ast import literal_eval
from collections import Counter, defaultdict
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, get_args

import networkx as nx
import pandas as pd
from pydantic import BaseModel
from q2d.common.types import Graph
from q2d.graph_to_plantuml import GraphConvertConfig, convert_graph

SEVERETY_LEVEL = Literal["Low", "Medium", "High"]


class ErrorMessage(BaseModel):
    description: str
    severity: SEVERETY_LEVEL
    number: int = 1


def check_format(
    generated_graph_dict: Dict[str, Any], code: str, return_warnings: bool = False
) -> Tuple[Dict[str, Any], List[ErrorMessage]]:
    error_messages: List[ErrorMessage] = []

    # modify fields
    nodes_ids_map = {node["node_id"]: node["node_id"].replace(" ", "") for node in generated_graph_dict["nodes"]}
    packages_ids_map = {
        package["package_id"]: package["package_id"].replace(" ", "")
        for package in generated_graph_dict["packages"]
        if package["package_id"] not in nodes_ids_map
    }
    ids_map = packages_ids_map | nodes_ids_map

    nodes_ids_to_clean = [before for before, after in nodes_ids_map.items() if before != after]
    packages_ids_to_clean = [before for before, after in packages_ids_map.items() if before != after]
    # nodes ids with spaces
    if len(nodes_ids_to_clean) > 0:
        error_messages.append(
            ErrorMessage(
                description=f"Ids of nodes should be cleaned from spaces: {nodes_ids_to_clean}",
                severity="Low",
                number=len(nodes_ids_to_clean),
            )
        )
    # packages ids with spaces
    if len(packages_ids_to_clean) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Ids of packages should be cleaned from spaces: {packages_ids_to_clean}"),
                severity="Low",
                number=len(packages_ids_to_clean),
            )
        )

    # collision in nodes ids after cleaning
    if len(nodes_ids_map) != len(generated_graph_dict["nodes"]):
        error_messages.append(
            ErrorMessage(
                description=(
                    f'Cleaned ids of nodes are not unique: {len(nodes_ids_map)}!={len(generated_graph_dict["nodes"])}'
                ),
                severity="Medium",
                number=abs(len(nodes_ids_map) - len(generated_graph_dict["nodes"])),
            )
        )
    # collision in packages ids after cleaning
    if len(packages_ids_map) != len(generated_graph_dict["packages"]):
        error_messages.append(
            ErrorMessage(
                description=(
                    f'Cleaned ids of packages are not unique: {len(packages_ids_map)}!={len(generated_graph_dict["packages"])}'
                ),
                severity="Medium",
                number=abs(len(packages_ids_map) - len(generated_graph_dict["packages"])),
            )
        )

    init_edges_num = len(generated_graph_dict["edges"])

    # update fields in the graph
    generated_graph_dict["nodes"] = [
        node | {"node_id": nodes_ids_map[node["node_id"]]} for node in generated_graph_dict["nodes"]
    ]
    generated_graph_dict["edges"] = [
        edge | {"node_id_from": nodes_ids_map[edge["node_id_from"]], "node_id_to": nodes_ids_map[edge["node_id_to"]]}
        for edge in generated_graph_dict["edges"]
        if edge["node_id_from"] in nodes_ids_map and edge["node_id_to"] in nodes_ids_map
    ]
    generated_graph_dict["packages"] = [
        package
        | {
            "package_id": packages_ids_map[package["package_id"]],
            "children": [ids_map[node_id] for node_id in package["children"] if node_id in ids_map],
        }
        for package in generated_graph_dict["packages"]
        if package["package_id"] in packages_ids_map and packages_ids_map[package["package_id"]] not in nodes_ids_map
    ]

    # utility structures
    graph: Dict[str, Dict[str, Optional[str]]] = defaultdict(dict)
    for e in generated_graph_dict["edges"]:
        graph[e["node_id_from"]][e["node_id_to"]] = e["description"] if "description" in e else ""
    package_graph = {package["package_id"]: package["children"] for package in generated_graph_dict["packages"]}

    # not valid edges
    if init_edges_num > len(generated_graph_dict["edges"]):
        error_messages.append(
            ErrorMessage(
                description=(f'Some edges are not valid: {init_edges_num}>{len(generated_graph_dict["edges"])}'),
                severity="Medium",
                number=abs(init_edges_num - len(generated_graph_dict["edges"])),
            )
        )

    # single node
    if len(generated_graph_dict["nodes"]) == 1:
        error_messages.append(
            ErrorMessage(description=(f'Single node in the graph: {generated_graph_dict["nodes"]}'), severity="Low")
        )

    # [warning] less than 3 nodes
    if return_warnings and len(generated_graph_dict["nodes"]) <= 3:
        error_messages.append(
            ErrorMessage(
                description=(f'Less than 3 nodes in the graph: {generated_graph_dict["nodes"]}'), severity="Low"
            )
        )

    # no edges
    if len(generated_graph_dict["edges"]) == 0:
        error_messages.append(
            ErrorMessage(description=(f'No edges in the graph: {generated_graph_dict["edges"]}'), severity="Low")
        )

    edges_to_itself = [node_from for node_from, val in graph.items() if node_from in val]
    if len(edges_to_itself) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Nodes have edge to itself: {edges_to_itself}"),
                severity="Low",
                number=len(edges_to_itself),
            )
        )

    # not unique edges
    unique_edges = set(f'{edge["node_id_from"]}->{edge["node_id_to"]}' for edge in generated_graph_dict["edges"])
    if len(unique_edges) != len(generated_graph_dict["edges"]):
        error_messages.append(
            ErrorMessage(
                description=(
                    f'Cleaned edges are not unique: {len(unique_edges)}!={len(generated_graph_dict["edges"])}'
                ),
                severity="Low",
                number=abs(len(unique_edges) - len(generated_graph_dict["edges"])),
            )
        )

    # more than 1 edge between two nodes
    num_edges = Counter(e["node_id_from"] for e in generated_graph_dict["edges"])
    more_than_one_edge = [
        (start_id, children_num) for start_id, children_num in num_edges.items() if children_num > len(graph[start_id])
    ]
    if len(more_than_one_edge) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"More than one edge between two nodes for (nodes, num_edges): {more_than_one_edge}"),
                severity="Low",
                number=len(more_than_one_edge),
            )
        )

    # Check if method/field have...
    edge_to_parent_class = []
    no_edge_from_parent_class = []
    empty_source_class_id = []
    for node in generated_graph_dict["nodes"]:
        if node["type"] in ("method", "field"):
            # ...empty source class id
            if node["source_class_id"] is None:
                empty_source_class_id.append(node["node_id"])
            else:
                # ...edge TO parent class
                if node["source_class_id"] in graph[node["node_id"]]:
                    edge_to_parent_class.append(node["node_id"])
                # ...no edges FROM parent class
                if node["node_id"] not in graph[node["source_class_id"]]:
                    no_edge_from_parent_class.append(node["node_id"])

    # no source class id
    if len(empty_source_class_id) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Methods/fields have empty source class id: {empty_source_class_id}"),
                severity="Medium",
                number=len(empty_source_class_id),
            )
        )
    # edge to class-parent
    if len(edge_to_parent_class) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Methods/fields have edge to class-parent: {edge_to_parent_class}"),
                severity="Low",
                number=len(edge_to_parent_class),
            )
        )
    # no edge from class-parent
    if len(no_edge_from_parent_class) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Methods/fields have no edge from class-parent: {no_edge_from_parent_class}"),
                severity="Low",
                number=len(no_edge_from_parent_class),
            )
        )

    # not only r"[a-zA-Z0-9_]+" symbols are in names
    not_valid_names = [
        node["name"] for node in generated_graph_dict["nodes"] if re.match(r"[^a-zA-Z0-9_]+", node["name"])
    ]
    if len(not_valid_names) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Nodes have not valid names: {not_valid_names}"),
                severity="Low",
                number=len(not_valid_names),
            )
        )

    # not only r"[a-zA-Z0-9_]+" symbols are in ids
    not_valid_node_ids = [
        node["node_id"] for node in generated_graph_dict["nodes"] if re.match(r"[^a-zA-Z0-9_]+", node["node_id"])
    ]
    if len(not_valid_node_ids) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Nodes have not valid node_ids: {not_valid_node_ids}"),
                severity="Low",
                number=len(not_valid_node_ids),
            )
        )

    # not only r"[^a-zA-Z0-9_\(\):\\\/ ]+" symbols are in package_ids
    not_valid_package_ids = [
        package["package_id"]
        for package in generated_graph_dict["packages"]
        if re.match(r"[^a-zA-Z0-9_\(\):\\\/ ]+", package["package_id"])
    ]
    if len(not_valid_package_ids) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Packages have not valid package_ids: {not_valid_package_ids}"),
                severity="Low",
                number=len(not_valid_package_ids),
            )
        )

    # ids not equal to names (warning)
    ids_not_equal_to_names = [
        (node["node_id"], node["name"]) for node in generated_graph_dict["nodes"] if node["node_id"] != node["name"]
    ]
    if return_warnings and len(ids_not_equal_to_names) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Ids not equal to names: {ids_not_equal_to_names}"),
                severity="Low",
                number=len(ids_not_equal_to_names),
            )
        )

    # names of nodes (excluding entities) not in code
    names_not_in_code = [
        node["name"] for node in generated_graph_dict["nodes"] if node["type"] != "entity" and node["name"] not in code
    ]
    if len(names_not_in_code) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Names of nodes not in code: {names_not_in_code}"),
                severity="Low",
                number=len(names_not_in_code),
            )
        )

    # packages with one node
    packages_with_one_node = [
        (package["package_id"], package["children"])
        for package in generated_graph_dict["packages"]
        if len(package["children"]) == 1
    ]
    if len(packages_with_one_node) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Packages with only one node: {packages_with_one_node}"),
                severity="Low",
                number=len(packages_with_one_node),
            )
        )

    # packages without nodes
    packages_without_nodes = [
        (package["package_id"], package["children"])
        for package in generated_graph_dict["packages"]
        if len(package["children"]) == 0
    ]
    if len(packages_without_nodes) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Packages without nodes: {packages_without_nodes}"),
                severity="Medium",
                number=len(packages_without_nodes),
            )
        )

    child_to_package = defaultdict(list)
    for package in generated_graph_dict["packages"]:
        for child in package["children"]:
            child_to_package[child].append(package["package_id"])

    # child in multiple packages
    child_in_multiple_packages = [child for child, packages in child_to_package.items() if len(packages) > 1]
    if len(child_in_multiple_packages) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Child in multiple packages: {child_in_multiple_packages}"),
                severity="Low",
                number=len(child_in_multiple_packages),
            )
        )

    # more packages than nodes (warning)
    if return_warnings and len(generated_graph_dict["packages"]) > len(generated_graph_dict["nodes"]):
        error_messages.append(
            ErrorMessage(
                description=(
                    f'More packages than nodes: {len(generated_graph_dict["packages"])}>{len(generated_graph_dict["nodes"])}'
                ),
                severity="Low",
                number=abs(len(generated_graph_dict["packages"]) - len(generated_graph_dict["nodes"])),
            )
        )

    # class in package, but method/field not in the same package
    # FIXME: cover case with multiple nested packages
    method_or_field_not_in_source_class_package = [
        node["node_id"]
        for node in generated_graph_dict["nodes"]
        if (
            node["type"] in ("method", "field")
            and len(child_to_package[node["source_class_id"]]) == 1
            and (
                len(child_to_package[node["node_id"]]) == 0
                or (
                    len(child_to_package[node["node_id"]]) == 1
                    and child_to_package[node["source_class_id"]][0] != child_to_package[node["node_id"]][0]
                    and (
                        child_to_package[node["node_id"]][0]
                        not in package_graph[child_to_package[node["source_class_id"]][0]]
                    )
                )
            )
        )
    ]
    if len(method_or_field_not_in_source_class_package) > 0:
        error_messages.append(
            ErrorMessage(
                description=(
                    f"Class in package, but method/field not in the same package: {method_or_field_not_in_source_class_package}"
                ),
                severity="Low",
                number=len(method_or_field_not_in_source_class_package),
            )
        )

    # Check packages recursion
    visited = set()

    def is_recursive_dfs(package_id: str, visited: Set[str], visited_current: Set[str]) -> bool:
        if package_id in visited_current:
            return True
        visited.add(package_id)
        visited_current.add(package_id)

        for component_id in package_graph[package_id]:
            if component_id in package_graph and (
                component_id in visited_current or is_recursive_dfs(component_id, visited, visited_current)
            ):
                return True

        return False

    recursive_packages = []
    for package_id in package_graph:
        if package_id not in visited:
            if is_recursive_dfs(package_id, visited=visited, visited_current=set()):
                recursive_packages.append(package_id)

    if len(recursive_packages) > 0:
        error_messages.append(
            ErrorMessage(
                description=(f"Packages recursion: {recursive_packages}"),
                severity="Medium",
                number=len(recursive_packages),
            )
        )

    # multiple connected components
    nx_graph = nx.Graph(
        [(edge_from, edge_to) for edge_from, val in graph.items() for edge_to, desc in val.items()]
        + [(edge_to, edge_from) for edge_from, val in graph.items() for edge_to, desc in val.items()]
    )
    nx_graph.add_nodes_from([node["node_id"] for node in generated_graph_dict["nodes"]])
    connected_components_num = nx.number_connected_components(nx_graph)

    if connected_components_num > 1:
        error_messages.append(
            ErrorMessage(description=(f"Multiple connected components: {connected_components_num}"), severity="Medium")
        )

    return generated_graph_dict, error_messages


def convert_str_graph_with_check(
    generated_graph_str: Optional[str], code: str, graph_convert_config: GraphConvertConfig
) -> Tuple[Optional[str], List[ErrorMessage], Optional[Graph]]:
    if generated_graph_str is None:
        return None, [], None

    generated_graph_str_clean = re.sub(r"(?<![a-zA-Z0-9_])null(?![a-zA-Z0-9_])", "None", generated_graph_str.strip())
    generated_graph_dict = literal_eval(generated_graph_str_clean)
    generated_graph_dict, errors = check_format(generated_graph_dict, code)

    converted_graph = None
    generated_graph = None
    try:
        generated_graph = Graph(**generated_graph_dict)
        converted_graph = convert_graph(generated_graph, graph_convert_config)
    except (ValueError, TypeError, AttributeError, SyntaxError) as e:
        print(str(e))
    return converted_graph, errors, generated_graph


def analyze_errors_by_severity(
    points: List[Dict[str, Any]],
    mode: Literal["total", "unique"] = "total",
    average: Literal[None, "micro", "macro"] = None,
) -> Dict[SEVERETY_LEVEL, float]:
    error_counters = []
    total_nodes = 0
    for point in points:
        converted_diagram, errors, generated_graph = convert_str_graph_with_check(
            point["diagram"], point["code"], GraphConvertConfig()
        )
        errors_freq = dict(Counter({er.severity: er.number if mode == "total" else 1 for er in errors}))

        # Add High severity error if diagram couldn't be converted
        errors_freq["High"] = int(converted_diagram is None)

        if average == "macro":
            assert generated_graph is not None, "Graph should be convertable for `macro` average"
            for key, _ in errors_freq.items():
                errors_freq[key] /= len(generated_graph["nodes"])
        elif average == "micro":
            assert generated_graph is not None, "Graph should be convertable for `micro` average"
            total_nodes += len(generated_graph["nodes"])
        error_counters.append(errors_freq)

    df = pd.DataFrame(error_counters, columns=get_args(SEVERETY_LEVEL)).fillna(0)
    df_aggregated = df.mean(axis=0)

    if average == "micro":
        df_aggregated /= total_nodes / len(error_counters)
    return df_aggregated.to_dict()


def analyze_multiple_connected_components(points: List[Dict[str, Any]]) -> float:
    error_counters = 0
    for point in points:
        _, errors, _ = convert_str_graph_with_check(point["diagram"], point["code"], GraphConvertConfig())
        error_counters += int(any("Multiple connected components" in e.description for e in errors))

    return error_counters / len(points)
