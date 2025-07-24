from typing import Optional

from q2d.common.types import SubscriptableBaseModel


class BasePrompt(SubscriptableBaseModel):
    system_prompt_template: Optional[str] = None
    user_prompt_template: str
    continue_prompt_template: Optional[str] = None


DiagramsFinetunedPrompt = BasePrompt(
    system_prompt_template=r"""Your task is to generate json with "nodes", "edges" and "packages" for a diagram. The diagram must answer to the user query using the code.""",
    user_prompt_template=(
        r"""<code>
{code} </code>
<query>
{query} [{version} version] </query>"""
    ),
)
DiagramsZeroShotPrompt = BasePrompt(
    system_prompt_template=(
        r"""1. You should generate Python list of nodes, list of edges and list of packages. Each element is presented by JSON.
2. Make sure the final diagram is comprehensible: it should be readable, understandable by user.
3. For each node write a short description.
4. The diagram should contain all important nodes and edges to code understanding.
5. You can omit some of the standard, boilerplate-code steps.
6. You can add conceptual entities to diagram (high-level concepts that are not functions or classes).
7. Try to group related nodes (including those you introduce) using package elements, where possible, do not use classes for this purpose.
8. Package elements can be nested.
9. Do not add fake classes to aggregate code entities, use packages for this purpose. All fields and methods should be presented as is in code.
10. JSON template for node: {
    "type": Literal["class", "variable", "function", "entity", "method", "field"], # a type of node from the predefined types; use the most appropriate
    "name": str, # the actual name of the node to show on the diagram
    "node_id": str, # id or unique name of the node; use the actual name of the node if possible
    "description": str, # a short description of node
    "visibility": Literal["private", "protected", "package private", "public"], # an access modifier of the node from the predefined types; use the most appropriate
    "return_type": Optional[str], # a return type for functions and methods or current node type for fields and variables; can be skipped for other types of nodes
    "params": Optional[str], # parameters of functions and methods (as in brackets); can be skipped for other types of nodes
    "source_class_id": Optional[str], # node id of the source class for fields and methods; can be skipped for other types of nodes
}
11. JSON template for edge: {
    "node_id_from": str, # id of the node where the edge starts
    "node_id_to": str, # id of the node where the edge ends
    "description": Optional[str], # a short description of the edge; can be None
}
12. JSON template for packages: {
    "package_id": str, # id or unique name of the package; it will be shown on the diagram
    "children": List[str], # list of ids of nodes to include in the package or names of nested packages
    "description": Optional[str], # a short description of the package; can be None
}
13. JSON template for graph: {
    "nodes": [{node1_JSON}, ..., {nodeN_JSON}],
    "edges": [{edge1_JSON}, ..., {edgeN_JSON}],
    "packages": [{package1_JSON}, ..., {packageN_JSON}]
}
14. Prefer to use camelCase, snake_case, PascalCase for names, do not use spaces in names.
15. Package names can not be in edges list, use them only in packages.
16. Do not write any explanations, just write expected output.
17. You should generate three versions of the `Graph template`. The first one, called the "minimal version," should include only the essential nodes and logic, removing all unnecessary elements. The second one, the "medium version," should contain all important nodes along with some additional details. The third one, the "full version," should incorporate every possible node and all possible edges relevant to query.
18. After generating all three graph templates, you should provide the shortest possible "text answer" to the user's query using the generated diagrams.
19. Expected output template: {
    "minimal_version": {graph_JSON},
    "medium_version": {graph_JSON},
    "full_version": {graph_JSON},
    text_answer: str,
}"""
    ),
    user_prompt_template=(
        r"""Your task is write nodes, edges and packages to build a diagram in different formats (PlantUML, mermaid-js, ...) for the following file in project:
```
{code}
```
You need to generate a diagram for the following user query: \"{query}\""""
    ),
)

DiagramsUserOnlyZeroShotPrompt = BasePrompt(
    system_prompt_template=None,
    user_prompt_template=(
        r"""1. You should generate Python list of nodes, list of edges and list of packages. Each element is presented by JSON.
2. Make sure the final diagram is comprehensible: it should be readable, understandable by user.
3. For each node write a short description.
4. The diagram should contain all important nodes and edges to code understanding.
5. You can omit some of the standard, boilerplate-code steps.
6. You can add conceptual entities to diagram (high-level concepts that are not functions or classes).
7. Try to group related nodes (including those you introduce) using package elements, where possible, do not use classes for this purpose.
8. Package elements can be nested.
9. Do not add fake classes to aggregate code entities, use packages for this purpose. All fields and methods should be presented as is in code.
10. JSON template for node: {
    "type": Literal["class", "variable", "function", "entity", "method", "field"], # a type of node from the predefined types; use the most appropriate
    "name": str, # the actual name of the node to show on the diagram
    "node_id": str, # id or unique name of the node; use the actual name of the node if possible
    "description": str, # a short description of node
    "visibility": Literal["private", "protected", "package private", "public"], # an access modifier of the node from the predefined types; use the most appropriate
    "return_type": Optional[str], # a return type for functions and methods or current node type for fields and variables; can be skipped for other types of nodes
    "params": Optional[str], # parameters of functions and methods (as in brackets); can be skipped for other types of nodes
    "source_class_id": Optional[str], # node id of the source class for fields and methods; can be skipped for other types of nodes
}
11. JSON template for edge: {
    "node_id_from": str, # id of the node where the edge starts
    "node_id_to": str, # id of the node where the edge ends
    "description": Optional[str], # a short description of the edge; can be None
}
12. JSON template for packages: {
    "package_id": str, # id or unique name of the package; it will be shown on the diagram
    "children": List[str], # list of ids of nodes to include in the package or names of nested packages
    "description": Optional[str], # a short description of the package; can be None
}
13. JSON template for graph: {
    "nodes": [{node1_JSON}, ..., {nodeN_JSON}],
    "edges": [{edge1_JSON}, ..., {edgeN_JSON}],
    "packages": [{package1_JSON}, ..., {packageN_JSON}]
}
14. Prefer to use camelCase, snake_case, PascalCase for names, do not use spaces in names.
15. Package names can not be in edges list, use them only in packages.
16. Do not write any explanations, just write expected output.
17. You should generate three versions of the `Graph template`. The first one, called the "minimal version," should include only the essential nodes and logic, removing all unnecessary elements. The second one, the "medium version," should contain all important nodes along with some additional details. The third one, the "full version," should incorporate every possible node and all possible edges relevant to query.
18. After generating all three graph templates, you should provide the shortest possible "text answer" to the user's query using the generated diagrams.
19. Expected output template: {
    "minimal_version": {graph_JSON},
    "medium_version": {graph_JSON},
    "full_version": {graph_JSON},
    text_answer: str,
}

Your task is write nodes, edges and packages to build a diagram in different formats (PlantUML, mermaid-js, ...) for the following file in project:
```
{code}
```
You need to generate a diagram for the following user query: \"{query}\""""
    ),
)


QuestionsPrompt = BasePrompt(
    system_prompt_template=(
        r"""1. The final output must be a JSON array of strings representing questions, enclosed within `<candidates>` and `<final_output>` XML tags. You must strictly follow this template for the final answer: `<candidates>["question1", ..., "questionN"]</candidates><final_output>["question1", ..., "questionK"]</final_output>`. During processing, you may represent questions in any format.  
2. Questions must focus on aspects of the provided code, such as project structure, component interactions, imported modules, code behavior, design patterns, code design principles, external API usage, deployment strategies, or the potential integration of new components into the code.  
3. Each question will be used as a user query for another LLM. Therefore, each generated question should either ask the LLM for information or instruct it to perform a task.
4. The expected answer for each question must be some kind of diagram (e.g., PlantUML or Mermaid.js). Ensure each question can be effectively answered with a diagrammatic representation rather than just text.  
5. Do not mention diagrams explicitly; instead, use similar words such as structure, relationships, interactions and so on.  
6. You must generate 10 candidate questions within the `<candidates>` tags.
7. Filter or combine candidate questions to create a final selection of 0 to 3 high-quality questions within the `<final_output>` tags. Prioritize question quality over quantity. Try to cover different topics in the final selection.  
8. The LLM answering the questions will only have access to the content of the provided code file. Therefore, ensure all generated questions can be answered solely based on the information within that file, without requiring external context or knowledge beyond the given code.  
9. Each question must be straightforward and as short as possible. Since each question is intended to be answered by a single diagram, avoid combining multiple questions into one.  
10. Each question must be unique. Avoid creating duplicate or near-duplicate questions, even with slight variations in wording.
11. If you cannot identify any questions that meet all the specified criteria, it is perfectly acceptable to return an empty list within the `<final_output>` tags. If a question does not fully meet all requirements (or if you're not 100% sure), you should remove it."""
    ),
    user_prompt_template=(
        r"""Generate a list of questions based on the following code file:
```
{code}
```"""
    ),
    continue_prompt_template=(
        r"""Return the final answer with the following template: `<candidates>["question1", ..., "questionN"]</candidates><final_output>["question1", ..., "questionN"]</final_output>`"""
    ),
)

QuestionsUserOnlyPrompt = BasePrompt(
    system_prompt_template=None,
    user_prompt_template=(
        r"""1. The final output must be a JSON array of strings representing questions, enclosed within `<candidates>` and `<final_output>` XML tags. You must strictly follow this template for the final answer: `<candidates>["question1", ..., "questionN"]</candidates><final_output>["question1", ..., "questionK"]</final_output>`. During processing, you may represent questions in any format.  
2. Questions must focus on aspects of the provided code, such as project structure, component interactions, imported modules, code behavior, design patterns, code design principles, external API usage, deployment strategies, or the potential integration of new components into the code.  
3. Each question will be used as a user query for another LLM. Therefore, each generated question should either ask the LLM for information or instruct it to perform a task.
4. The expected answer for each question must be some kind of diagram (e.g., PlantUML or Mermaid.js). Ensure each question can be effectively answered with a diagrammatic representation rather than just text.  
5. Do not mention diagrams explicitly; instead, use similar words such as structure, relationships, interactions and so on.  
6. You must generate 10 candidate questions within the `<candidates>` tags.
7. Filter or combine candidate questions to create a final selection of 0 to 3 high-quality questions within the `<final_output>` tags. Prioritize question quality over quantity. Try to cover different topics in the final selection.  
8. The LLM answering the questions will only have access to the content of the provided code file. Therefore, ensure all generated questions can be answered solely based on the information within that file, without requiring external context or knowledge beyond the given code.  
9. Each question must be straightforward and as short as possible. Since each question is intended to be answered by a single diagram, avoid combining multiple questions into one.  
10. Each question must be unique. Avoid creating duplicate or near-duplicate questions, even with slight variations in wording.
11. If you cannot identify any questions that meet all the specified criteria, it is perfectly acceptable to return an empty list within the `<final_output>` tags. If a question does not fully meet all requirements (or if you're not 100% sure), you should remove it.

Generate a list of questions based on the following code file:
```
{code}
```"""
    ),
    continue_prompt_template=(
        r"""Return the final answer with the following template: `<candidates>["question1", ..., "questionN"]</candidates><final_output>["question1", ..., "questionN"]</final_output>`"""
    ),
)
