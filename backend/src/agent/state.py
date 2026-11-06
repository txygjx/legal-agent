from __future__ import annotations
from typing import List
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
import operator

class SubQuery(TypedDict):
    id: str
    query: str
    route: str  # "rag", "web", "auto"

class OverallState(TypedDict):
    messages: Annotated[list, add_messages]

    # 【完美并发合并】使用 operator.add 保证并行搜索的结果绝不丢失、绝不冲突
    sources_gathered: Annotated[list, operator.add]
    research_results: Annotated[list, operator.add]

    # 【无缝覆盖】不需要 reducer，下一轮反思的新任务直接覆盖旧任务
    sub_queries: List[SubQuery]

    research_loop_count: int
    max_research_loops: int
    is_sufficient: bool

    # 其他兼容参数
    reasoning_model: str
    initial_search_query_count: int
    route: str

    final_context: str
    confirmed_findings: str   # 反思节点逐轮蒸馏出的"已确认信息摘要"，每轮整体替换
    processed_count: int      # 记录 research_results 里已经被折叠进摘要的位置