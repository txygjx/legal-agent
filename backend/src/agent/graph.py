import os
import uuid
import json
import re
import requests

from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.types import Send
from langgraph.graph import StateGraph, START, END

# 请确保这些包路径与你的项目匹配
from agent.state import OverallState
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    web_searcher_instructions,
    answer_instructions,
    reflection_evaluation_instructions,
    decomposition_instructions,
    tool_agent_instructions,
)
from agent.tools_and_schemas import DecompositionPlan, NewReflection
from agent.utils import get_research_topic

load_dotenv()


# ============================================================
# 消融测试开关说明（通过环境变量控制）
#
# TEST_NO_RAG=true        → 关闭RAG，所有子任务强制走Web搜索
#                           对照：意图拆解+路由 vs 意图拆解+纯Web
#
# TEST_NO_REFLECTION=true → 关闭增量反思摘要，每轮把全部结果重新拼接
#                           对照：增量confirmed_findings vs 全量累积
#
# TEST_NO_CTX=true        → 关闭ContextManager，直接拼接所有结果
#                           对照：有无去重/排序/Token控制
#
# 使用方式：
#   全开（你的完整系统）：正常启动，不设置任何开关
#   基线（三个全关）：TEST_NO_RAG=true TEST_NO_REFLECTION=true TEST_NO_CTX=true
#   只测RAG：TEST_NO_REFLECTION=true TEST_NO_CTX=true（其他两个关掉，只保留RAG）
# ============================================================

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

_llm = ChatOpenAI(
    model="qwen-plus",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


import re
import hashlib
from dataclasses import dataclass
from typing import List

@dataclass
class ContextItem:
    """上下文候选项"""
    content: str
    source: str = ""
    source_id: str = ""
    round: int = 0
    score: float = 0.0


class ContextManager:
    """上下文管理器：去重 → 相关性排序 → Token预算 → Top-K"""

    def __init__(self, max_tokens: int = 6000, top_k: int = 8):
        self.max_tokens = max_tokens
        self.top_k = top_k

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """简化Token估算（1字符≈1token）"""
        return len(text) if text else 0

    def deduplicate(self, items: List[ContextItem]) -> List[ContextItem]:
        """基于内容MD5去重"""
        seen = set()
        unique = []
        for item in items:
            content = item.content.strip()
            if not content:
                continue
            normalized = re.sub(r'\s+', ' ', content).strip()
            content_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(item)
        return unique

    def calculate_relevance(self, query: str, content: str) -> float:
        """关键词匹配评分（可替换为Cross-Encoder）"""
        if not query or not content:
            return 0.0
        terms = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+', query.lower())
        if not terms:
            return 0.0
        content_lower = content.lower()
        matched = sum(1 for term in terms if term in content_lower)
        return matched / len(terms)

    def rerank(self, query: str, items: List[ContextItem]) -> List[ContextItem]:
        """重新计算相关性分数并降序排列"""
        for item in items:
            item.score = self.calculate_relevance(query, item.content)
        return sorted(items, key=lambda x: x.score, reverse=True)

    def fit_token_budget(self, items: List[ContextItem], reserved_tokens: int = 0) -> List[ContextItem]:
        """按Token预算截断（保留97%上限）"""
        available_budget = max(0, self.max_tokens - reserved_tokens)
        budget = int(available_budget * 0.97)
        selected = []
        used = 0
        for item in items:
            tokens = self.estimate_tokens(item.content)
            if tokens > budget:          # 单个条目超预算，跳过
                continue
            if used + tokens > budget:
                break
            selected.append(item)
            used += tokens
        return selected

    def select_top_k(self, items: List[ContextItem]) -> List[ContextItem]:
        """取Top-K个条目"""
        return items[:self.top_k]

    def build_context(self, query: str, items: List[ContextItem],reserved_tokens: int = 0) -> List[ContextItem]:
        """完整流水线：去重 → 重排序 → Token预算 → Top-K"""
        print("\n========== Context Manager ==========")
        print(f"原始 Context 数量: {len(items)}")

        items = self.deduplicate(items)
        print(f"去重后数量: {len(items)}")

        items = self.rerank(query, items)
        print("相关性排序完成")

        items = self.fit_token_budget(items, reserved_tokens=reserved_tokens)
        print(f"Token Budget 后: {len(items)}")

        items = self.select_top_k(items)
        print(f"最终 Top-K: {len(items)}")

        total_tokens = sum(self.estimate_tokens(item.content) for item in items)
        print(f"最终 Context Token: {total_tokens}")
        print("====================================\n")
        return items

    def format_context(self, items: List[ContextItem]) -> str:
        """格式化为LLM输入文本"""
        parts = []
        for i, item in enumerate(items, 1):
            parts.append(
                f"[Context-{i}]\n"
                f"来源: {item.source}\n"
                f"相关性: {item.score:.4f}\n"
                f"内容:\n{item.content}"
            )
        return "\n\n---\n\n".join(parts)

    def fit(self, text: str) -> str:
        if not text:
            return ""
        if self.estimate_tokens(text) <= self.max_tokens:
            return text

        print(f"⚠️ confirmed_findings 超出预算({self.estimate_tokens(text)} > {self.max_tokens})，触发兜底截断")
        return text[: self.max_tokens] + "\n\n[注：内容超出预算，后续部分已被截断]"


# 创建全局上下文管理器实例，你后面应该根据你的模型上下文窗口、answer_instructions 长度以及实际业务测试调整。
context_manager = ContextManager(max_tokens=6000, top_k=8)


CONFIRMED_FINDINGS_MAX_TOKENS = 1500
confirmed_findings_manager = ContextManager(max_tokens=CONFIRMED_FINDINGS_MAX_TOKENS)
# ==============================
# 工具辅助函数
# ==============================
def _format_sources_for_text(sources: list) -> str:
    return "".join([f"【SOURCE_START】{json.dumps(s, ensure_ascii=False)}【SOURCE_END】" for s in sources])

def _extract_sources_from_result(text: str) -> list:
    sources = []
    pattern = r"【SOURCE_START】(.*?)【SOURCE_END】"
    matches = re.findall(pattern, text)
    for m in matches:
        try:
            sources.append(json.loads(m))
        except Exception:
            pass
    return sources

def _remove_source_markers(text: str) -> str:
    return re.sub(r"【SOURCE_START】.*?【SOURCE_END】", "", text)


# ==============================
# 注册工具 (增加异常防崩溃保护)
# ==============================
@tool
def web_search_tool(query: str, config: RunnableConfig) -> str:
    """互联网实时搜索工具"""
    try:
        search_result = tavily.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            timeout=120
        )
    except Exception as e:
        return f"【WEB检索失败】 错误信息: {str(e)}"

    docs = search_result.get("results", [])
    context = ""
    sources = []

    for i, doc in enumerate(docs):
        short = f"[WEB-{i+1}]"
        sources.append({"short_url": short, "source": doc.get("url", "")})
        context += f"文档{i+1}\n来源编号:{short}\n标题:{doc.get('title','')}\n正文:\n{doc.get('content','')}\n\n"

    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=query,
        context=context
    )

    response = _llm.invoke(formatted_prompt)
    answer = response.content

    used_sources = [s for s in sources if s["short_url"] in answer]
    answer += _format_sources_for_text(used_sources)
    return f"【WEB检索结果】\n{answer}"


@tool
def legal_rag_tool(query: str, config: RunnableConfig) -> str:
    """法律知识库检索工具"""
    try:
        resp = requests.post(
            "http://127.0.0.1:7091/api/search",
            json={"question": query, "chunks": 2, "active_docs": "6a1a6ba7c4305daa55b4f195"},
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return f"【法律知识库查询失败】 错误信息: {str(e)}"

    if not isinstance(data, list):
        data = data.get("results") or data.get("data") or []

    if not data:
        return f"针对问题【{query}】，未检索到相关内容"

    context_parts = []
    sources = []
    for i, item in enumerate(data):
        short = f"[RAG-{i+1}]"
        sources.append({"short_url": short, "source": item.get("source", "法律知识库")})
        context_parts.append(f"来源编号:{short}\n正文:{item.get('text','')}")

    result = "以下是法律知识库内容:\n" + "\n".join(context_parts)
    result += _format_sources_for_text(sources)
    return result

tools = [legal_rag_tool, web_search_tool]
tools_map = {t.name: t for t in tools}


# ==============================
# 节点 1: 意图拆解节点
# ==============================
def analyze_and_decompose(state: OverallState, config: RunnableConfig) -> dict:
    question = get_research_topic(state["messages"])
    structured_llm = _llm.with_structured_output(DecompositionPlan)
    prompt = decomposition_instructions.format(question=question)

    plan = structured_llm.invoke(prompt)

    sub_queries = []
    for q in plan.queries:
        sub_queries.append({
            "id": str(uuid.uuid4()),
            "query": q.query,
            "route": getattr(q, "route", "auto")
        })

    loop_count = state.get("research_loop_count", 0)
    print(f"========== 意图拆解完成 当前循环:{loop_count} ==========")

    return {
        "sub_queries": sub_queries,
        "research_loop_count": loop_count
    }



    # 获取用户原始问题
    # 将原始问题拼接到提示词
    # 让大模型按照预设的结构输出内容，子问题以及子问题对应RAG,WEB,AUTO哪个路由
    # 将大模型输出的内容存入子问题列表
    # 将子问题列表传入下一个节点

# ==============================
# 路由 1: 并发分发 Router
# ==============================
def dispatch_searches(state: OverallState):
    sends = []
    subs = state.get("sub_queries", [])

    # ★ 改动：原来这里直接跳 "finalize_answer"，会绕过 context_manager，
    # 导致 finalize_answer 读到的 final_context 是空的。统一跳 context_manager。
    # 兜底防护：如果没有拆解出问题，直接跳去最后回答
    if not subs:
        return "context_manager"

    current_round = state.get("research_loop_count", 0)

    for sub in subs:
        sends.append(
            Send(
                "execute_search_branch",
                {
                    "query": sub["query"],
                    "id": sub["id"],
                    "route": sub["route"],
                    "round": current_round
                }
            )
        )
    return sends


# ==============================
# 节点 2: 独立闭环并发执行 (彻底消除竞态条件)
# ==============================
def execute_search_branch(branch_state: dict, config: RunnableConfig) -> dict:
    query = branch_state["query"]
    route = branch_state["route"]
    sub_id = branch_state["id"]
    round_num = branch_state.get("round", 0)


    # ============================================================
    # 【消融开关 1】TEST_NO_RAG
    # 作用：强制所有子任务走 Web，关闭 RAG 路由能力
    # 对照的是：有RAG路由 vs 无RAG路由（纯Web）
    # ============================================================
    if os.getenv("TEST_NO_RAG") == "true":
        route = "web"
        print(f"[消融模式] TEST_NO_RAG=true，强制走Web: {query}")
    # ============================================================

    result_text = ""

    if route == "rag":
        result_text = legal_rag_tool.invoke({"query": query}, config=config)
    elif route == "web":
        result_text = web_search_tool.invoke({"query": query}, config=config)
    else:  # auto，让 LLM 选择工具
        llm_with_tools = _llm.bind_tools(tools)
        prompt = tool_agent_instructions.format(search_query=query)
        response = llm_with_tools.invoke([HumanMessage(content=prompt)], config=config)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for call in response.tool_calls:
                func = tools_map.get(call["name"])
                if func:
                    try:
                        result_text += func.invoke(call.get("args", {}), config=config) + "\n"
                    except Exception as e:
                        result_text += f"工具执行失败:{str(e)}\n"
        else:
            result_text = response.content

    # 提取来源
    parsed_sources = _extract_sources_from_result(result_text)
    unique_sources = []

    # 【关键防冲突逻辑】：防止不同并发线程都返回 [WEB-1]，导致替换错误
    short_id_prefix = sub_id[:4]
    for s in parsed_sources:
        original_short = s["short_url"]
        new_short = original_short.replace("[", f"[{short_id_prefix}-") # 例如从 [WEB-1] 变成 [a1b2-WEB-1]

        # 替换文本里的占位符和提取出的对象里的占位符
        result_text = result_text.replace(original_short, new_short)
        s["short_url"] = new_short
        unique_sources.append(s)

    cleaned_result = _remove_source_markers(result_text)
    final_output = f"[子任务:{query}|round:{round_num}]\n{cleaned_result}"

    # LangGraph 将安全地把并行返回的 list 进行相加 (因为我们在 state 里用了 operator.add)
    return {
        "research_results": [final_output],
        "sources_gathered": unique_sources
    }


# ==============================
# 节点 3: 反思节点
# ==============================
def reflection_test(state: OverallState, config: RunnableConfig) -> dict:
    #当前轮数累加1
    loop_count = state.get("research_loop_count", 0) + 1

    #先获取rag和网络检索的所有历史内容，
    result_all = state.get("research_results")

    #通过获取上一次检索内容总条数的长度，截取新增的检索部分
    processed_count = state.get("processed_count", 0)
    result_new = result_all[processed_count:]

    #获取上一次查询内容的提炼结果
    previous_findings = state.get("confirmed_findings", "")


    #将用户提问，上一次提炼的结果, 新增检索内容，拼接成 prompt 传给大模型
    formatted_prompt = reflection_evaluation_instructions.format(
        question=get_research_topic(state["messages"]),
        confirmed_findings=previous_findings or "（暂无）",
        new_results=result_new or "（本轮无新增检索结果）",
    )

    #获取结构化输出
    structured_llm = _llm.with_structured_output(NewReflection)
    result = structured_llm.invoke(formatted_prompt, config=config)

    #将大模型返回的内容更新到state
    update = {
        "is_sufficient": result.is_sufficient,
        "confirmed_findings": confirmed_findings_manager.fit(result.key_findings),
        "processed_count": len(result_all),
        "research_loop_count": loop_count,
    }


    #如果不充足，就要根据原始问题生成新的子查询，然后存入state
    if not result.is_sufficient:
        new_sub_queries = []
        for q in result.follow_up_queries:
            new_sub_queries.append({
                "id": str(uuid.uuid4()),
                "query":q.query,
                "route": getattr(q, "route", "auto")
            })
        update["sub_queries"] = new_sub_queries

    # 无论是否充足，都要返回update，然后让下一个路由节点根据update更新的字段，去判断进入最终回答还是按照新的子问题重新查询
    return update


# ==============================
# 路由 2: 判断是否进入最终答案
# ==============================
def reflection_router(state: OverallState, config: RunnableConfig):
    # 完美兼容你现有的 Configuration 类读取方式
    configurable = Configuration.from_runnable_config(config)
    # 获取最大循环次数，如果配置里没有，兜底写为3
    max_loops = state.get("max_research_loops") or getattr(configurable, "max_research_loops", 3)

    if state.get("is_sufficient") or state.get("research_loop_count", 0) >= max_loops:
        return "context_manager"
    # 如果信息不足，继续进入新一轮的分发
    return dispatch_searches(state)


# ==============================
# 节点 4: 最终答案生成
# ==============================
# def finalize_answer(state: OverallState, config: RunnableConfig) -> dict:
    # formatted_prompt = answer_instructions.format(
    #     current_date=get_current_date(),
    #     research_topic=get_research_topic(state["messages"]),
    #     summaries="\n---\n\n".join(state.get("research_results", []))
    # )

    # result = _llm.invoke(formatted_prompt, config=config)
    # answer = result.content
    # unique_sources = []

    # # 替换文中引用的短链为实际来源
    # for source in state.get("sources_gathered", []):
    #     short = source["short_url"]
    #     if short in answer:
    #         answer = answer.replace(short, source["source"])
    #         unique_sources.append(source)

    # return {
    #     "messages": [AIMessage(content=answer)],
    #     "sources_gathered": unique_sources
    # }

def finalize_answer(state: OverallState, config: RunnableConfig) -> dict:
    """生成最终回答，使用 Context Manager 处理后的 final_context 作为上下文。"""
    final_context = state.get("final_context", "")

    formatted_prompt = answer_instructions.format(
        current_date=get_current_date(),
        research_topic=get_research_topic(state["messages"]),
        summaries=final_context,
    )

    result = _llm.invoke(formatted_prompt, config=config)
    answer = result.content

    unique_sources = []
    for source in state.get("sources_gathered", []):
        short = source["short_url"]
        if short in answer:
            answer = answer.replace(short, source["source"])
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=answer)],
        "sources_gathered": unique_sources,
    }

def context_manager_node(state: OverallState, config: RunnableConfig) -> dict:
    """节点4：将多轮 research_results 转换为最终上下文 final_context。

    research_results 完整保留在 State 中，此处仅生成最终给 LLM 的上下文。

    ✅ 修复(P0-3): confirmed_findings 之前只在反思节点内部自用，
    finalize_answer 完全看不到——等于反思阶段蒸馏出的核心结论从没真正
    进入过最终答案。现在把它作为"优先内容"纳入 final_context，单独预留
    token 预算、不参与关键词打分淘汰；剩下的预算再交给按相关性排序后的
    原始材料（保留可引用的来源短链，供 finalize_answer 做引用替换）。
    """

     # ============================================================
    # 【消融开关 3】TEST_NO_CTX
    # 作用：跳过去重/相关性排序/Token预算/Top-K，直接把所有结果拼接
    # 对照的是：有ContextManager vs 无ContextManager（原始结果全量传入）
    # ============================================================
    if os.getenv("TEST_NO_CTX") == "true":
        print("========== Context Manager 跳过（基线模式）==========")
        raw = "\n\n---\n\n".join(state.get("research_results", []))
        confirmed_findings = state.get("confirmed_findings", "")
        parts = []
        if confirmed_findings:
            parts.append(f"[已确认核心结论]\n{confirmed_findings}")
        if raw:
            parts.append(f"[检索结果（未经管理）]\n{raw}")
        final_context = "\n\n---\n\n".join(parts)
        print(f"原始结果数量: {len(state.get('research_results', []))}")
        print(f"直接拼接，Token数量: {len(final_context)}")
        return {"final_context": final_context}
    # ============================================================

    question = get_research_topic(state["messages"])
    research_results = state.get("research_results", [])
    confirmed_findings = state.get("confirmed_findings", "")

    # 解析每条原始结果为 ContextItem，并还原它实际产生于哪一轮
    candidates = []
    round_pattern = re.compile(r"\[子任务:(.*?)\|round:(\d+)\]\n(.*)", flags=re.DOTALL)
    legacy_pattern = re.compile(r"\[子任务:(.*?)\]\n(.*)", flags=re.DOTALL)

    for idx, result in enumerate(research_results):
        match = round_pattern.match(result)
        if match:
            content = match.group(3)
            round_num = int(match.group(2))
        else:
            # ✅ 修复(P2)的兼容分支：老格式（没有 round 标记）的结果依然能正常解析
            legacy_match = legacy_pattern.match(result)
            content = legacy_match.group(2) if legacy_match else result
            round_num = 0

        candidates.append(
            ContextItem(
                content=content,
                source="Agent检索结果",
                source_id=f"research-{idx}",
                round=round_num,
            )
        )

    # confirmed_findings 单独预留预算，优先进入最终上下文
    reserved_tokens = context_manager.estimate_tokens(confirmed_findings)
    selected = context_manager.build_context(query=question, items=candidates, reserved_tokens=reserved_tokens)
    supporting_text = context_manager.format_context(selected)

    parts = []
    if confirmed_findings:
        parts.append(f"[已确认核心结论（反思阶段蒸馏，优先采信）]\n{confirmed_findings}")
    if supporting_text:
        parts.append(f"[支持材料，含可引用来源编号]\n{supporting_text}")
    final_context = "\n\n---\n\n".join(parts)

    print("========== Context Manager 完成 ==========")
    print(f"原始结果: {len(research_results)}")
    print(f"最终结果: {len(selected)}")
    print(f"confirmed_findings 占用预算: {reserved_tokens} tokens")

    return {"final_context": final_context}

# ==============================
# 构建严密的 Graph 拓扑
# ==============================
builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("analyze_and_decompose", analyze_and_decompose)
builder.add_node("execute_search_branch", execute_search_branch)
builder.add_node("reflection_test", reflection_test)
builder.add_node("finalize_answer", finalize_answer)
# ============================================================
# 【新增节点】
# ============================================================

builder.add_node(
    "context_manager",
    context_manager_node,
)

# 1. 起点 -> 意图拆解
builder.add_edge(START, "analyze_and_decompose")

# 2. 意图拆解 -> 触发路由调度 (Send并发生效)
# ★ 改动：目的地从 ["execute_search_branch", "finalize_answer"]
# 改成 ["execute_search_branch", "context_manager"]，
# 跟 dispatch_searches 的返回值改动保持一致。
builder.add_conditional_edges(
    "analyze_and_decompose",
    dispatch_searches,
    ["execute_search_branch", "context_manager"]
)

# 3. 所有的独立并行分支执行完毕 -> 自动汇聚到反思节点
builder.add_edge("execute_search_branch", "reflection_test")

# 4. 反思节点 -> 继续循环调度 / 结束
builder.add_conditional_edges(
    "reflection_test",
    reflection_router,
    ["execute_search_branch", "context_manager"]
)

builder.add_edge(
    "context_manager",
    "finalize_answer",
)

# 5. 回答完毕 -> 结束
builder.add_edge("finalize_answer", END)

graph = builder.compile()