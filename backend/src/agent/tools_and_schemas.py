from typing import List, Literal
from pydantic import BaseModel, Field


class DecomposedQuery(BaseModel):

    query: str = Field(
        description="具体检索问题"
    )

    route: Literal[
        "rag",
        "web",
        "auto"
    ] = Field(
        default="auto",
        description="选择检索来源"
    )


class DecompositionPlan(BaseModel):

    queries: List[DecomposedQuery]


class NewReflection(BaseModel):

    is_sufficient: bool = Field(
        description="当前信息是否足够回答问题"
    )
    key_findings: str = Field(
        description="合并'已确认信息摘要'和'本轮新增检索结果'后，"
                    "更新出的一份完整、自洽的摘要（替换旧摘要，不是拼接）。"
                    "必须逐字保留原文中的引用标记(如 [xxxx-WEB-1])和具体数字/编号。"
    )
    follow_up_queries: List[DecomposedQuery] = Field(
        default_factory=list,
        description="不足时生成补充查询"
    )