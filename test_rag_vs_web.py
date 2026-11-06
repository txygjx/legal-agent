import os
import time
import json
from datetime import datetime

# 引入你的 graph
from agent.graph import graph


def run_single_test(question: str, mode: str) -> dict:
    """跑一次测试，返回结果字典"""

    # 设置消融开关
    if mode == "no_rag":
        os.environ["TEST_NO_RAG"] = "true"
    else:
        os.environ.pop("TEST_NO_RAG", None)

    start_time = time.time()

    try:
        result = graph.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        elapsed = time.time() - start_time

        # 取最后一条 AI 消息
        answer = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "content"):
                answer = msg.content
                break

        # 统计来源
        sources = result.get("sources_gathered", [])
        rag_sources = [s for s in sources if "RAG" in s.get("short_url", "")]
        web_sources = [s for s in sources if "WEB" in s.get("short_url", "")]

        return {
            "mode": mode,
            "question": question,
            "answer": answer,
            "elapsed_seconds": round(elapsed, 2),
            "answer_length": len(answer),
            "rag_source_count": len(rag_sources),
            "web_source_count": len(web_sources),
            "total_source_count": len(sources),
            "error": None
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "mode": mode,
            "question": question,
            "answer": "",
            "elapsed_seconds": round(elapsed, 2),
            "answer_length": 0,
            "rag_source_count": 0,
            "web_source_count": 0,
            "total_source_count": 0,
            "error": str(e)
        }


def run_all_tests(questions: list):
    """对所有问题跑两种模式，结果保存到文件"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_results_{timestamp}.json"
    all_results = []

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*50}")
        print(f"[{i}/{len(questions)}] 测试问题: {question}")

        # 跑带 RAG 的版本
        print(f"  → 模式: with_rag ...")
        result_rag = run_single_test(question, mode="with_rag")
        print(f"     耗时: {result_rag['elapsed_seconds']}s | "
              f"RAG来源: {result_rag['rag_source_count']} | "
              f"Web来源: {result_rag['web_source_count']}")

        # 跑纯 Web 的版本
        print(f"  → 模式: no_rag ...")
        result_web = run_single_test(question, mode="no_rag")
        print(f"     耗时: {result_web['elapsed_seconds']}s | "
              f"Web来源: {result_web['web_source_count']}")

        all_results.append({
            "question_id": i,
            "question": question,
            "with_rag": result_rag,
            "no_rag": result_web,
            # 留给你事后手动填写的评分
            "manual_score": {
                "with_rag_relevance": None,   # 0=不相关 1=部分相关 2=高度相关
                "no_rag_relevance": None,
                "with_rag_has_law_citation": None,  # True/False
                "no_rag_has_law_citation": None,
                "winner": None,   # "rag" / "web" / "tie"
                "notes": ""
            }
        })

        # 每跑完一题就保存，防止中途崩溃丢数据
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"  ✓ 已保存到 {output_file}")

    print(f"\n{'='*50}")
    print(f"测试完成，结果已保存到: {output_file}")
    return all_results, output_file


# ==============================
# 运行入口
# ==============================
if __name__ == "__main__":

    questions = [
        # RAG 应该更好（民法典前100条范围内）
        "民法典规定了哪些基本原则？",
        "自然人的民事权利能力从什么时候开始和终止？",
        "限制民事行为能力人的年龄标准是什么，可以独立实施哪些法律行为？",
        "监护人的职责有哪些，监护人资格可以被撤销吗？",
        "民法典对法人的定义和设立条件是怎么规定的？",
        # Web 应该更好（时效性内容，RAG里没有）
        "2025年以来有哪些涉及民事合同纠纷的典型判例？",
        "当前法院对网络虚假交易合同纠纷的主流判决倾向？",
        "最高法院最新发布的民事类司法解释有哪些？",
        "近期未成年人监护权纠纷案件的裁判思路？",
        "2025年民法领域有哪些新的立法或政策变化？",
    ]

    run_all_tests(questions)