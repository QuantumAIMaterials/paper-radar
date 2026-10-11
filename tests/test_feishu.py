from datetime import UTC, datetime

import httpx

from paper_radar.feishu import (
    FeishuClient,
    build_alert_card,
    build_card,
    build_deep_read_card,
    build_digest_card,
    build_signature,
)
from paper_radar.models import DeepRead, MatchResult, Paper, Recommendation


def test_signature_is_stable() -> None:
    expected = "mbm4Y4oluIPQ00qlBIhX8vAZ0EKv3nw0LuTb91jPL84="
    assert build_signature("test-secret", 1_700_000_000) == expected


def test_card_contains_links_and_priority() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00001",
        title="Graphene paper",
        authors=("A. Researcher",),
        abstract="Abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00001",
        pdf_url="https://arxiv.org/pdf/2608.00001",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=3,
        reason="Directly relevant",
        key_relevance=("graphene",),
        title_zh="",
        summary_zh="",
        used_ai=False,
    )
    card = build_card(recommendation, MatchResult(3, ("graphene",), ()))
    assert card["header"]["template"] == "red"
    buttons = card["elements"][1]["actions"]
    assert buttons[0]["url"] == paper.abstract_url
    assert buttons[1]["url"] == paper.pdf_url


def test_card_uses_complete_ai_recommendation() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00002",
        title="Excitons in a two-dimensional semiconductor",
        authors=("A. Researcher",),
        abstract="Original abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00002",
        pdf_url="https://arxiv.org/pdf/2608.00002",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=3,
        reason="与二维材料激子研究直接相关。",
        key_relevance=("二维半导体", "激子"),
        title_zh="二维半导体中的激子",
        summary_zh="本文研究单层半导体中的激子和光学性质。",
        used_ai=True,
    )

    card = build_card(
        recommendation,
        MatchResult(6, ("two-dimensional materials",), ("exciton",)),
    )

    assert card["header"]["title"]["content"] == recommendation.title_zh
    content = card["elements"][0]["text"]["content"]
    assert recommendation.summary_zh in content
    assert recommendation.reason in content
    assert "二维半导体, 激子" in content


def test_digest_card_groups_recommendations_in_chinese() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00003",
        title="Strain-tunable excitons in a monolayer semiconductor",
        authors=("A. Researcher", "B. Scientist"),
        abstract="Original abstract",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00003",
        pdf_url="https://arxiv.org/pdf/2608.00003",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=3,
        reason="与二维材料的激子调控和光学研究直接相关。",
        key_relevance=("二维半导体", "应变工程", "激子"),
        title_zh="单层半导体中的应变可调激子",
        summary_zh="本文研究应变对单层半导体激子和光学性质的影响。",
        used_ai=True,
        priority_score=86,
        group_fit_score=5,
        novelty_score=4,
        method_value_score=4,
        evidence_score=3,
        study_type="实验",
        reading_action="精读",
        quality_signals=("包含具体实验方法", "给出了明确物理结果"),
    )

    card = build_digest_card(
        [recommendation],
        {paper.paper_id: MatchResult(8, ("monolayer",), ("exciton",))},
        generated_at=now,
    )

    assert card["header"]["title"]["content"] == "二维量子材料论文速递 | 2026-08-14"
    assert card["header"]["template"] == "red"
    assert card["schema"] == "2.0"
    elements = card["body"]["elements"]
    assert len(elements) == 1
    panel = elements[0]
    assert panel["tag"] == "collapsible_panel"
    assert panel["expanded"] is False
    assert panel["header"]["icon_position"] == "right"
    assert panel["header"]["title"]["content"] == "今日精选 1 篇 · 展开全文"
    intro = panel["elements"][0]["text"]["content"]
    assert intro == "今天 Top 1 里对 二维量子材料 比较值得看的："
    detail_blocks = [
        element
        for element in panel["elements"]
        if element["tag"] == "div"
        and "**做什么：**" in element["text"]["content"]
    ]
    assert len(detail_blocks) == 1
    content = detail_blocks[0]["text"]["content"]
    assert recommendation.title_zh in content
    assert "（推荐 #1）" in content
    assert "**做什么：**" in content
    assert "**和我们组的关系：**" in content
    assert "arXiv / 2026-08-14" in content
    assert f"[{paper.title}]({paper.abstract_url})" in content
    assert f"[PDF]({paper.pdf_url})" in content
    assert "关键相关性" not in content
    assert "英文题目" not in content
    assert "作者：" not in content
    assert "优先级" not in content
    assert not any(element["tag"] == "action" for element in panel["elements"])
    assert "分类：" not in content
    assert "评分依据" not in content
    assert "质量信号" not in content
    assert "关键词：" not in content

    multi_card = build_digest_card(
        [recommendation, recommendation],
        {paper.paper_id: MatchResult(8, ("monolayer",), ("exciton",))},
        generated_at=now,
    )
    assert len(multi_card["body"]["elements"]) == 1
    digest_panel = multi_card["body"]["elements"][0]
    assert digest_panel["tag"] == "collapsible_panel"
    assert digest_panel["expanded"] is False
    assert "今日精选 2 篇" in digest_panel["header"]["title"]["content"]
    detail_blocks = [
        element
        for element in digest_panel["elements"]
        if element["tag"] == "div"
        and "**做什么：**" in element["text"]["content"]
    ]
    assert len(detail_blocks) == 2
    footer = digest_panel["elements"][-1]["text"]["content"]
    assert footer == "如果明天没有新增值得看的论文，我会保持安静。"
    assert not any(element["tag"] == "note" for element in digest_panel["elements"])


def test_digest_card_uses_quantum_ai_profile_text() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00004",
        title="A foundation model for materials discovery",
        authors=("A. Researcher",),
        abstract="A materials model.",
        published=now,
        updated=now,
        categories=("cond-mat.mtrl-sci",),
        abstract_url="https://arxiv.org/abs/2608.00004",
        pdf_url="https://arxiv.org/pdf/2608.00004",
    )
    recommendation = Recommendation(
        paper=paper,
        relevance_score=2,
        reason="与课题组的材料智能研究直接相关。",
        key_relevance=("材料基础模型",),
        title_zh="用于材料发现的基础模型",
        summary_zh="本文提出用于材料发现的基础模型。",
        used_ai=True,
    )

    card = build_digest_card(
        [recommendation],
        {paper.paper_id: MatchResult(5, ("foundation model",), (), ("materials",))},
        generated_at=now,
        digest_title="Quantum AI Materials 论文速递",
        digest_intro="量子科学、人工智能与材料研究的交叉方向",
    )

    assert card["header"]["title"]["content"] == (
        "Quantum AI Materials 论文速递 | 2026-08-14"
    )
    assert card["body"]["elements"][0]["elements"][0]["text"]["content"] == (
        "今天 Top 1 里对 Quantum AI Materials 比较值得看的："
    )


def test_digest_card_rejects_empty_recommendations() -> None:
    try:
        build_digest_card([], {})
    except ValueError as error:
        assert "empty" in str(error)
    else:
        raise AssertionError("Expected an empty digest to be rejected")


def test_deep_read_card_contains_pdf_grounded_sections() -> None:
    now = datetime(2026, 8, 14, tzinfo=UTC)
    paper = Paper(
        paper_id="2608.00005",
        title="Consensus mapping for superconductivity",
        authors=("A. Researcher", "B. Scientist"),
        abstract="We map scientific consensus.",
        published=now,
        updated=now,
        categories=("cond-mat.supr-con",),
        abstract_url="https://arxiv.org/abs/2608.00005",
        pdf_url="https://arxiv.org/pdf/2608.00005",
    )
    deep_read = DeepRead(
        paper=paper,
        title_zh="高温超导科学共识图谱",
        selection_reason="为量子材料文献智能体提供了可迁移的方法范式。",
        one_sentence_summary="论文使用大模型抽取超导机制支持并构建引用图谱。",
        technical_route=("文献池：摘要 -> 机制抽取 -> 共识图谱",),
        takeaways=("共识随材料族和时间演化。",),
        advances=("从文献总结推进到动态共识建模。",),
        limitations=("高被引筛选可能放大主流叙事。",),
        group_inspirations=("先用千篇论文验证量子材料 ontology。",),
        author_context=("通讯作者单位由论文首页确认。",),
    )

    card = build_deep_read_card(deep_read)
    serialized = str(card)

    assert card["schema"] == "2.0"
    assert card["header"]["title"]["content"].startswith("论文速读｜")
    assert "elements" not in card
    panel = card["body"]["elements"][0]
    assert panel["tag"] == "collapsible_panel"
    assert panel["expanded"] is False
    assert panel["header"]["icon_position"] == "right"
    assert panel["header"]["title"]["content"].endswith("展开全文")
    assert paper.abstract_url in serialized
    assert paper.pdf_url in serialized
    assert "A. Researcher, B. Scientist" in serialized
    assert "技术路线" in serialized
    assert "Takeaway" in serialized
    assert "先进在哪里" in serialized
    assert "我对局限的判断" in serialized
    assert "作者信息" in serialized
    assert "对我们组的启发" in serialized
    assert "关键结论请以原文为准" in serialized
    assert all(element["tag"] in {"markdown", "hr"} for element in panel["elements"])


def test_alert_card_uses_fatal_template() -> None:
    card = build_alert_card("论文雷达运行失败", "请检查日志", fatal=True)
    assert card["header"]["template"] == "red"
    assert "请检查日志" in card["elements"][0]["text"]["content"]


def test_client_retries_feishu_frequency_limit(monkeypatch) -> None:
    responses = [
        {"code": 11232, "msg": "frequency limited"},
        {"code": 0, "msg": "success"},
    ]
    calls = []
    sleeps = []

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        request = httpx.Request("POST", url)
        return httpx.Response(200, json=responses.pop(0), request=request)

    monkeypatch.setattr("paper_radar.feishu.httpx.post", fake_post)
    monkeypatch.setattr("paper_radar.feishu.time.sleep", sleeps.append)

    client = FeishuClient(
        "https://example.test/webhook",
        retry_delays=(0.25,),
    )
    client.send_alert("test", "retry")

    assert len(calls) == 2
    assert sleeps == [0.25]
