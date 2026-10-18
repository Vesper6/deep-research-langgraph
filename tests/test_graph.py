"""单元测试：只测不调模型的小函数，跑得快、不花钱。运行：python -m pytest -v"""

from deep_research.graph import after_reflect, format_hits, unique


def test_unique_removes_duplicates():
    a = {"source": "a.md", "text": "年假 12 天"}
    b = {"source": "b.md", "text": "餐补 120 元"}
    assert unique([a, b, a]) == [a, b]


def test_format_hits_numbers_from_1():
    hits = [{"source": "a.md", "text": "甲"}, {"source": "b.md", "text": "乙"}]
    assert format_hits(hits) == "[1] 甲\n[2] 乙"


def test_after_reflect_enough_goes_to_report():
    state = {"enough": True, "questions": []}
    assert after_reflect(state) == "write_report"


def test_after_reflect_not_enough_searches_again():
    state = {"enough": False, "questions": ["腾讯海外业务"]}
    sends = after_reflect(state)
    # TODO：写 2 个 assert —— sends 里有 1 个；sends[0].node 是 "research_one"
    assert len(sends) == 1
    assert sends[0].node == "research_one"

