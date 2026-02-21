"""Build benchmark dataset for extraction + KG QA comparison."""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml

from .gemini_labeler import GeminiLabelError, GeminiLabeler, GeminiLabelerConfig
from .schema import QUESTION_TYPES, dump_jsonl, validate_dataset

# Curated enterprise-style seed graph used to synthesize benchmark items.
ORG_PROFILES: Dict[str, Dict[str, Any]] = {
    "台積電": {
        "founder": "張忠謀",
        "hq": "新竹市",
        "supplies_to": ["Apple", "NVIDIA", "AMD", "高通"],
        "produces": ["3奈米製程", "2奈米製程"],
        "uses": ["EUV微影設備"],
        "competes_with": ["三星電子", "英特爾"],
    },
    "三星電子": {
        "founder": "李秉喆",
        "hq": "韓國首爾",
        "supplies_to": ["高通", "特斯拉", "小米"],
        "produces": ["記憶體晶片", "手機SoC"],
        "uses": ["EUV微影設備"],
        "competes_with": ["台積電", "英特爾"],
    },
    "英特爾": {
        "founder": "Gordon Moore",
        "hq": "美國聖塔克拉拉",
        "supplies_to": ["Dell", "Lenovo"],
        "produces": ["CPU", "GPU"],
        "uses": ["FinFET製程"],
        "competes_with": ["台積電", "三星電子"],
    },
    "聯發科": {
        "founder": "蔡明介",
        "hq": "新竹市",
        "supplies_to": ["OPPO", "vivo", "小米"],
        "produces": ["手機SoC", "Wi-Fi晶片"],
        "uses": ["5奈米製程"],
        "competes_with": ["高通", "紫光展銳"],
    },
    "鴻海": {
        "founder": "郭台銘",
        "hq": "新北市",
        "supplies_to": ["Apple", "Sony", "任天堂"],
        "produces": ["伺服器", "電子代工服務"],
        "uses": ["工業機器人"],
        "competes_with": ["和碩", "廣達"],
    },
    "ASML": {
        "founder": "Arthur del Prado",
        "hq": "荷蘭費爾德霍芬",
        "supplies_to": ["台積電", "三星電子", "英特爾"],
        "produces": ["EUV微影設備"],
        "uses": ["高精度光學"],
        "competes_with": ["Nikon"],
    },
    "日月光": {
        "founder": "張虔生",
        "hq": "高雄市",
        "supplies_to": ["NVIDIA", "AMD", "博通"],
        "produces": ["先進封裝", "測試服務"],
        "uses": ["2.5D封裝"],
        "competes_with": ["Amkor"],
    },
    "台達電": {
        "founder": "鄭崇華",
        "hq": "台北市",
        "supplies_to": ["微軟", "Google"],
        "produces": ["電源管理系統", "資料中心電源"],
        "uses": ["碳化矽元件"],
        "competes_with": ["施耐德電機"],
    },
    "華碩": {
        "founder": "施崇棠",
        "hq": "台北市",
        "supplies_to": ["全球零售通路"],
        "produces": ["筆記型電腦", "主機板"],
        "uses": ["AI運算模組"],
        "competes_with": ["宏碁", "微星"],
    },
    "高通": {
        "founder": "Irwin Jacobs",
        "hq": "美國聖地牙哥",
        "supplies_to": ["Samsung", "小米", "OPPO"],
        "produces": ["手機SoC", "5G數據機"],
        "uses": ["3奈米製程"],
        "competes_with": ["聯發科", "紫光展銳"],
    },
}

QUESTION_DISTRIBUTION_TOTAL = {
    "fact": 25,
    "relation": 25,
    "multi_hop": 20,
    "comparison": 10,
    "count": 10,
    "boolean": 10,
}

SOURCE_TYPE_DISTRIBUTION = {
    "graph_seed": {
        "fact": 10,
        "relation": 8,
        "multi_hop": 5,
        "comparison": 3,
        "count": 2,
        "boolean": 2,
    },
    "gemini_synth": {
        "fact": 15,
        "relation": 17,
        "multi_hop": 15,
        "comparison": 7,
        "count": 8,
        "boolean": 8,
    },
}

REL_MAP = {
    "founder": "FOUNDED_BY",
    "hq": "HEADQUARTERED_IN",
    "supplies_to": "SUPPLIES_TO",
    "produces": "PRODUCES",
    "uses": "USES",
    "competes_with": "COMPETES_WITH",
}

ZH_NUM = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
}


@dataclass(frozen=True)
class BuilderConfig:
    dataset_path: Path
    random_seed: int
    use_gemini_labeling: bool
    allow_gemini_fallback: bool
    gemini_model: str
    gemini_timeout_seconds: int
    gemini_max_retries: int
    gemini_backoff_seconds: float
    gemini_rate_limit_seconds: float


@dataclass(frozen=True)
class CandidateItem:
    source_type: str
    question_type: str
    difficulty: str
    question_zh_tw: str
    context_text: str
    gold_triples: List[Dict[str, str]]
    gold_answer: Dict[str, Any]


def _set_canonical(values: Sequence[str]) -> str:
    unique = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return "、".join(unique)


def _set_aliases(values: Sequence[str]) -> List[str]:
    cleaned = [v for v in values if str(v).strip()]
    canonical = _set_canonical(cleaned)
    reverse = _set_canonical(list(reversed(cleaned)))
    conjunction = "與".join(cleaned)
    out = []
    for alias in [canonical, reverse, conjunction]:
        alias = alias.strip()
        if alias and alias != canonical and alias not in out:
            out.append(alias)
    return out


def _number_aliases(number: int) -> List[str]:
    aliases = [str(number)]
    if number in ZH_NUM:
        aliases.extend([ZH_NUM[number], f"{ZH_NUM[number]}個", f"{ZH_NUM[number]}家"])
    aliases.extend([f"{number}個", f"{number}家"])
    unique = []
    for alias in aliases:
        if alias not in unique:
            unique.append(alias)
    return unique


def _triple(subject: str, relation: str, object_: str) -> Dict[str, str]:
    return {"subject": subject, "relation": relation, "object": object_}


def _fact_candidates(source_type: str) -> List[CandidateItem]:
    items: List[CandidateItem] = []
    for org, profile in ORG_PROFILES.items():
        founder = profile["founder"]
        hq = profile["hq"]
        product = profile["produces"][0]

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="fact",
                difficulty="easy",
                question_zh_tw=f"{org}的創辦人是誰？",
                context_text=f"{org}由{founder}創立，總部位於{hq}。",
                gold_triples=[_triple(org, REL_MAP["founder"], founder), _triple(org, REL_MAP["hq"], hq)],
                gold_answer={
                    "answer_type": "string",
                    "canonical": founder,
                    "accepted_aliases": [f"{org}的創辦人是{founder}"],
                    "required_entities": [founder],
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="fact",
                difficulty="easy",
                question_zh_tw=f"{org}的總部在哪裡？",
                context_text=f"{org}的總部設在{hq}，由{founder}創立。",
                gold_triples=[_triple(org, REL_MAP["hq"], hq), _triple(org, REL_MAP["founder"], founder)],
                gold_answer={
                    "answer_type": "string",
                    "canonical": hq,
                    "accepted_aliases": [f"{org}總部位於{hq}"],
                    "required_entities": [hq],
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="fact",
                difficulty="medium",
                question_zh_tw=f"{org}主要生產什麼技術或產品？",
                context_text=f"{org}主要生產{product}，並持續擴大相關產能。",
                gold_triples=[_triple(org, REL_MAP["produces"], product)],
                gold_answer={
                    "answer_type": "string",
                    "canonical": product,
                    "accepted_aliases": [f"{org}主要生產{product}"],
                    "required_entities": [product],
                },
            )
        )
    return items


def _relation_candidates(source_type: str) -> List[CandidateItem]:
    items: List[CandidateItem] = []
    for org, profile in ORG_PROFILES.items():
        customers = profile["supplies_to"]
        competitors = profile["competes_with"]
        used_tech = profile["uses"]

        relation_context = (
            f"{org}供應給{_set_canonical(customers)}，"
            f"同時與{_set_canonical(competitors)}存在競爭關係。"
        )
        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="relation",
                difficulty="medium",
                question_zh_tw=f"{org}的合作夥伴（客戶）有哪些？",
                context_text=relation_context,
                gold_triples=[_triple(org, REL_MAP["supplies_to"], c) for c in customers],
                gold_answer={
                    "answer_type": "set",
                    "canonical": _set_canonical(customers),
                    "accepted_aliases": _set_aliases(customers),
                    "required_entities": list(customers),
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="relation",
                difficulty="medium",
                question_zh_tw=f"{org}的主要競爭對手是哪些公司？",
                context_text=relation_context,
                gold_triples=[_triple(org, REL_MAP["competes_with"], c) for c in competitors],
                gold_answer={
                    "answer_type": "set",
                    "canonical": _set_canonical(competitors),
                    "accepted_aliases": _set_aliases(competitors),
                    "required_entities": list(competitors),
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="relation",
                difficulty="medium",
                question_zh_tw=f"{org}使用哪些關鍵技術？",
                context_text=f"{org}目前使用{_set_canonical(used_tech)}，並與多家供應鏈夥伴合作。",
                gold_triples=[_triple(org, REL_MAP["uses"], t) for t in used_tech],
                gold_answer={
                    "answer_type": "set",
                    "canonical": _set_canonical(used_tech),
                    "accepted_aliases": _set_aliases(used_tech),
                    "required_entities": list(used_tech),
                },
            )
        )

    return items


def _multi_hop_candidates(source_type: str) -> List[CandidateItem]:
    items: List[CandidateItem] = []
    for org, profile in ORG_PROFILES.items():
        founder = profile["founder"]
        hq = profile["hq"]
        first_customer = profile["supplies_to"][0]
        first_tech = profile["uses"][0]

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="multi_hop",
                difficulty="hard",
                question_zh_tw=f"哪家公司由{founder}創立且總部位於{hq}？",
                context_text=f"{org}由{founder}創立，總部位於{hq}，並供應給{first_customer}。",
                gold_triples=[
                    _triple(org, REL_MAP["founder"], founder),
                    _triple(org, REL_MAP["hq"], hq),
                    _triple(org, REL_MAP["supplies_to"], first_customer),
                ],
                gold_answer={
                    "answer_type": "string",
                    "canonical": org,
                    "accepted_aliases": [f"答案是{org}"],
                    "required_entities": [org],
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="multi_hop",
                difficulty="hard",
                question_zh_tw=f"哪家公司同時使用{first_tech}且供應給{first_customer}？",
                context_text=f"{org}使用{first_tech}並供應給{first_customer}。另外，{org}也與其他公司競爭。",
                gold_triples=[
                    _triple(org, REL_MAP["uses"], first_tech),
                    _triple(org, REL_MAP["supplies_to"], first_customer),
                ],
                gold_answer={
                    "answer_type": "string",
                    "canonical": org,
                    "accepted_aliases": [f"{org}符合條件"],
                    "required_entities": [org],
                },
            )
        )

    # Global multi-hop set questions.
    for customer in ["Apple", "NVIDIA", "小米", "Google"]:
        matched = [
            org
            for org, profile in ORG_PROFILES.items()
            if customer in profile["supplies_to"] and profile["uses"]
        ]
        if not matched:
            continue
        triples: List[Dict[str, str]] = []
        sentences: List[str] = []
        for org in matched:
            tech = ORG_PROFILES[org]["uses"][0]
            triples.append(_triple(org, REL_MAP["uses"], tech))
            triples.append(_triple(org, REL_MAP["supplies_to"], customer))
            sentences.append(f"{org}使用{tech}並供應給{customer}")

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="multi_hop",
                difficulty="hard",
                question_zh_tw=f"哪些公司同時使用關鍵技術且供應給{customer}？",
                context_text="；".join(sentences) + "。",
                gold_triples=triples,
                gold_answer={
                    "answer_type": "set",
                    "canonical": _set_canonical(matched),
                    "accepted_aliases": _set_aliases(matched),
                    "required_entities": matched,
                },
            )
        )

    return items


def _comparison_candidates(source_type: str) -> List[CandidateItem]:
    pairs = [
        ("台積電", "英特爾"),
        ("台積電", "三星電子"),
        ("鴻海", "華碩"),
        ("日月光", "台達電"),
        ("聯發科", "華碩"),
        ("ASML", "華碩"),
        ("高通", "華碩"),
        ("鴻海", "英特爾"),
        ("台積電", "台達電"),
        ("三星電子", "華碩"),
    ]

    items: List[CandidateItem] = []
    for left, right in pairs:
        left_count = len(ORG_PROFILES[left]["supplies_to"])
        right_count = len(ORG_PROFILES[right]["supplies_to"])
        if left_count == right_count:
            continue
        winner = left if left_count > right_count else right

        triples = [_triple(left, REL_MAP["supplies_to"], c) for c in ORG_PROFILES[left]["supplies_to"]]
        triples.extend(_triple(right, REL_MAP["supplies_to"], c) for c in ORG_PROFILES[right]["supplies_to"])

        context = (
            f"{left}供應給{_set_canonical(ORG_PROFILES[left]['supplies_to'])}；"
            f"{right}供應給{_set_canonical(ORG_PROFILES[right]['supplies_to'])}。"
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="comparison",
                difficulty="hard",
                question_zh_tw=f"{left}與{right}相比，哪家公司供應給更多客戶？",
                context_text=context,
                gold_triples=triples,
                gold_answer={
                    "answer_type": "string",
                    "canonical": winner,
                    "accepted_aliases": [f"{winner}較多", f"{winner}供應客戶較多"],
                    "required_entities": [winner],
                },
            )
        )

    return items


def _count_candidates(source_type: str) -> List[CandidateItem]:
    items: List[CandidateItem] = []
    for org, profile in ORG_PROFILES.items():
        customers = profile["supplies_to"]
        products = profile["produces"]

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="count",
                difficulty="medium",
                question_zh_tw=f"{org}供應給幾家公司？",
                context_text=f"{org}供應給{_set_canonical(customers)}。",
                gold_triples=[_triple(org, REL_MAP["supplies_to"], c) for c in customers],
                gold_answer={
                    "answer_type": "number",
                    "canonical": str(len(customers)),
                    "accepted_aliases": _number_aliases(len(customers)),
                    "required_entities": [],
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="count",
                difficulty="medium",
                question_zh_tw=f"{org}生產幾種主要技術或產品？",
                context_text=f"{org}目前主要生產{_set_canonical(products)}。",
                gold_triples=[_triple(org, REL_MAP["produces"], p) for p in products],
                gold_answer={
                    "answer_type": "number",
                    "canonical": str(len(products)),
                    "accepted_aliases": _number_aliases(len(products)),
                    "required_entities": [],
                },
            )
        )

    return items


def _boolean_candidates(source_type: str) -> List[CandidateItem]:
    items: List[CandidateItem] = []

    for org, profile in ORG_PROFILES.items():
        customer_true = profile["supplies_to"][0]
        customer_false = "Meta"
        if customer_false in profile["supplies_to"]:
            customer_false = "OpenAI"

        shared_context = (
            f"{org}供應給{_set_canonical(profile['supplies_to'])}，"
            f"並使用{_set_canonical(profile['uses'])}。"
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="boolean",
                difficulty="medium",
                question_zh_tw=f"{org}是否供應給{customer_true}？",
                context_text=shared_context,
                gold_triples=[_triple(org, REL_MAP["supplies_to"], c) for c in profile["supplies_to"]],
                gold_answer={
                    "answer_type": "boolean",
                    "canonical": "是",
                    "accepted_aliases": ["正確", "yes", "true"],
                    "required_entities": [],
                },
            )
        )

        items.append(
            CandidateItem(
                source_type=source_type,
                question_type="boolean",
                difficulty="medium",
                question_zh_tw=f"{org}是否供應給{customer_false}？",
                context_text=shared_context,
                gold_triples=[_triple(org, REL_MAP["supplies_to"], c) for c in profile["supplies_to"]],
                gold_answer={
                    "answer_type": "boolean",
                    "canonical": "否",
                    "accepted_aliases": ["不是", "錯誤", "false", "no"],
                    "required_entities": [],
                },
            )
        )

    return items


def _candidate_pool_for_source(source_type: str) -> Dict[str, List[CandidateItem]]:
    pool = {
        "fact": _fact_candidates(source_type),
        "relation": _relation_candidates(source_type),
        "multi_hop": _multi_hop_candidates(source_type),
        "comparison": _comparison_candidates(source_type),
        "count": _count_candidates(source_type),
        "boolean": _boolean_candidates(source_type),
    }
    for qtype, rows in pool.items():
        if qtype not in QUESTION_TYPES:
            raise ValueError(f"Unsupported question type in pool: {qtype}")
        if not rows:
            raise ValueError(f"Candidate pool is empty for {qtype}")
    return pool


def _sample_from_pool(
    *,
    pool: Dict[str, List[CandidateItem]],
    quota: Dict[str, int],
    rng: random.Random,
) -> List[CandidateItem]:
    out: List[CandidateItem] = []
    for qtype in QUESTION_TYPES:
        needed = int(quota.get(qtype, 0))
        candidates = copy.deepcopy(pool[qtype])
        rng.shuffle(candidates)
        if len(candidates) < needed:
            raise ValueError(f"Insufficient candidates for {qtype}: need={needed}, got={len(candidates)}")
        out.extend(candidates[:needed])
    rng.shuffle(out)
    return out


def _with_gemini_labels(
    items: Sequence[CandidateItem],
    labeler: GeminiLabeler,
    *,
    allow_fallback: bool,
) -> List[CandidateItem]:
    out: List[CandidateItem] = []
    for item in items:
        try:
            labels = labeler.label_candidate(
                question_zh_tw=item.question_zh_tw,
                context_text=item.context_text,
                answer_type=str(item.gold_answer.get("answer_type", "string")),
            )
            answer = labels["gold_answer"]
            if (
                answer["answer_type"] in {"string", "set"}
                and not answer["required_entities"]
                and answer["canonical"].strip()
            ):
                answer["required_entities"] = [token.strip() for token in answer["canonical"].split("、") if token.strip()]

            out.append(
                CandidateItem(
                    source_type=item.source_type,
                    question_type=item.question_type,
                    difficulty=item.difficulty,
                    question_zh_tw=item.question_zh_tw,
                    context_text=item.context_text,
                    gold_triples=labels["gold_triples"],
                    gold_answer=answer,
                )
            )
        except GeminiLabelError:
            if not allow_fallback:
                raise
            out.append(item)
    return out


def _candidate_to_dataset_dict(item: CandidateItem, index: int) -> Dict[str, Any]:
    return {
        "id": f"Q{index:04d}",
        "source_type": item.source_type,
        "question_zh_tw": item.question_zh_tw,
        "context_text": item.context_text,
        "gold_triples": item.gold_triples,
        "gold_answer": item.gold_answer,
        "metadata": {
            "difficulty": item.difficulty,
            "question_type": item.question_type,
        },
    }


def _load_builder_config(path: Path) -> BuilderConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    dataset_cfg = payload.get("dataset", {})
    builder_cfg = payload.get("builder", {})

    dataset_path = Path(dataset_cfg.get("path", "genai_project/llm_kg/benchmark/datasets/kgqa_zh_tw_100_v1.jsonl"))

    return BuilderConfig(
        dataset_path=dataset_path,
        random_seed=int(builder_cfg.get("random_seed", 42)),
        use_gemini_labeling=bool(builder_cfg.get("use_gemini_labeling", True)),
        allow_gemini_fallback=bool(builder_cfg.get("allow_gemini_fallback", True)),
        gemini_model=str(builder_cfg.get("gemini_model", "gemini-3-pro-preview")),
        gemini_timeout_seconds=int(builder_cfg.get("gemini_timeout_seconds", 60)),
        gemini_max_retries=int(builder_cfg.get("gemini_max_retries", 3)),
        gemini_backoff_seconds=float(builder_cfg.get("gemini_backoff_seconds", 1.5)),
        gemini_rate_limit_seconds=float(builder_cfg.get("gemini_rate_limit_seconds", 0.4)),
    )


def _print_summary(items: Sequence[Dict[str, Any]], dataset_path: Path) -> None:
    source_counts = Counter(item["source_type"] for item in items)
    qtype_counts = Counter(item["metadata"]["question_type"] for item in items)

    print(f"[dataset] path={dataset_path}")
    print(f"[dataset] total={len(items)}")
    print("[dataset] source_counts=" + json.dumps(source_counts, ensure_ascii=False, sort_keys=True))
    print("[dataset] question_type_counts=" + json.dumps(qtype_counts, ensure_ascii=False, sort_keys=True))


def build_dataset(config: BuilderConfig) -> List[Dict[str, Any]]:
    rng = random.Random(config.random_seed)

    graph_pool = _candidate_pool_for_source("graph_seed")
    graph_items = _sample_from_pool(pool=graph_pool, quota=SOURCE_TYPE_DISTRIBUTION["graph_seed"], rng=rng)

    synth_pool = _candidate_pool_for_source("gemini_synth")
    synth_items = _sample_from_pool(pool=synth_pool, quota=SOURCE_TYPE_DISTRIBUTION["gemini_synth"], rng=rng)

    if config.use_gemini_labeling:
        try:
            labeler = GeminiLabeler(
                config=GeminiLabelerConfig(
                    model=config.gemini_model,
                    timeout_seconds=config.gemini_timeout_seconds,
                    max_retries=config.gemini_max_retries,
                    retry_backoff_seconds=config.gemini_backoff_seconds,
                    rate_limit_seconds=config.gemini_rate_limit_seconds,
                )
            )
            synth_items = _with_gemini_labels(
                synth_items,
                labeler,
                allow_fallback=config.allow_gemini_fallback,
            )
            print("[dataset] gemini labeling enabled for gemini_synth subset")
        except GeminiLabelError as exc:
            if not config.allow_gemini_fallback:
                raise
            print(f"[dataset][warn] gemini unavailable, fallback to deterministic labels: {exc}")

    all_items = graph_items + synth_items
    rng.shuffle(all_items)

    dataset_rows = [_candidate_to_dataset_dict(item, index + 1) for index, item in enumerate(all_items)]

    if len(dataset_rows) != 100:
        raise ValueError(f"Dataset size must be 100, got {len(dataset_rows)}")

    source_counts = Counter(row["source_type"] for row in dataset_rows)
    if source_counts.get("graph_seed", 0) != 30 or source_counts.get("gemini_synth", 0) != 70:
        raise ValueError(f"Invalid source ratio: {source_counts}")

    qtype_counts = Counter(row["metadata"]["question_type"] for row in dataset_rows)
    for qtype, needed in QUESTION_DISTRIBUTION_TOTAL.items():
        if qtype_counts.get(qtype, 0) != needed:
            raise ValueError(f"Invalid question type distribution for {qtype}: {qtype_counts.get(qtype, 0)} != {needed}")

    validate_dataset(dataset_rows)
    return dataset_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build zh-TW KG benchmark dataset")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to benchmark config yaml",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = _load_builder_config(args.config)
    rows = build_dataset(cfg)
    dump_jsonl(cfg.dataset_path, rows)
    _print_summary(rows, cfg.dataset_path)


if __name__ == "__main__":
    main()
