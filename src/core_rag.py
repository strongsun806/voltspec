import os
import re
import sys
import json
from typing import TypedDict, List, Dict, Any
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    get_api_key,
    get_base_url,
    CHAT_MODEL,
    PERSIST_DIRECTORY,
    META_COLLECTION,
    WIKI_COLLECTION,
    WIKI_TOP_K,
    SEARCH_TYPE,
    MMR_FETCH_K,
    MMR_LAMBDA,
)
from index_metadata import embeddings

api_key = get_api_key()
base_url = get_base_url()

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.0,
    openai_api_key=api_key or "missing_key",
    openai_api_base=base_url,
)

CITE_REGEX = re.compile(r"\[근거\s*([\d,\s]+)\]")


# ── 1. LangGraph 상태(State) 정의 ──
class RAGState(TypedDict):
    ev_id: str                        # 선택된 차량 ID
    car_name: str                     # 선택된 차량 이름
    question: str                     # 사용자 질문
    top_k: int                        # 검색할 외부 문서 근거 수 (F2006)
    search_type: str                  # "similarity" 또는 "mmr" (F2014 심화)
    context_docs: List[Dict[str, Any]]  # 검색/병합된 근거 청크 목록
    answer: str                       # LLM이 생성한 최종 답변
    cited_sources: List[Dict[str, Any]]  # 답변에서 실제 인용된 출처 목록


def extract_cited_sources(answer: str, context_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """F2008: 답변 본문의 [근거 N] 또는 [근거 1, 2] 표기를 파싱해 실제 인용된 문서만 매핑"""
    raw_matches = CITE_REGEX.findall(answer)
    used_indices = {
        int(n.strip())
        for group in raw_matches
        for n in group.split(",")
        if n.strip().isdigit()
    }

    cited_sources = []
    for idx in sorted(used_indices):
        if 1 <= idx <= len(context_docs):
            doc = context_docs[idx - 1]
            cited_sources.append({
                "index": idx,
                "source_type": doc["source_type"],
                "title": doc["title"],
                "url": doc["url"],
                "content": doc["content"],
            })
    return cited_sources


# ── 2. F2006 & F2014: Node 1 - Retrieve (두 컬렉션 병합 검색 + 유사도/MMR 지원) ──
def retrieve_node(state: RAGState) -> Dict[str, Any]:
    ev_id = str(state["ev_id"])
    car_name = state["car_name"]
    question = state["question"]
    top_k = state.get("top_k") or WIKI_TOP_K
    search_type = state.get("search_type") or SEARCH_TYPE

    merged_docs = []
    seen_texts = set()

    # 1) ev_meta 컬렉션에서 선택 차량의 메타데이터 청크를 항상 근거에 기본 포함 (F2006)
    meta_store = Chroma(
        collection_name=META_COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    meta_results = meta_store.get(ids=[f"EV_{ev_id}"])
    if meta_results and meta_results.get("documents"):
        meta_content = meta_results["documents"][0]
        merged_docs.append({
            "source_type": "차량 메타데이터",
            "title": f"{car_name} 공식 제원 (ev_metadata.json)",
            "url": "내부 등록 데이터 (ev_meta)",
            "content": meta_content,
        })
        seen_texts.add(meta_content.strip())

    # 2) ev_wiki 컬렉션에서 의미 유사도 또는 MMR(다양성) 기반 검색 (F2006, F2014)
    wiki_store = Chroma(
        collection_name=WIKI_COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    wiki_results = []
    try:
        if search_type == "mmr":
            # F2014 심화: MMR 검색 적용 (후보를 넉넉히 가져온 뒤 다양성 기준으로 선별)
            wiki_results = wiki_store.max_marginal_relevance_search(
                query=question,
                k=top_k,
                fetch_k=max(MMR_FETCH_K, top_k * 3),
                lambda_mult=MMR_LAMBDA,
                filter={"ev_id": ev_id},
            )
        else:
            wiki_results = wiki_store.similarity_search(
                query=question,
                k=top_k,
                filter={"ev_id": ev_id},
            )
    except Exception:
        wiki_results = []

    # 해당 차량 필터 결과가 없으면 차량명+질문으로 전체 위키 컬렉션 폴백 검색
    if not wiki_results:
        try:
            if search_type == "mmr":
                wiki_results = wiki_store.max_marginal_relevance_search(
                    query=f"{car_name} {question}",
                    k=top_k,
                    fetch_k=max(MMR_FETCH_K, top_k * 3),
                    lambda_mult=MMR_LAMBDA,
                )
            else:
                wiki_results = wiki_store.similarity_search(
                    query=f"{car_name} {question}",
                    k=top_k,
                )
        except Exception:
            wiki_results = []

    # 3) 중복 제거 및 근거 병합 (F2006)
    for doc in wiki_results:
        clean_text = doc.page_content.strip()
        if clean_text and clean_text not in seen_texts:
            seen_texts.add(clean_text)
            merged_docs.append({
                "source_type": doc.metadata.get("source_type", "한국어 위키백과"),
                "title": doc.metadata.get("doc_title", "위키백과 문서"),
                "url": doc.metadata.get("source_url", "https://ko.wikipedia.org"),
                "content": doc.page_content,
            })

    return {"context_docs": merged_docs}


# ── 3. F2007 & F2008: Node 2 - Generate (프롬프트 3 근거 기반 답변 생성 및 인용 매핑) ──
def generate_node(state: RAGState) -> Dict[str, Any]:
    context_docs = state["context_docs"]
    question = state["question"]
    car_name = state["car_name"]

    if not context_docs:
        return {
            "answer": "제공된 문서에서 확인되지 않습니다.",
            "cited_sources": [],
        }

    # 근거 텍스트 포맷팅 (출처 유형 구분: 차량 메타데이터 vs 한국어 위키백과)
    context_text = ""
    for idx, doc in enumerate(context_docs, start=1):
        context_text += f"\n[근거 {idx}] (출처 유형: {doc['source_type']} | 문서명: {doc['title']})\n{doc['content']}\n"

    # F2007: 프롬프트 3 - 근거 기반 답변 생성 프롬프트
    system_prompt = f"""당신은 전기차 및 배터리 시스템 전문 수석 분석 AI 'Volt-Spec'입니다.
반드시 제공된 [참조 문서]의 내용만을 근거로 삼아 사용자의 질문에 답하십시오.

[작성 및 인용 원칙]
1. 답변의 각 문장이나 주요 주장의 끝에는 반드시 그 내용이 출현한 정확한 근거 번호를 표기하십시오. (예: ...입니다 [근거 1]. ...로 확인됩니다 [근거 2].)
2. 기본 요약 제원([근거 1])에 없는 세부 수치(예: 10→80% 충전 시간, 세부 트림 사양, 역사·배경 등)를 서술할 때는 반드시 해당 내용이 포함된 위키백과 문서의 근거 번호([근거 2], [근거 3] 등)를 인용하십시오.
3. 제공된 문서에 명시되지 않은 사실이나 불확실한 수치는 절대 임의로 추측하지 마십시오.
4. 질문에 대한 내용이 문서에 없다면 추측하지 말고 반드시 "제공된 문서에서 확인되지 않습니다"라고 명확히 밝히십시오.
5. 기술 제원, 충전 속도, 플랫폼 명칭 등은 정확한 고유명사와 단위를 유지하여 건조하고 명확한 어조로 서술하십시오.

[참조 문서]
{context_text}
"""

    human_prompt = f"질문 대상 차량: {car_name}\n질문: {question}"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]).content.strip()
    except Exception:
        response = f"{car_name} 기본 제원 요약: {context_docs[0]['content']} [근거 1]."

    cited_sources = extract_cited_sources(response, context_docs)

    return {
        "answer": response,
        "cited_sources": cited_sources,
    }


# ── 4. LangGraph 워크플로우 빌드 (retrieve -> generate -> END) ──
def build_rag_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


rag_app = build_rag_graph()


def query_volt_spec(
    ev_id: str,
    car_name: str,
    question: str,
    top_k: int = WIKI_TOP_K,
    search_type: str = SEARCH_TYPE,
) -> Dict[str, Any]:
    """단일 인터페이스 RAG 질의 함수 (유사도 / MMR 및 Top-k 조정 지원)"""
    initial_state: RAGState = {
        "ev_id": str(ev_id),
        "car_name": car_name,
        "question": question,
        "top_k": top_k,
        "search_type": search_type,
        "context_docs": [],
        "answer": "",
        "cited_sources": [],
    }

    result = rag_app.invoke(initial_state)
    return {
        "answer": result["answer"],
        "cited_sources": result["cited_sources"],
        "all_contexts": result["context_docs"],
    }


# ── 5. F2010: 도메인 특화 기능 1 — 전기차·배터리 심층 기술 리포트 생성 ──
def generate_spec_report(ev_id: str, car_name: str, top_k: int = 4) -> Dict[str, Any]:
    """선택한 차량의 색인된 문서만을 근거로 3단 심층 기술 분석 리포트를 생성하고 근거를 함께 제시 (F2010)"""
    retrieval = retrieve_node({
        "ev_id": str(ev_id),
        "car_name": car_name,
        "question": f"{car_name} 플랫폼 배터리 충전 파워트레인 제원 특징 역사",
        "top_k": top_k,
        "search_type": "mmr",
        "context_docs": [],
        "answer": "",
        "cited_sources": [],
    })
    context_docs = retrieval["context_docs"]
    context_text = "\n".join(
        f"[근거 {i}] ({d['source_type']}: {d['title']})\n{d['content']}"
        for i, d in enumerate(context_docs, start=1)
    )

    prompt = f"""당신은 전기차·배터리 수석 엔지니어 'Volt-Spec'입니다.
아래 [참조 문서]에 있는 내용만 사용하여 '{car_name}'의 종합 기술 분석 리포트를 작성하십시오.
문서에 확인되지 않은 내용은 절대 지어내지 말고, 각 문장 끝에 반드시 [근거 N]을 표기하십시오.

[리포트 구성 형식]
### 1. 전동화 플랫폼 및 차급 포지셔닝
### 2. 고전압 배터리 및 충전 시스템 분석
### 3. 파워트레인 구동 특성 및 핵심 기술 요약

[참조 문서]
{context_text}
"""
    answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {
        "answer": answer,
        "cited_sources": extract_cited_sources(answer, context_docs),
        "all_contexts": context_docs,
    }


# ── 6. F2010: 도메인 특화 기능 2 — 전기차 기술 주장 팩트체크 (사실 여부 판정) ──
def verify_technical_claim(ev_id: str, car_name: str, claim: str, top_k: int = 3) -> Dict[str, Any]:
    """사용자가 입력한 전기차/배터리 관련 기술 주장을 색인된 문서 근거로 사실 여부를 판정 (F2010)"""
    retrieval = retrieve_node({
        "ev_id": str(ev_id),
        "car_name": car_name,
        "question": claim,
        "top_k": top_k,
        "search_type": "similarity",
        "context_docs": [],
        "answer": "",
        "cited_sources": [],
    })
    context_docs = retrieval["context_docs"]
    context_text = "\n".join(
        f"[근거 {i}] ({d['source_type']}: {d['title']})\n{d['content']}"
        for i, d in enumerate(context_docs, start=1)
    )

    prompt = f"""당신은 전기차 제원 및 배터리 팩트체크 검증관 'Volt-Spec'입니다.
오직 아래 [참조 문서]의 내용만을 근거로 사용자의 주장을 검증하십시오.

[출력 형식]
- **판정 결과**: `사실` / `문서 내용과 다름` / `문서에서 확인되지 않음` 중 하나를 첫 줄에 명시
- **판정 근거 설명**: 문서의 구체적인 수치와 내용을 인용하여 [근거 N]과 함께 서술 (문서에 없으면 확인되지 않는다고만 답할 것)

[검증 대상 차량]: {car_name}
[검증할 주장]: {claim}

[참조 문서]
{context_text}
"""
    answer = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    return {
        "answer": answer,
        "cited_sources": extract_cited_sources(answer, context_docs),
        "all_contexts": context_docs,
    }


# ── 7. F2011: 자연어 조건 기반 항목 탐색 ──
def search_vehicles_by_condition(condition_query: str, candidate_k: int = 12) -> List[Dict[str, Any]]:
    """사용자의 자연어 조건으로 ev_meta 컬렉션에서 후보를 검색한 뒤,
    LLM이 실제 조건에 부합하는 차량만 엄격히 선별하여 반환한다 (F2011)."""
    meta_store = Chroma(
        collection_name=META_COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )
    candidates = meta_store.similarity_search(query=condition_query, k=candidate_k)
    if not candidates:
        return []

    cand_list = []
    for doc in candidates:
        cand_list.append({
            "ev_id": doc.metadata.get("ev_id"),
            "name": doc.metadata.get("name"),
            "maker": doc.metadata.get("maker"),
            "segment": doc.metadata.get("segment"),
            "platform": doc.metadata.get("platform"),
            "battery": doc.metadata.get("battery"),
            "range": doc.metadata.get("range"),
            "drivetrain": doc.metadata.get("drivetrain"),
            "year": doc.metadata.get("year"),
            "spec_text": doc.page_content,
        })

    system_msg = SystemMessage(
        content=(
            "너는 전기차 조건 검색 필터링 엔진이다. "
            "사용자의 자연어 조건과 후보 전기차 목록(spec_text 포함)을 비교하여, "
            "사용자의 조건에 정확히 부합하는 차량만 선별하라. 조건과 맞지 않는 차량은 절대 포함하지 마라. "
            "출력은 반드시 JSON 배열 형식으로만 반환하며, 각 원소는 "
            "{\"ev_id\": \"차량ID\", \"reason\": \"조건 부합 근거 한 줄 요약\"} 형태여야 한다."
        )
    )
    user_msg = HumanMessage(
        content=f"사용자 검색 조건: {condition_query}\n\n후보 차량 목록:\n{json.dumps(cand_list, ensure_ascii=False)}"
    )

    try:
        res_text = llm.invoke([system_msg, user_msg]).content.strip()
        if "```" in res_text:
            res_text = re.sub(r"```(?:json)?", "", res_text).replace("```", "").strip()
        selected = json.loads(res_text)
        reason_map = {str(item["ev_id"]): item.get("reason", "조건 일치") for item in selected if "ev_id" in item}
        matched = []
        for c in cand_list:
            if str(c["ev_id"]) in reason_map:
                c["match_reason"] = reason_map[str(c["ev_id"])]
                matched.append(c)
        return matched
    except Exception:
        return cand_list[:4]


if __name__ == "__main__":
    TEST_EV_ID = "1"
    TEST_CAR_NAME = "아이오닉 5"
    TEST_QUESTION = "800V 초급속 충전 지원 여부와 10%에서 80%까지 충전하는 데 걸리는 시간은 얼마인가요?"

    print("\n🧪 [Volt-Spec LangGraph RAG 파이프라인 단독 테스트]")
    print("그래프 노드 목록:", list(rag_app.nodes.keys()))
    print(f"질문: {TEST_QUESTION}\n")

    res = query_volt_spec(TEST_EV_ID, TEST_CAR_NAME, TEST_QUESTION)

    print("================ [생성된 답변] ================")
    print(res["answer"])

    print("\n============== [답변에 사용된 근거 목록 (F2008)] ==============")
    if res["cited_sources"]:
        for src in res["cited_sources"]:
            print(f"- [근거 {src['index']}] {src['source_type']} | {src['title']} ({src['url']})")
    else:
        print("ⓘ 답변에 인용된 근거가 없습니다 (문서 외 질문 또는 불일치).")
