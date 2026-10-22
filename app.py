import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv

# src 모듈 경로 추가
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from config import (
    get_api_key,
    META_COLLECTION,
    WIKI_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    WIKI_TOP_K,
)
from index_metadata import get_collection_count
from collect_and_index_wiki import process_wiki_for_vehicle, load_wiki_cache
from core_rag import (
    query_volt_spec,
    generate_spec_report,
    verify_technical_claim,
    search_vehicles_by_condition,
)

load_dotenv()

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="Volt-Spec | 전기차·배터리 RAG 분석 서비스",
    page_icon="⚡",
    layout="wide",
)

DISCLAIMER_TEXT = (
    "⚠️ **면책 안내 (F2012)**: 본 서비스의 차량 제원 및 배터리 분석 답변은 공공 위키 문서를 근거로 생성된 참고용 데이터입니다. "
    "실제 차량의 안전 규정, 고전압 배터리 취급, 충전 및 정비 관련 사항은 반드시 제조사의 공식 취급 설명서를 준수해야 합니다."
)


@st.cache_data
def load_ev_metadata() -> list[dict]:
    json_path = os.path.join(BASE_DIR, "data", "ev_metadata.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_cited_sources(cited_sources: list[dict], all_contexts: list[dict] | None = None) -> None:
    """F2008: 답변에 실제로 사용된 근거 문서의 제목과 원문 출처를 표시하고, 인용이 없으면 안내"""
    if not cited_sources:
        st.caption("ⓘ 답변에 인용된 근거가 없습니다 (문서에서 확인되지 않은 내용 또는 인용 번호 미포함).")
    else:
        with st.expander(f"📚 답변에 사용한 근거 {len(cited_sources)}건 (F2008)", expanded=False):
            for src in cited_sources:
                idx = src.get("index")
                src_type = src.get("source_type", "문서")
                title = src.get("title", "제목 없음")
                url = src.get("url", "")
                content = src.get("content", "")

                if str(url).startswith("http"):
                    st.markdown(f"**[근거 {idx}]** `{src_type}` · **{title}** · [원문 링크]({url})")
                else:
                    st.markdown(f"**[근거 {idx}]** `{src_type}` · **{title}** ({url})")

                if content:
                    preview = content[:280] + ("..." if len(content) > 280 else "")
                    st.caption(preview)

    if all_contexts:
        with st.expander(f"🔎 검색된 전체 참조 문맥 보기 (총 {len(all_contexts)}건 · F2006 병합 결과)", expanded=False):
            for i, ctx in enumerate(all_contexts, start=1):
                st.markdown(f"**[문맥 {i}]** `{ctx['source_type']}` — {ctx['title']}")
                st.caption(ctx["content"][:300] + ("..." if len(ctx["content"]) > 300 else ""))


def ensure_wiki_indexed(
    ev_id: str,
    car_name: str,
    question: str,
    force_recollect: bool,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """선택 차량의 위키 문서를 수집·색인하고 결과 요약을 반환 (캐시 적중 시 생략)"""
    return process_wiki_for_vehicle(
        ev_id=ev_id,
        car_name=car_name,
        sample_question=question,
        force_recollect=force_recollect,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


# 2. 비정상 상황 점검 안내 (F2012: API 키 누락, 데이터/색인 없음 안내)
api_key = get_api_key()
if not api_key or api_key == "your_key_here":
    st.error("⚠️ `.env` 파일에 유효한 `GMS_KEY`가 설정되지 않았습니다. `.env`에 키를 입력한 뒤 다시 실행해 주세요. (F2012)")
    st.stop()

ev_list = load_ev_metadata()
if not ev_list:
    st.error("⚠️ `data/ev_metadata.json` 파일이 없습니다. 터미널에서 `python src/make_data.py`를 먼저 실행해 주세요. (F2012)")
    st.stop()

meta_count = get_collection_count(META_COLLECTION)
wiki_count = get_collection_count(WIKI_COLLECTION)

if meta_count == 0:
    st.warning(
        "⚠️ ChromaDB `ev_meta` 컬렉션에 색인된 데이터가 0건입니다. "
        "터미널에서 `python src/index_metadata.py`를 실행해 메타데이터를 색인해 주세요. (F2012)"
    )

# 3. 세션 상태 초기화 (F2009: 차량별 대화 기록 분리, F2011: 카드 선택 연동)
if "messages_by_ev" not in st.session_state:
    st.session_state.messages_by_ev = {}
if "selected_ev_id" not in st.session_state:
    st.session_state.selected_ev_id = str(ev_list[0]["id"])
if "condition_results" not in st.session_state:
    st.session_state.condition_results = None


def select_vehicle_callback(target_ev_id: str):
    """F2011 조건 탐색 카드에서 차량 선택 시 사이드바 선택 상태까지 즉시 동기화"""
    st.session_state.selected_ev_id = str(target_ev_id)
    st.session_state.maker_filter = "전체 제조사"


# 4. 사이드바 구성 (F2002 색인 건수 상시 표시, F2009 차량 선택, F2005/F2006/F2014 설정 조정)
with st.sidebar:
    st.title("⚡ Volt-Spec")
    st.caption("RAG 기반 전기차·배터리 기술 분석 지식 서비스")

    # F2002: 화면 왼쪽에 현재 색인 건수를 항상 표시
    st.markdown(
        f"📊 **메타데이터 색인(`ev_meta`)**: `{meta_count}건`  \n"
        f"📚 **위키 청크 색인(`ev_wiki`)**: `{wiki_count}건`"
    )
    st.divider()

    makers = ["전체 제조사"] + sorted(list({item["maker"] for item in ev_list}))
    selected_maker = st.selectbox("제조사 필터", makers, key="maker_filter")

    filtered_evs = (
        ev_list
        if selected_maker == "전체 제조사"
        else [item for item in ev_list if item["maker"] == selected_maker]
    )

    ev_id_to_item = {str(item["id"]): item for item in filtered_evs}
    if st.session_state.selected_ev_id not in ev_id_to_item:
        st.session_state.selected_ev_id = str(filtered_evs[0]["id"])

    id_options = list(ev_id_to_item.keys())
    current_idx = id_options.index(st.session_state.selected_ev_id)

    chosen_id = st.selectbox(
        "질문할 전기차 선택 (F2009)",
        options=id_options,
        index=current_idx,
        format_func=lambda eid: f"[{ev_id_to_item[eid]['maker']}] {ev_id_to_item[eid]['name']} ({ev_id_to_item[eid]['year']})",
    )
    st.session_state.selected_ev_id = chosen_id
    selected_ev = ev_id_to_item[chosen_id]
    ev_id = str(selected_ev["id"])
    car_name = selected_ev["name"]

    # 현재 차량의 수집된 위키 문서 캐시 현황 표시
    wiki_cache = load_wiki_cache()
    ev_cached_info = wiki_cache.get(ev_id, {})
    cached_titles = ev_cached_info.get("titles", [])

    st.subheader("📋 선택 차량 공식 제원 요약")
    st.markdown(
        f"""
- **차량명**: {selected_ev['name']} ({selected_ev['year']}년식)
- **제조사 / 차급**: {selected_ev['maker']} · {selected_ev['segment']}
- **전용/기반 플랫폼**: `{selected_ev['platform']}`
- **배터리 용량**: `{selected_ev['battery']}`
- **1회 충전 주행거리**: `{selected_ev['range']}`
- **구동 방식**: {selected_ev['drivetrain']}
- **핵심 특징**: {selected_ev['summary']}
- **수집된 위키 문서**: `{len(cached_titles)}건` {('(' + ', '.join(cached_titles) + ')') if cached_titles else '(첫 질문 시 자동 수집)'}
        """
    )

    with st.expander("⚙️ RAG 검색·청킹 설정 조정 (F2005·F2006·F2014)", expanded=False):
        search_type_label = st.radio(
            "검색 방식 (F2014 심화 비교)",
            options=["similarity", "mmr"],
            format_func=lambda x: "유사도 검색 (Similarity)" if x == "similarity" else "MMR 다양성 검색 (F2014 심화)",
            index=0,
        )
        top_k_val = st.slider("외부 문서 검색 수 (Top-k, F2006)", min_value=1, max_value=8, value=WIKI_TOP_K)
        chunk_size_val = st.select_slider("청크 크기 (F2005)", options=[500, 800, 1000, 1500], value=CHUNK_SIZE)
        chunk_overlap_val = st.select_slider("청크 중복 길이 (F2005)", options=[50, 100, 200, 300], value=CHUNK_OVERLAP)
        force_recollect = st.checkbox("위키 문서 강제 재수집·재색인", value=False)

    if st.button("🗑️ 현재 차량 대화 지우기", use_container_width=True):
        st.session_state.messages_by_ev[ev_id] = []
        st.rerun()

    st.divider()
    st.warning(DISCLAIMER_TEXT)
    st.caption("외부 문서 출처: 한국어 위키백과 (CC BY-SA 4.0)")

# 5. 메인 화면: 탭 구성 (탭 1: RAG 질의응답 | 탭 2: 도메인 특화 기능 F2010 | 탭 3: 자연어 조건 탐색 F2011)
st.title(f"⚡ {selected_ev['maker']} {car_name} ({selected_ev['year']}) — Volt-Spec 분석")
st.caption(
    f"플랫폼: **{selected_ev['platform']}** | 배터리: **{selected_ev['battery']}** | "
    f"주행거리: **{selected_ev['range']}** | 구동방식: **{selected_ev['drivetrain']}** | "
    f"현재 검색 설정: **{search_type_label.upper()} (Top-k={top_k_val})**"
)

tab_chat, tab_special, tab_explore = st.tabs([
    "💬 근거 기반 기술 질의응답 (F2006~F2009)",
    "🛠️ 도메인 특화 기능: 리포트 & 팩트체크 (F2010)",
    "🔎 자연어 조건 기반 전기차 탐색 (F2011)",
])

# ── 탭 1: 차량별 멀티턴 RAG 질의응답 ──
with tab_chat:
    with st.expander("💡 근거가 필요한 대표 질문 예시 (클릭하여 참고)", expanded=False):
        st.markdown(
            f"""
1. **800V 충전 및 플랫폼 검증**: `"{car_name}의 플랫폼({selected_ev['platform']}) 특징과 초급속 충전 지원 여부, 충전 시간 스펙은 어떻게 되나요?"`
2. **배터리 및 파워트레인 검증**: `"{car_name}에 탑재된 배터리 용량·특성과 구동 방식({selected_ev['drivetrain']})의 기술적 특징을 설명해 주세요."`
3. **문서 외 질문 방어 확인 (4.3 검증용)**: `"{car_name}의 2030년형 완전변경 모델 공식 출시 가격과 전고체 배터리 용량은 얼마인가요?"`
            """
        )

    if ev_id not in st.session_state.messages_by_ev:
        st.session_state.messages_by_ev[ev_id] = []

    current_messages = st.session_state.messages_by_ev[ev_id]

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_cited_sources(msg.get("cited_sources", []), msg.get("all_contexts"))

    question = st.chat_input(f"'{car_name}'의 배터리, 플랫폼, 충전, 파워트레인에 대해 질문해 보세요...")

    if question:
        current_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            # F2012: 문서 수집과 답변 생성 중 진행 상태 표시
            with st.status("🔍 외부 지식 수집 및 LangGraph RAG 파이프라인 실행 중...", expanded=True) as status:
                st.write(f"1️⃣ **[프롬프트 1·2 & 위키백과 수집]** `{car_name}` 질문 관련 외부 문서 점검 중...")
                collect_info = ensure_wiki_indexed(
                    ev_id=ev_id,
                    car_name=car_name,
                    question=question,
                    force_recollect=force_recollect,
                    chunk_size=chunk_size_val,
                    chunk_overlap=chunk_overlap_val,
                )
                if collect_info["cached"]:
                    st.caption(
                        f"⚡ [캐시 적중 - NF2002] 이미 수집된 검색어({', '.join(collect_info['keywords'])}) 재수집 생략 · "
                        f"기존 색인 문서({', '.join(collect_info['selected_titles'])}) 재사용"
                    )
                else:
                    st.caption(
                        f"🔍 추출 키워드: {', '.join(collect_info['keywords'])}  \n"
                        f"✅ 선별·색인 문서: {', '.join(collect_info['selected_titles']) or '없음'} "
                        f"(청크 {collect_info['indexed_chunks']}개 적재)  \n"
                        f"🗑️ 선별 제외 후보: {', '.join(collect_info['rejected_titles']) or '없음'}"
                    )

                st.write(
                    f"2️⃣ **[LangGraph RAG: retrieve -> generate]** `ev_meta` + `ev_wiki` 병합 검색 "
                    f"(`{search_type_label}`, `Top-k={top_k_val}`) 및 답변 생성 중..."
                )
                rag_result = query_volt_spec(
                    ev_id=ev_id,
                    car_name=car_name,
                    question=question,
                    top_k=top_k_val,
                    search_type=search_type_label,
                )
                status.update(
                    label=(
                        f"✅ 답변 생성 완료! (병합 문맥 {len(rag_result['all_contexts'])}건 · "
                        f"실제 인용 근거 {len(rag_result['cited_sources'])}건)"
                    ),
                    state="complete",
                    expanded=False,
                )

            answer = rag_result["answer"]
            cited_sources = rag_result["cited_sources"]
            all_contexts = rag_result["all_contexts"]

            st.markdown(answer)
            render_cited_sources(cited_sources, all_contexts)

        current_messages.append({
            "role": "assistant",
            "content": answer,
            "cited_sources": cited_sources,
            "all_contexts": all_contexts,
        })

# ── 탭 2: 도메인 특화 기능 (F2010: 스펙 리포트 생성 & 기술 주장 사실 여부 판정) ──
with tab_special:
    st.subheader(f"🛠️ {car_name} 도메인 특화 기능 (F2010)")
    st.caption("도메인 특화 기능 역시 색인된 `ev_meta` 및 `ev_wiki` 문서만을 근거로 동작하며 `[근거 N]` 출처를 함께 제시합니다.")

    col_rep, col_fc = st.columns(2)

    with col_rep:
        st.markdown("#### 📊 1. 전기차·배터리 3단 심층 기술 리포트 생성")
        st.write(f"선택된 **{car_name}**의 플랫폼, 배터리·충전 아키텍처, 파워트레인 특성을 종합 분석합니다.")
        if st.button(f"⚡ '{car_name}' 심층 기술 리포트 생성", use_container_width=True):
            with st.status("심층 리포트용 위키 문서 확인 및 리포트 생성 중...", expanded=True) as s1:
                ensure_wiki_indexed(
                    ev_id=ev_id,
                    car_name=car_name,
                    question=f"{car_name} 플랫폼 배터리 충전 파워트레인 제원",
                    force_recollect=False,
                    chunk_size=chunk_size_val,
                    chunk_overlap=chunk_overlap_val,
                )
                rep_res = generate_spec_report(ev_id=ev_id, car_name=car_name, top_k=top_k_val)
                s1.update(label="✅ 심층 기술 리포트 생성 완료!", state="complete", expanded=False)
            st.markdown(rep_res["answer"])
            render_cited_sources(rep_res["cited_sources"], rep_res["all_contexts"])

    with col_fc:
        st.markdown("#### ⚖️ 2. 전기차·배터리 기술 주장 팩트체크 (사실 여부 판정)")
        default_claim = f"{car_name}는 {selected_ev['platform']} 플랫폼을 기반으로 하며 배터리 용량은 {selected_ev['battery']}이다."
        claim_input = st.text_input("검증할 기술 주장 입력", value=default_claim)
        if st.button("🔍 근거 문서 기반 사실 여부 판정 실행", use_container_width=True):
            with st.status("근거 문서 검색 및 사실 여부 판정 중...", expanded=True) as s2:
                ensure_wiki_indexed(
                    ev_id=ev_id,
                    car_name=car_name,
                    question=claim_input,
                    force_recollect=False,
                    chunk_size=chunk_size_val,
                    chunk_overlap=chunk_overlap_val,
                )
                fc_res = verify_technical_claim(ev_id=ev_id, car_name=car_name, claim=claim_input, top_k=top_k_val)
                s2.update(label="✅ 사실 여부 판정 완료!", state="complete", expanded=False)
            st.markdown(fc_res["answer"])
            render_cited_sources(fc_res["cited_sources"], fc_res["all_contexts"])

# ── 탭 3: 자연어 조건 기반 항목 탐색 (F2011) ──
with tab_explore:
    st.subheader("🔎 자연어 조건 기반 전기차 탐색 (F2011)")
    st.caption(
        "자연어로 원하는 전기차 조건(플랫폼, 충전 전압, 배터리 종류/용량, 차급, 제조사 등)을 입력하면 "
        "`ev_meta` 벡터 검색 후 LLM이 조건에 부합하는 차량만 선별하여 카드로 제시합니다."
    )

    cond_query = st.text_input(
        "찾고 싶은 전기차 조건을 자연어로 입력하세요",
        placeholder="예: 800V 초급속 충전을 지원하는 현대·기아 전기차 / 배터리 100kWh 이상 대형 SUV / LFP 배터리를 탑재한 전기차",
    )
    if st.button("🚀 조건에 맞는 전기차 탐색", type="primary"):
        if not cond_query.strip():
            st.warning("검색할 조건을 입력해 주세요.")
        else:
            with st.spinner("ChromaDB `ev_meta` 후보 검색 및 LLM 조건 일치 차량 선별 중..."):
                st.session_state.condition_results = search_vehicles_by_condition(cond_query.strip())

    if st.session_state.condition_results is not None:
        results = st.session_state.condition_results
        if not results:
            st.info("입력하신 조건과 정확히 일치하는 전기차를 찾지 못했습니다. 다른 조건으로 검색해 보세요.")
        else:
            st.success(f"✅ 조건에 부합하는 전기차 **{len(results)}종**을 선별했습니다. 카드를 선택하면 해당 차량 대화로 이어집니다.")
            for item in results:
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(
                            f"### [{item['maker']}] {item['name']} ({item['year']}) — `{item['segment']}`"
                        )
                        st.markdown(
                            f"- **플랫폼**: `{item['platform']}` | **배터리**: `{item['battery']}` | "
                            f"**주행거리**: `{item['range']}` | **구동방식**: `{item['drivetrain']}`\n"
                            f"- **선별 근거**: {item.get('match_reason', '')}"
                        )
                    with c2:
                        st.button(
                            "👉 이 차량 선택",
                            key=f"select_btn_{item['ev_id']}",
                            on_click=select_vehicle_callback,
                            args=(str(item["ev_id"]),),
                            use_container_width=True,
                        )

st.divider()
st.caption(DISCLAIMER_TEXT)
