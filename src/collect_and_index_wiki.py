import os
import re
import sys
import html
import json
import time
import requests
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import (
    get_api_key,
    get_base_url,
    get_wiki_contact,
    CHAT_MODEL,
    PERSIST_DIRECTORY,
    WIKI_COLLECTION,
    WIKI_CACHE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from index_metadata import embeddings

api_key = get_api_key()
base_url = get_base_url()
wiki_contact = get_wiki_contact()

llm = ChatOpenAI(
    model=CHAT_MODEL,
    temperature=0.0,
    openai_api_key=api_key or "missing_key",
    openai_api_base=base_url,
)

# 위키백과 API 호출 설정 (NF2006, NF2007 준수)
WIKI_API_URL = "https://ko.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": f"VoltSpecApp/1.0 ({wiki_contact}) python-requests"}


# ── 수집 캐시 관리 (F2004, NF2002: 이미 수집·색인한 검색어는 다시 수집하지 않음) ──
def load_wiki_cache() -> dict:
    if os.path.exists(WIKI_CACHE_PATH):
        try:
            with open(WIKI_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_wiki_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(WIKI_CACHE_PATH), exist_ok=True)
    with open(WIKI_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ── 위키백과 HTTP 호출 (NF2006: 재시도 및 호출 간격 준수) ──
def request_wikipedia(params: dict, max_retries: int = 3) -> dict | list | None:
    wait = 1.5
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(WIKI_API_URL, headers=HEADERS, params=params, timeout=10)
            if res.status_code == 200:
                time.sleep(0.25)  # NF2006: 호출 간격 예절 준수
                return res.json()
            if res.status_code == 429:
                retry_after = float(res.headers.get("Retry-After") or wait)
                print(f"⚠️ [429 Too Many Requests] {retry_after:.1f}초 대기 후 재시도 ({attempt}/{max_retries})")
                time.sleep(retry_after)
                wait *= 2
                continue
        except Exception as e:
            print(f"⚠️ 위키백과 API 요청 오류 ({attempt}/{max_retries}): {e}")
            time.sleep(wait)
    return None


def search_wiki_candidates(query: str, limit: int = 4) -> list[dict]:
    """위키백과 검색어로 후보 문서의 제목과 요약(snippet) 목록을 검색 (F2004)"""
    data = request_wikipedia({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json",
        "formatversion": 2,
    })
    if not isinstance(data, dict):
        return []
    search_items = data.get("query", {}).get("search", [])
    results = []
    for item in search_items:
        clean_snippet = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", "")))
        results.append({
            "page_id": item.get("pageid"),
            "title": item.get("title"),
            "snippet": clean_snippet,
            "search_query": query,
        })
    return results


def search_wiki_titles(query: str, limit: int = 4) -> list[str]:
    """위키백과 검색어로 관련 문서 제목 리스트 반환"""
    return [item["title"] for item in search_wiki_candidates(query, limit=limit)]


def get_wiki_content(title: str) -> dict | None:
    """위키백과 문서 본문(plain text) 및 원문 URL 조회 (F2004)"""
    data = request_wikipedia({
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "titles": title,
        "inprop": "url",
        "redirects": 1,
        "format": "json",
        "formatversion": 2,
    })
    if not isinstance(data, dict):
        return None
    pages = data.get("query", {}).get("pages", [])
    for page_data in pages:
        if not page_data.get("missing"):
            return {
                "page_id": page_data.get("pageid"),
                "title": page_data.get("title"),
                "text": page_data.get("extract", ""),
                "url": page_data.get("fullurl", f"https://ko.wikipedia.org/wiki/{title}"),
            }
    return None


# ── F2003: 프롬프트 1 - 도메인 전문가 검색어 추출 ──
def extract_search_keywords(car_name: str, question: str, maker: str = "", platform: str = "") -> list[str]:
    system_msg = SystemMessage(
        content=(
            "너는 전기차 및 고전압 배터리 시스템 전문 수석 엔지니어다. "
            "주어진 전기차 명칭과 사용자 질문을 분석하여, 한국어 위키백과에서 기술적 근거를 찾기 위한 검색 키워드 3~5개를 선정하라. "
            "반드시 선택된 차량 명칭 자체는 첫 번째 키워드로 포함해야 한다. "
            "결과는 오직 JSON 문자열 배열 형식(예: [\"키워드1\", \"키워드2\"])으로만 출력하라."
        )
    )
    context_hint = f" (제조사: {maker}, 플랫폼: {platform})" if (maker or platform) else ""
    user_msg = HumanMessage(content=f"차량명: {car_name}{context_hint}\n질문: {question}")

    try:
        response = llm.invoke([system_msg, user_msg]).content.strip()
        if "```" in response:
            response = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
        keywords = json.loads(response)
        if not isinstance(keywords, list):
            keywords = [car_name]
        # F2003: 선택한 항목 자체는 항상 첫 번째 검색 대상에 포함
        if car_name in keywords:
            keywords.remove(car_name)
        keywords.insert(0, car_name)
        return keywords[:5]
    except Exception:
        fallback = [car_name]
        if platform and platform not in fallback:
            fallback.append(platform)
        fallback.extend(["전기차", "리튬 이온 배터리"])
        return fallback[:4]


# ── F2004: 프롬프트 2 - 후보 문서 선별 및 본문 수집 ──
def filter_and_collect_docs(
    car_name: str,
    candidates: list,
    return_details: bool = False,
) -> list[dict] | tuple[list[dict], list[str], list[str]]:
    """후보 목록(문자열 제목 리스트 또는 {title, snippet} 딕셔너리 리스트)에서
    프롬프트 2로 관련성이 높은 문서만 선별하여 본문과 원문 URL을 수집한다."""
    normalized_candidates = []
    for c in candidates:
        if isinstance(c, dict):
            normalized_candidates.append(c)
        else:
            normalized_candidates.append({"title": str(c), "snippet": ""})

    candidate_titles = [c["title"] for c in normalized_candidates]

    system_msg = SystemMessage(
        content=(
            "너는 전기차 기술 문서 선별 전문가다. "
            "검색된 위키백과 후보 문서(제목 및 요약) 목록 중, 해당 차량의 제원, 파워트레인, 플랫폼, 배터리, 모터, 제조사 기술과 연관성이 높은 문서만 선별하라. "
            "단, 차량 명칭과 직접 일치하거나 차량 자체를 다룬 문서는 어떤 제외 규칙보다 우선하여 무조건 포함해야 한다. "
            "결과는 선별된 문서의 제목(title)들만 JSON 문자열 배열 형식으로 반환하라."
        )
    )
    user_msg = HumanMessage(
        content=f"기준 차량: {car_name}\n후보 문서 목록: {json.dumps(normalized_candidates, ensure_ascii=False)}"
    )

    try:
        response = llm.invoke([system_msg, user_msg]).content.strip()
        if "```" in response:
            response = re.sub(r"```(?:json)?", "", response).replace("```", "").strip()
        selected_titles = json.loads(response)
        if not isinstance(selected_titles, list):
            selected_titles = candidate_titles[:3]
    except Exception:
        domain_terms = (car_name, "전기", "배터리", "리튬", "충전", "플랫폼", "모터", "자동차", "E-GMP", "테슬라")
        selected_titles = [
            t for t in candidate_titles if any(term in t for term in domain_terms)
        ][:4] or candidate_titles[:3]

    # F2004: 선택한 항목 자체를 다룬 문서는 선별·길이 등 어떤 제외 규칙보다 우선하여 항상 포함
    for t in candidate_titles:
        if (t == car_name or car_name in t) and t not in selected_titles:
            selected_titles.insert(0, t)

    rejected_titles = [t for t in candidate_titles if t not in selected_titles]

    collected_docs = []
    for title in selected_titles:
        doc_data = get_wiki_content(title)
        if not doc_data:
            continue
        text_len = len(doc_data["text"].strip())
        # 차량 본체 문서는 길이 제한 없이 무조건 포함, 일반 배경 문서는 50자 초과 시 포함
        is_self_doc = (title == car_name) or (car_name in title)
        if is_self_doc or text_len > 50:
            collected_docs.append(doc_data)
            print(f"  📄 [선별/수집 완료] {doc_data['title']} ({text_len:,}자) - {doc_data['url']}")

    if rejected_titles:
        print(f"  🗑️ [선별 제외 문서] {rejected_titles}")

    if return_details:
        return collected_docs, selected_titles, rejected_titles
    return collected_docs


# ── F2005: 청킹 및 ChromaDB (ev_wiki) 별도 컬렉션 색인 ──
def chunk_and_index_wiki(
    ev_id: str,
    car_name: str,
    docs: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents = []
    ids = []
    collected_date = datetime.now().strftime("%Y-%m-%d")

    for doc in docs:
        chunks = splitter.split_text(doc["text"])
        for idx, chunk_text in enumerate(chunks):
            # 2.3 청크 메타데이터 설계 적용
            meta = {
                "ev_id": str(ev_id),
                "car_name": car_name,
                "doc_title": doc["title"],
                "page_id": str(doc.get("page_id", "")),
                "source_url": doc["url"],
                "source_type": "한국어 위키백과",
                "chunk_id": idx,
                "collected_at": collected_date,
            }
            # F2005 식별자 규칙: {ev_id}:{doc_title}:{chunk_index}
            chunk_unique_id = f"{ev_id}:{doc['title']}:{idx}"

            documents.append(Document(page_content=chunk_text, metadata=meta))
            ids.append(chunk_unique_id)

    if not documents:
        print("⚠️ 색인할 위키 청크가 없습니다.")
        return 0

    vectorstore = Chroma(
        collection_name=WIKI_COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    print(
        f"⏳ '{car_name}' 관련 위키 문서 {len(documents)}개 청크를 "
        f"(청크 크기={chunk_size}, 중복={chunk_overlap}) 임베딩/색인 중..."
    )
    vectorstore.add_documents(documents=documents, ids=ids)

    total_chunks = vectorstore._collection.count()
    print(f"✅ 색인 완료! 컬렉션 '{WIKI_COLLECTION}'의 누적 청크 수: {total_chunks}건")
    return len(documents)


# ── 전체 수집·색인 파이프라인 통합 실행 함수 (캐시 지원) ──
def process_wiki_for_vehicle(
    ev_id: str,
    car_name: str,
    sample_question: str,
    force_recollect: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    print("\n==========================================")
    print(f"🚀 위키 문서 수집 및 색인 파이프라인: {car_name} (ID: {ev_id})")
    print("==========================================")

    cache = load_wiki_cache()
    ev_cache = cache.get(str(ev_id), {"keywords": [], "titles": []})

    # 1) F2003: 검색어 추출
    print("\n[Step 1] 프롬프트 1 - 검색어 추출 중...")
    keywords = extract_search_keywords(car_name, sample_question)
    print(f"  🔍 추출된 검색어: {keywords}")

    # NF2002: 이미 수집한 검색어는 재수집 생략
    if not force_recollect:
        new_keywords = [kw for kw in keywords if kw not in ev_cache.get("keywords", [])]
        skipped_keywords = [kw for kw in keywords if kw in ev_cache.get("keywords", [])]
    else:
        new_keywords = list(keywords)
        skipped_keywords = []

    if skipped_keywords:
        print(f"  ⚡ [캐시 적중 - NF2002] 이미 수집된 검색어 재수집 생략: {skipped_keywords}")

    if not new_keywords:
        print("  ✅ 모든 추출 검색어가 이미 수집·색인되어 있어 외부 API 호출을 생략합니다.")
        return {
            "keywords": keywords,
            "new_keywords": [],
            "skipped_keywords": skipped_keywords,
            "candidates": [],
            "selected_titles": ev_cache.get("titles", []),
            "rejected_titles": [],
            "docs": [],
            "indexed_chunks": 0,
            "cached": True,
        }

    # 2) 후보 문서 검색
    print("\n[Step 2] 위키백과 후보 검색 중...")
    candidates = []
    seen_titles = set()
    for kw in new_keywords:
        for cand in search_wiki_candidates(kw, limit=3):
            if cand["title"] not in seen_titles:
                seen_titles.add(cand["title"])
                candidates.append(cand)
    print(f"  📋 발견된 후보 목록 ({len(candidates)}개): {[c['title'] for c in candidates]}")

    # 3) F2004: 프롬프트 2 - 후보 문서 선별 및 본문 로드
    print("\n[Step 3] 프롬프트 2 - 유효 문서 선별 및 본문 수집 중...")
    valid_docs, selected_titles, rejected_titles = filter_and_collect_docs(
        car_name, candidates, return_details=True
    )

    # 4) F2005: 문서 청킹 및 벡터 색인
    print("\n[Step 4] 문서 청킹 및 ev_wiki 컬렉션 색인...")
    indexed_chunks = chunk_and_index_wiki(
        ev_id,
        car_name,
        valid_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 캐시 업데이트 (NF2002)
    merged_keywords = list(dict.fromkeys(ev_cache.get("keywords", []) + new_keywords))
    merged_titles = list(dict.fromkeys(ev_cache.get("titles", []) + [d["title"] for d in valid_docs]))
    cache[str(ev_id)] = {
        "car_name": car_name,
        "keywords": merged_keywords,
        "titles": merged_titles,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_wiki_cache(cache)

    return {
        "keywords": keywords,
        "new_keywords": new_keywords,
        "skipped_keywords": skipped_keywords,
        "candidates": [c["title"] for c in candidates],
        "selected_titles": selected_titles,
        "rejected_titles": rejected_titles,
        "docs": valid_docs,
        "indexed_chunks": indexed_chunks,
        "cached": False,
    }


if __name__ == "__main__":
    TEST_EV_ID = "1"
    TEST_EV_NAME = "아이오닉 5"
    TEST_QUESTION = "800V 초급속 충전 지원 여부와 충전 시간 스펙을 알고 싶어"

    process_wiki_for_vehicle(TEST_EV_ID, TEST_EV_NAME, TEST_QUESTION, force_recollect=True)
