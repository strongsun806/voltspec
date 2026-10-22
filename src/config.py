"""Volt-Spec 전역 설정 모듈 (NF2005, F2005, F2006, F2014 준수)

청크 크기, 중복 길이, 검색 Top-k, 검색 방식(유사도/MMR), 경로 및 모델 설정을
코드 로직에서 분리하여 한곳에서 관리하고 실험·비교할 수 있도록 구성한다.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 1. 디렉터리 및 파일 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
METADATA_JSON_PATH = os.path.join(DATA_DIR, "ev_metadata.json")
WIKI_CACHE_PATH = os.path.join(DATA_DIR, "wiki_cache.json")

# 2. ChromaDB 컬렉션 이름 (2.3 컬렉션 분리 설계)
META_COLLECTION = "ev_meta"
WIKI_COLLECTION = "ev_wiki"

# 3. API 및 모델 설정 (NF2003 보안 관리)
GMS_DEFAULT_URL = "https://gms.ssafy.io/gmsapi/api.openai.com/v1/"


def get_api_key() -> str:
    return (os.getenv("GMS_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()


def get_base_url() -> str:
    raw_url = (os.getenv("OPENAI_BASE_URL") or GMS_DEFAULT_URL).strip()
    # 구버전 예시 URL(https://gms.ssafy.io/api/v1) 입력 시 실제 GMS 프록시 경로로 자동 보정
    if raw_url.rstrip("/") == "https://gms.ssafy.io/api/v1":
        return GMS_DEFAULT_URL
    return raw_url


def get_wiki_contact() -> str:
    return (os.getenv("WIKI_CONTACT") or "ssafy_student@example.com").strip()


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# 4. 청킹 및 검색 하이퍼파라미터 설정 (2.5 검색 설정 설계: F2005, F2006, F2014)
CHUNK_SIZE = int(os.getenv("VOLTSPEC_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("VOLTSPEC_CHUNK_OVERLAP", "200"))
WIKI_TOP_K = int(os.getenv("VOLTSPEC_TOP_K", "3"))

# 검색 방식: "similarity" (기본 유사도 검색) 또는 "mmr" (심화 F2014 다양성 검색)
SEARCH_TYPE = os.getenv("VOLTSPEC_SEARCH_TYPE", "similarity")
MMR_FETCH_K = int(os.getenv("VOLTSPEC_MMR_FETCH_K", "10"))
MMR_LAMBDA = float(os.getenv("VOLTSPEC_MMR_LAMBDA", "0.6"))
