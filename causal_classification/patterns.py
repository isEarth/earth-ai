"""
causal_patterns.py

한국어 문장에서 인과 관계(causal relationship)를 식별하기 위한 정규 표현식 패턴 리스트를 정의합니다.

사용 목적:
    - 텍스트 분석 및 문장 분류에서 인과적 단서(문장 연결 표현 또는 이유/결과 관련 어휘)를 탐지하기 위해 사용됩니다.
    - 각 패턴은 문장 내 인과 연결 표현을 식별할 수 있도록 설계되었습니다.

구성 요소:
    1. `prefixes`: '이', '그', '저' 같은 한국어 지시사 접두어
    2. `causal_phrases`: 조사 및 연결어미 중심의 인과 표현 리스트 (예: "러 니까", "런 고 로")
    3. `causal_keywords`: 단독으로 인과관계를 나타낼 수 있는 단어 또는 구 (예: "왜냐하면", "따라서", "원인")
    4. `CAUSAL_PATTERNS`: 위 요소들을 바탕으로 컴파일되지 않은 정규식 패턴 문자열 리스트

예시 사용:
    from causal_patterns import CAUSAL_PATTERNS
    for p in CAUSAL_PATTERNS:
        if re.search(p, sentence):
            print("인과 표현 발견!")

"""

# causal_patterns.py

import re

# 이/그/저 고려한 인과관계 패턴
prefixes = ["이", "그", "저"]
causal_phrases = [
    "도 그렇 ㄹ 것 이",
    "러 니까",
    "러 기에",
    "런 즉",
    "러 하 ㄴ 즉",
    "래서",
    "러므로",
    "러 니만큼",
    "러 니 만 하 지",
    "런 만큼",
    "런 고 로",
    "리하(?: 여| 어)?",
    "러 다가",
    "러 자",
    " 에",
    "렇 으므로",
    " 로써",
    "러 느라(?: 고)?",
    "렇 어 가지 고",
    "렇 어 가지 고",
    "러 다 보 니",
    "렇 게 하 어",
    "렇 게 되 자"
]

# 인과관계 단어 패턴
causal_keywords = [
    "고로", "따라서", "왜냐하면",
    "연고로", "시고로",
    "연즉", "한즉", "즉",
    "요컨대", "어쩐지", "어차피",
    "하기는", "그야", "마침내", "드디어", "끝내", "급기야",
    "거든", "잖아", "어서", "니까", "으니까", "길래",
    "느라", "느라고", "는데", "ㄴ다고",
    "결론", "결과", "바탕", "맥락", "원인", "이유",
    "까닭", "예증", "때문", "탓", "기인", "전제", "야기",
    "인하", "의하", "따르", "결과", "귀결", "결정"
]

CAUSAL_PATTERNS = []

# 접두사+패턴
for phrase in causal_phrases:
    for pre in prefixes:
        pattern = rf"(?:^|\s){pre}{phrase}(?:\s|$)"
        CAUSAL_PATTERNS.append(pattern)

# 단어 그대로 패턴
for kw in causal_keywords:
    pattern = rf"(?:^|\s){kw}(?:\s|$)"
    CAUSAL_PATTERNS.append(pattern)