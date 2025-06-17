"""
causal_patterns.py

인과관계 문장 분류를 위한 정규표현식 패턴 정의 모듈입니다.

이 모듈은 한국어 문장에서 인과관계를 표현하는 접두사 기반 어절 및
단어 패턴을 정의하여 리스트 `CAUSAL_PATTERNS`에 담습니다.

외부에서는 `from causal_patterns import CAUSAL_PATTERNS` 형태로
해당 패턴 리스트를 불러와 정규식 매칭에 사용할 수 있습니다.

예시:
    import re
    from causal_patterns import CAUSAL_PATTERNS

    text = "그러므로 우리는 준비가 필요하다."
    for pattern in CAUSAL_PATTERNS:
        if re.search(pattern, text):
            print("인과 패턴 포함")
            break
"""

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