"""
    문장에서 일반명사(NNG)와 고유명사(NNP)만 추출하는 사용자 정의 토크나이저 클래스.

    특징:
        - `kiwipiepy` 기반 형태소 분석기 사용
        - 단일 문장 입력에 대해 두 글자 이상의 명사만 리스트로 반환

    Example:
        >>> tokenizer = CustomTokenizer()
        >>> tokenizer("한국은행은 기준금리를 인상했다.")
        ['한국은행', '기준', '금리']

    Methods:
        __call__(sent: str) -> List[str]: 입력 문장에서 명사만 추출해 반환
    """

from kiwipiepy import Kiwi

### 문장에서 명사만 남기는 토크나이저
class CustomTokenizer:
    def __init__(self):
        """Kiwi 형태소 분석기 초기화"""
        self.tagger = Kiwi()

    def __call__(self, sent):
        """
        입력 문장에서 일반명사(NNG), 고유명사(NNP)만 필터링하여 추출

        Args:
            sent (str): 형태소 분석할 문장

        Returns:
            List[str]: 두 글자 이상의 명사 리스트
        """
        morphs = self.tagger.analyze(sent)[0][0]
        result = [form for form, tag, _, _ in morphs if tag in ['NNG', 'NNP'] and len(form) > 1]
        return result