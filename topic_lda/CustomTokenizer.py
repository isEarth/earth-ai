from kiwipiepy import Kiwi

### 문장에서 명사만 남기는 토크나이저
class CustomTokenizer:
    """
    문장에서 일반명사(NNG) 및 고유명사(NNP)만 추출하는 사용자 정의 토크나이저.

    Kiwi 형태소 분석기를 사용하여 의미 있는 명사만 필터링합니다.
    - 2자 이상 단어만 포함
    - 불용어/형용사/부사 등은 제외

    Example:
        tokenizer = CustomTokenizer()
        tokens = tokenizer("한국은행이 기준금리를 인상하였다.")
        print(tokens)  # 출력 예: ['한국은행', '기준금리']
    """
    def __init__(self):
        self.tagger = Kiwi()

    def __call__(self, sent):
        """
        주어진 문장에서 명사만 추출

        Args:
            sent (str): 입력 문장

        Returns:
            List[str]: 명사 토큰 리스트
        """
        morphs = self.tagger.analyze(sent)[0][0]
        result = [form for form, tag, _, _ in morphs if tag in ['NNG', 'NNP'] and len(form) > 1]
        return result