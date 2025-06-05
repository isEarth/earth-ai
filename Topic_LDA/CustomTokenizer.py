from kiwipiepy import Kiwi

### 문장에서 명사만 남기는 토크나이저
class CustomTokenizer:
    def __init__(self):
        self.tagger = Kiwi()

    def __call__(self, sent):
        morphs = self.tagger.analyze(sent)[0][0]
        result = [form for form, tag, _, _ in morphs if tag in ['NNG', 'NNP'] and len(form) > 1]
        return result