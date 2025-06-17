import os
import re
import json
from tqdm import tqdm
from kiwipiepy import Kiwi

kiwi = Kiwi(typos='basic_with_continual')

# 파일을 열고, 오타, 띄어쓰기, 문장분리
def open_and_preprocess(dir:str, save_file :str, min_length :int=30):

    def open_dir(dir):
        files = [f for f in os.listdir(dir) if f.startswith("file_") and f.endswith(".txt")]
        if not files:
            raise FileNotFoundError("파일이 없습니다.")
        files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return files

    def open_json(path):
        with open(path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        return data

    def sanitize_text(text):
        pattern = r'(\xa0|\n|\u200b)'
        text = re.sub(pattern, ' ',text)
        pattern = r'\s+'
        text = re.sub(pattern, ' ',text)
        text = kiwi.space(text)
        text = text.strip()
        return text
    
    def split_again(text :str) -> list[str]:
        sentences = []
        cursor = 0
        for match in re.compile(r'[^ \n]+?(?:요|죠|까|구나)([.!?\s])').finditer(text):
            end_pos = match.end()
            last_eojeol = match.group(0).strip()  # 전체 어절 기반 분석
            last = kiwi.tokenize(last_eojeol)[-1]
            if (last.tag in ['JX', 'EF']) or last.form == 'ᆸ니까':
                sentences.append(text[cursor:end_pos].strip())
                cursor = end_pos
        if cursor < len(text):
            sentences.append(text[cursor:].strip())
        return sentences

    if  '.' in dir and dir.split('.')[-1] in ['txt','csv','json','db']:
        # 이미 나눠져 있지만, 전처리 필요함.
        if dir.endswith('.json'):
            files = open_json(dir)
            merged = []
            print("Preprocessing : ")
            for file in tqdm(files):
                new_file = []
                for sent in file:
                    split = [s.text for s in kiwi.split_into_sents(sent)]
                    for one_sent in split:
                        text = sanitize_text(one_sent)
                        new_file.extend([text] if len(text) < min_length else split_again(text))
                for sent in new_file[:]:
                    if len(sent.split())<3:
                        new_file.remove(sent)
                merged.append(new_file)
            
    else:
        files = open_dir(dir)
        merged = []
        print("Preprocessing : ")
        for file in tqdm(files):
            video_path = os.path.join(dir, file)
            with open(video_path, 'r', encoding='utf-8-sig') as f:
                raw = f.read().splitlines()
            video_code = raw[0]
            content = raw[1]
            sentences_raw = kiwi.split_into_sents(content, return_tokens=False, return_sub_sents=True)
            sentences = []
            for sent in sentences_raw:
                text = sanitize_text(sent.text)
                sentences.extend([text] if len(text) < min_length else split_again(text))
            sentences = [kiwi.join(kiwi.tokenize(s)) for s in sentences]
            merged.append(sentences)
    
    with open(save_file, "w", encoding="utf-8-sig") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
        
    return merged

# 경제 용어 탐지해서 필터링 
def select_terms(videos : list[list[str]], save_file : str, colored = False):
    # === 경제 용어 사전 불러오기 ===
    with open("../data/first.txt", "r", encoding="utf-8") as f:
        econ_terms = sorted(set(line.strip() for line in f if line.strip()))

    print(f"📌 경제 용어 수: {len(econ_terms)}")

    color  = ('\033[95m','\033[0m')

    number=0
    filtered_sentences = []
    print("finding terms in sentences: ")
    for split_sents in tqdm(videos):
        temp = []
        for sentence in split_sents:
            trig = 0
            for term in econ_terms:
                if term in sentence:
                    if colored:
                        sentence = re.sub(term, color[0]+term+color[1], sentence)
                    if trig == 0:
                        temp.append(sentence)
                    trig +=1
        filtered_sentences.append(temp)
        number+=len(temp)

    print(f"✅ 경제용어 포함 문장 수: {number}")

    with open(save_file, "w", encoding="utf-8-sig") as f:
        json.dump(filtered_sentences, f, ensure_ascii=False, indent=2)
    return filtered_sentences