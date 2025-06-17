# 전체 Triplet 추출 및 후처리 파이프라인
# === 목적 ===
# 이 코드는 문장을 절(clause) 단위로 분리한 후, 의미적 유사도 기반으로 관계(triplet)를 추출하고,
# 유사한 절들을 병합하며 중복 제거 및 GNN 학습 데이터로 정제하는 end-to-end 파이프라인입니다.
# 
# === 주요 컴포넌트 ===
# - Triplets: 절 간 임베딩 유사도 계산 및 관계 추론 (유사/반대)
# - AfterProcess: 유사 절 병합, triplet ID 정규화, 중복 제거, 최종 저장
# - ClauseDB: 절과 임베딩 벡터를 저장하는 SQLite + numpy 기반 DB
# - prepare_gnn: 인과 중심의 triplet을 edge 배열로 변환하여 GNN 학습에 적합한 포맷으로 저장
# 
# === 전체 실행 흐름 ===
# 1. json에서 절 정보 로딩 및 무효/공백 절 전처리
# 2. ClauseDB에 임베딩 저장 (기존 파일이 없을 경우 SBERT로 재생성)
# 3. cosine similarity 및 L2 distance 기반으로 유사한 절 쌍 추출
# 4. Triplets 클래스에서 유사/반대 관계로 태깅 및 저장
# 5. AfterProcess 클래스에서
#    - 삭제 대상 및 유사 ID 제거
#    - 클러스터링 기반 병합 및 임베딩 평균화
#    - triplet 내 ID 치환 및 정제
#    - 텍스트 기준 중복 제거
#    - 새로운 triplet 저장
# 6. prepare_gnn 함수에서 인과/상황/역인과 관계만 edge로 추출해 GNN 입력으로 저장
# 7. 최종적으로 정제된 triplet 개수와 관계 분포를 출력하여 확인



from dataclasses import dataclass
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kiwipiepy import Kiwi
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from train import Config
from prediction import ClauseSpliting, ClauseDB, FileNames, get_shape

filenames_db = FileNames()
@dataclass
class FilePaths:
    name: str = "_top6"
    db_dir: str = "./user_data/"                                  # DB 및 결과 저장 디렉토리
    db_path: str = f"{db_dir}user_clauses{name}.db"              # 절 텍스트 DB (SQLite)
    embedding_path: str = f"{db_dir}user_embeddings{name}.npy"   # 절 임베딩 저장 파일 (npy, [V][S][C][768])
    relation_np :str = f"{db_dir}relation_sbert{name}.npy"        # 관계 예측 결과 파일 (임시 저장)
    similar_np :str = f"{db_dir}similar{name}.npy"                # 유사한 절 ID 쌍 리스트 (List[Tuple[str, str]])
    opposite_np :str = f"{db_dir}opposite{name}.npy"              # 반대 의미 절 ID 쌍 리스트 (옵션)
    similar_temp_np :str = f"{db_dir}similar_temp{name}.npy"      # 유사쌍 중간 저장 파일 (옵션)
    new_triplet_np : str = f"{db_dir}new_triplet{name}.npy"       # 후처리된 최종 triplet (np.ndarray[(id1, id2, rel_id)])
    final_relation_triplets_np : str = f"{db_dir}final_relation_triplets{name}.npy"  # GNN용 edge (원인-결과)
    no_duplicated_triplets_np : str = f"{db_dir}no_duplicated_triplets{name}.npy"    # 텍스트 기준 중복 제거된 triplet
    similar_cluster_np: str = f"{db_dir}similar_cluster{name}.npy"  # 유사 절 클러스터링 결과 캐시

    temp_dir : str = f"./saved_temp/"                             # 임시 중간 저장 폴더

    saved_triplets_np: str = ""                                   # 초기 triplet 저장 파일
    def __post_init__(self):
        global filenames_db
        self.saved_triplets_np = filenames_db.triplets_np.replace("./", "../youtube_clause/")

filepaths = FilePaths()
config = Config()



class Triplets():
    """
    Triplets 클래스는 입력 절(clause)에 대해 다음 작업을 수행합니다:
    - 무효 절 필터링
    - 임베딩 생성 및 DB 저장
    - 절 간 유사도 계산 및 관계 분류 ('유사', '반대', '무관')

    Attributes:
        deleted (List[str]): 제거된 clause ID 목록
        similar_twins (List[Tuple[str, str]]): 유사 관계로 분류된 ID 쌍
        opposite_twins (List[Tuple[str, str]]): 반대 관계로 분류된 ID 쌍
        filepaths (FilePaths): 관련 경로 정보를 담은 객체
    """
    def __init__(self):
        self.deleted = []
        self.similar_twins = []
        self.opposite_twins = []
        self.filepaths = FilePaths()

    def preprocessing(self,clauses: dict):
        """
        의미 없는 절을 제거합니다.

        Args:
            clauses (dict): {id: clause_text} 형식의 딕셔너리

        Returns:
            dict: 필터링된 clause 딕셔너리
        """
        to_delete = []
        for cid, text in clauses.items():
            if not isinstance(text, str) or not text or text.isspace() or text == "입력하신 문장을 제공해 주세요.":
                to_delete.append(cid)
        
        for cid in to_delete:
            del clauses[cid]
        
        self.deleted = to_delete
        return clauses

    @staticmethod
    def infer_relation(sim: float, dist: float, sim_thresh=0.9, dist_thresh=0.1):
        """
        cosine similarity와 L2 distance를 기반으로 관계를 분류합니다.

        Args:
            sim (float): cosine similarity (-1 ~ 1 정규화됨)
            dist (float): L2 거리
            sim_thresh (float): 유사 기준 threshold
            dist_thresh (float): 반대 기준 threshold

        Returns:
            str: '유사', '반대', 또는 '무관'
        """
        if sim > sim_thresh:
            return "유사"  # 의미 유사
        elif sim < 0.3 and dist < dist_thresh:
            return "반대"  # 단어가 비슷한데, 의미가 다르면 → 반대
        else:
            return "무관"  # 의미도 다르고 단어도 다름

    def infer_relation_pair(self, pair_list, clauses_dict, sim_thresh=0.97, dist_thresh=0.6, print_rel = False):
        """
        저장된 유사도 결과(pair_list)를 기반으로 '유사', '반대' 관계를 추론.
        
        Args:
            pair_list (List[Tuple[str, str, float, float]]): (id1, id2, cos_sim, l2_dist)
            clauses_dict (dict): {id: clause_text}
            sim_thresh (float): 유사 기준 threshold
            dist_thresh (float): 반대 기준 threshold
            print_rel (bool): 관계 출력 여부

        Returns:
            Saves:
                - filepaths.similar_np
                - filepaths.opposite_np
            Sets:
                - self.similar_twins
                - self.opposite_twins
        """
        similar_twins = []
        opposite_twins = []

        for id1, id2, cos_sim, l2_dist in tqdm(pair_list, desc="관계 추론 중"):
            norm_sim = (cos_sim + 1) / 2             # [-1, 1] → [0, 1]
            norm_dist = 1 - (1 / (1 + l2_dist))       # [0, ∞) → [0, 1)

            rel = Triplets.infer_relation(norm_sim, norm_dist, sim_thresh, dist_thresh)

            if clauses_dict.get(id1, '').strip() == '':
                continue

            if rel in ["유사", "반대"] and print_rel:
                print(f"[{rel}] {clauses_dict[id1]}  ///  {clauses_dict[id2]}  (sim: {cos_sim:.3f})")

            if rel == '유사':
                similar_twins.append((id1, id2))
            elif rel == '반대':
                opposite_twins.append((id1, id2))

        # 결과 저장 및 전역 구조 업데이트
        self.similar_twins = similar_twins
        self.opposite_twins = opposite_twins

        np.save(filepaths.similar_np, np.array(similar_twins))
        np.save(filepaths.opposite_np, np.array(opposite_twins))

    def set_embedding(self, db, clauses, batch_size:int= 1000):
        """
        절 ID와 텍스트를 SBERT를 통해 임베딩하고 DB에 저장합니다.

        Args:
            db (ClauseDB): 절과 임베딩을 저장하는 DB 인스턴스
            clauses (List[Tuple[str, str]]): [(id, clause)] 리스트
            batch_size (int): DB 저장 배치 크기

        Returns:
            None
        """
        # sbert 처리된 embedding 사용 
        print("clause shape : ",get_shape(clauses))

        video = []
        for clause_id, clause in clauses:
            clause_id = int(clause_id)
            V = clause_id // 100000
            S = (clause_id % 100000) // 10
            C = clause_id % 10
            while len(video) <= V: # V 차원 확장
                video.append([])
            while len(video[V]) <= S: # S 차원 확장
                video[V].append([])
            while len(video[V][S]) <= C: # C 차원 확장
                video[V][S].append(None)
            video[V][S][C] = clause
        print("clause number  :",len([clause for video_unit in video for sentence in video_unit for clause in sentence]))
        
        cs = ClauseSpliting(config=config, filenames=filenames_db, reference_mode=True)

        cs.clause_embedding(video, highlight=False, sbert_option = True)

        for batch in (clauses[i:i+batch_size] for i in range(0, len(clauses), batch_size)):
            db.insert_batch(batch)
        print(len(db.get_all_embedding()), len(db.get_all_clauses(return_format="clauses")))

    def is_same_pair(self, db, clause_ids, threshold=0.9, max_pair=10000, device='cuda'):
        """
        임베딩 벡터 간 cosine similarity 및 L2 distance를 계산하여 유사쌍 후보를 추출합니다.

        Args:
            db (ClauseDB): 임베딩 정보를 포함한 DB
            clause_ids (List[str]): clause ID 리스트
            threshold (float): 유사 기준 threshold (cosine)
            max_pair (int): 최대 유사쌍 개수
            device (str): 'cuda' 또는 'cpu'

        Returns:
            List[Tuple[str, str, float, float]]: 유사쌍과 거리 정보
        """
        proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64)
        ).to(device)
        batch_size = 5000
        
        raw_list = db.get_all_embedding(return_id = True)[:max_pair] # sampling
        _clause_ids, emb_list = zip(*raw_list)
        dif = list(set(clause_ids) - set(_clause_ids))
        for dif_id in dif:
            print(dif_id,db.get_clause(dif_id))

        print("clause_ids, emb_list are ready!")
        print("db_clause : ",db.count_clauses())
        print("raw_list : ",len(raw_list))

        emb_list = [torch.tensor(emb, dtype=torch.float32) for emb in emb_list]
        embeddings = torch.stack(emb_list).to("cuda")

        print("Start fast_similarity : ",len(embeddings))
        N = len(embeddings)
        with tqdm(total=N*(N-1)//2, desc="cosine 추출") as pbar:
            with torch.no_grad():  # 추론용
                projected = proj(embeddings)  # [N, 64]

                #  cosine similarity matrix 계산
                normed = F.normalize(projected, dim=1)  # [N, D]
                fast_similarty = []
                for i in range(0, N, batch_size):
                    end_i = min(i + batch_size, N)
                    batch_i = normed[i:end_i]

                    for j in range(i, N, batch_size):
                        if j < i:
                            continue
                        end_j = min(j + batch_size, N)
                        batch_j = normed[j:end_j]

                        sim_block = torch.matmul(batch_i, batch_j.T)

                        if i == j:
                            for ii in range(end_i - i):
                                for jj in range(ii + 1, end_j - j):  # 상삼각만
                                    global_i = i + ii
                                    global_j = j + jj
                                    sim = sim_block[ii, jj].item()
                                    if sim > threshold:
                                        fast_similarty.append((global_i, global_j))
                                    pbar.update(1)
                        else:
                            for ii in range(end_i - i):
                                for jj in range(end_j - j):
                                    global_i = i + ii
                                    global_j = j + jj
                                    sim = sim_block[ii, jj].item()
                                    if sim > threshold:
                                        fast_similarty.append((global_i, global_j))
                                    pbar.update(1)

        final_results = []
        batch_size = 10000

        for k in tqdm(range(0, len(fast_similarty), batch_size), desc="정확 유사도 계산"):
            batch = fast_similarty[k:k+batch_size]

            # 배치 단위로 텐서 구성
            vec1_batch = torch.stack([embeddings[i] for i, j in batch])
            vec2_batch = torch.stack([embeddings[j] for i, j in batch])

            # GPU 연산
            cos_sim = F.cosine_similarity(vec1_batch, vec2_batch, dim=1)  # [B]
            eucl_dist = F.pairwise_distance(vec1_batch, vec2_batch, p=2)  # [B]

            # 결과 결합
            for n in range(len(batch)):
                i, j = batch[n]
                final_results.append((_clause_ids[i], _clause_ids[j], cos_sim[n].item(), eucl_dist[n].item()))

        np.save(filepaths.similar_temp_np, np.array(final_results, dtype=object))
        print(f"[완료] 저장 경로: {filepaths.similar_temp_np} / 총 유사 쌍 수: {len(final_results)}")
        return final_results

    def group_connected(self, edges=None):
        """
        유사한 ID들 간의 연결 관계를 기반으로 연결된 컴포넌트(그룹)를 추출합니다.
        Union-Find(Disjoint Set) 알고리즘을 사용하여, 서로 직접 혹은 간접적으로 연결된 ID들을 하나의 그룹으로 묶습니다.

        Args:
            edges (List[Tuple[str, str]], optional): 연결 관계를 나타내는 쌍의 리스트.
                예: [("A", "B"), ("B", "C")] → "A", "B", "C"는 하나의 그룹
                기본값은 self.similar_twins

        Returns:
            List[Set[str]]: 연결된 그룹들의 리스트. 각 그룹은 set 형태로 ID를 포함.
                예: [{"A", "B", "C"}, {"D", "E"}]
        """
        if not edges:
            edges = self.similar_twins
        parent = {}

        def find(x):
            # 경로 압축
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root != y_root:
                parent[y_root] = x_root

        # 초기 parent 설정
        nodes = set([n for pair in edges for n in pair])
        for node in nodes:
            parent[node] = node

        # 연결 정보 반영
        for a, b in edges:
            union(a, b)

        # 그룹화
        groups = defaultdict(set)
        for node in nodes:
            root = find(node)
            groups[root].add(node)

        return list(groups.values())


def delete_all_created_files():
    """
    미리 정의된 임시 파일들을 모두 삭제합니다.
    외부 인수 없이 호출만으로 삭제됩니다.
    """
    paths_to_delete = [
        filepaths.db_path,
        filenames_db.sbert_np,
        filepaths.similar_np,
        filepaths.opposite_np,
        filepaths.similar_temp_np,
        filepaths.final_relation_triplets_np,
        filepaths.no_duplicated_triplets_np,
    ]

    for path in paths_to_delete:
        if os.path.exists(path):
            os.remove(path)
            print(f"✅ 삭제됨: {path}")
        else:
            print(f"⚠️ 존재하지 않음: {path}")

def concat_saved_batches(save_dir, output_path='final_results.npy'):
    """
    저장된 pair_batch_*.npy 파일들을 병합하여 하나의 NPY로 저장합니다.

    Args:
        save_dir (str): 저장된 배치 파일 경로
        output_path (str): 병합 후 저장 경로

    Returns:
        np.ndarray: 병합된 (id1, id2, cos_sim, l2_dist) 배열
    """
    all_data = []
    files = sorted(f for f in os.listdir(save_dir) if f.startswith("pair_batch_") and f.endswith(".npy"))
    for fname in tqdm(files, desc="병합 중"):
        batch = np.load(os.path.join(save_dir, fname), allow_pickle=True)
        all_data.extend(batch)

    final = np.array(all_data, dtype=object)
    np.save(output_path, final)
    print(f"최종 병합 결과 저장 완료: {output_path}")
    return final

class AfterProcess():
    """
    AfterProcess 클래스는 triplet 정보를 후처리합니다:
    - 유사/삭제된 ID 제거
    - 임베딩 병합 및 triplet ID 통일
    - 클러스터링 기반 병합 (대형 유사 그룹)
    - 중복 제거 및 저장

    Attributes:
        tls (Triplets): 공유된 Triplets 인스턴스 (deleted, similar_twins 참조)
        triplets_np (np.ndarray): (id1, id2, rel_id) triplet 리스트
        embeddings (Dict[str, np.ndarray]): ID별 임베딩 벡터
        clause_dict (Dict[str, str]): ID별 절 텍스트
    """
    def __init__(self, tls:Triplets, triplet_file, clause_dict):
        """
        Args:
            tls (Triplets): Triplets 클래스 인스턴스 (deleted/similar_twins 참조용)
            triplet_file (str): 원본 triplet npy 파일 경로
            clause_dict (dict): {id: 절 텍스트} 매핑 딕셔너리
        """
        self.tls = tls
        self.triplets_np = np.load(triplet_file, allow_pickle=True)
        self.new_triplet_file = filepaths.new_triplet_np
        self.shrinked = []
        self.similar_data = np.load(filepaths.similar_np, allow_pickle=True)
        self.db = ClauseDB(filepaths.db_path, filenames_db.sbert_np)
        print("DB 꺼내기 시작.")
        self.embeddings = self.db.get_all_embedding(return_dict=True)
        self.clause_dict = clause_dict

        # 군집 기반 병합 수행
        # 클러스터 기반 병합 수행
        clusters = self.cluster_large_group(threshold=20)
        for member_ids in clusters.values():
            base_id = member_ids[0]
            self.merge_cluster_embeddings(base_id, member_ids)
        self.after_process()
        self.db.close()

    def cluster_large_group(self, threshold=1000):
        """
        TF-IDF + KMeans로 유사 절 그룹을 클러스터링하고 결과를 캐싱.

        Args:
            threshold (int): 클러스터링을 수행할 최소 절 수 기준

        Returns:
            dict: label → [절 ID 목록] 형태의 클러스터링 결과
        """
        if os.path.exists(filepaths.similar_cluster_np):
            print("🔄 캐시된 클러스터 결과 로딩 중...")
            return np.load(filepaths.similar_cluster_np, allow_pickle=True).item()  # dict 형태

        id_texts = [(id1, self.clause_dict[id1]) for id1, _ in self.similar_data]
        id_texts = list({id: text for id, text in id_texts}.items())  # 중복 제거

        if len(id_texts) < threshold:
            return {i: [id] for i, (id, _) in enumerate(id_texts)}  # 그대로 반환

        ids, texts = zip(*id_texts)
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(texts)

        k = int(len(texts) / 100) + 1  # 대략 100개 단위 클러스터
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(tfidf)

        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(ids[idx])

        np.save(filepaths.similar_cluster_np, dict(clusters))  # 캐싱 저장
        print(f"✅ 클러스터링 결과 캐시 저장: {filepaths.similar_cluster_np}")
        return clusters

    def merge_cluster_embeddings(self, base_id, member_ids):
        """ 
        클러스터 내 임베딩 병합 및 triplet ID 일괄 치환.

        Args:
            base_id (str): 기준 ID (대표 ID)
            member_ids (List[str]): 클러스터 내 포함된 ID 리스트
        """
        base_emb = self.embeddings[base_id]
        for mid in member_ids:
            if mid == base_id:
                continue
            if mid not in self.embeddings:
                continue
            base_emb = (base_emb + self.embeddings[mid]) / 2
            self.update_triplets_with_clusters(mid, base_id)
            self.shrinked.append(mid)
        self.db.update_embedding(base_id, base_emb)

    def update_triplets_with_clusters(self, old_id, new_id):
        """
        triplet 내 ID를 군집 기반으로 업데이트.

        Args:
            old_id (str): 병합 대상 ID
            new_id (str): 기준 ID (병합 후 대체 ID)
        """
        for i in range(len(self.triplets_np)):
            h, t, r = self.triplets_np[i]
            if h == old_id:
                h = new_id
            if t == old_id:
                t = new_id
            self.triplets_np[i] = (h, t, r)

    def after_process(self):
        """
        Triplet 리스트에서 삭제/유사 ID를 제거하고, 중복 관계를 필터링하여 정제된 triplet을 저장합니다.
        이후 텍스트 기준 중복도 제거하고 요약 출력까지 수행합니다.
        """
        new_triplets = []
        for triplet in self.triplets_np:
            A, B, R = triplet
            if A in self.tls.deleted or B in self.tls.deleted:
                continue
            elif (A, B) in self.tls.similar_twins or (B, A) in self.tls.similar_twins:
                continue
            else:
                new_triplets.append(triplet)
        self.triplets_np = new_triplets

        dedup = []
        seen = set()
        for triplet in self.triplets_np:
            id1, id2, rel = triplet
            key = (id1, id2)
            if key not in seen:
                seen.add(key)
                dedup.append(triplet)
        self.triplets_np = dedup
        np.save(self.new_triplet_file, np.array(self.triplets_np, dtype=object))
        print(f"최종 triplet 수: {len(self.triplets_np)}개")
        self.earse_duplication_n_print(self.new_triplet_file, self.clause_dict, self.shrinked)

    def earse_duplication_n_print(self, triplet_file, clause_dict, shrinked=None, number = 100):
        """
        텍스트 기준 중복 triplet 제거 후, 일부 샘플 출력 및 저장.

        Args:
            triplet_file (str): triplet npy 파일 경로
            clause_dict (dict): {id: 절 텍스트} 매핑
            shrinked (List[str], optional): 병합된 ID 리스트 (사용 X)
            number (int): 출력할 triplet 샘플 수
        """
        rel_map = {'없음': 0, '기타': 1, '대조/병렬': 2, '상황': 3, '수단': 4, '역인과': 5, '예시': 6, '인과': 7}
        rev_map = {v: k for k, v in rel_map.items()}
        triplets = np.load(triplet_file,allow_pickle=True)
        i=0
        no_duplicated = []
        seen = set()
        for id1, id2, rel_id in triplets:
            clause1 = clause_dict[int(id1)]
            clause2 = clause_dict[int(id2)] 
            if (clause1,clause2) in seen:
                continue
            seen.add((clause1,clause2))
            no_duplicated.append((id1,id2,rel_id))
            if rel_id == 7:
                color = ('\033[95m', '\033[0m')
            elif rel_id == 0:
                color = ('\033[94m', '\033[0m')
            else:
                color = ('\033[93m', '\033[0m')
            rel = rev_map.get(rel_id, '없음')
            if clause1 and clause2:
                print(f"{clause1}  -({color[0]}{rel}{color[1]})->  {clause2}")
            else:
                print(f"Invalid triplet: ({id1}:{clause1}, {id2}:{clause2}, {rel_id}:{rel})")
            if i >= number:
                break
            i += 1
        np.save(filepaths.no_duplicated_triplets_np,np.array(no_duplicated, dtype=object))


def prepare_gnn(triplets, save_file):
    """
    triplet 관계를 GNN 학습을 위한 edge array로 변환하여 저장합니다.

    Args:
        triplets (np.ndarray): shape (N, 3) - (id1, id2, rel_id)
        save_file (str): 저장 경로 (.npy)

    Returns:
        None
    """
    before_node = []
    after_node = []
    relation = []
    for id1,id2,rel in triplets:
        if rel == 0: # 관계없음
            continue
        elif rel == 5: # 역인과
            before_node.append(id2)
            after_node.append(id1)
            relation.append(1)
        elif rel in [3,4,7]: # 인과, 상황, 수단
            before_node.append(id1)
            after_node.append(id2)
            relation.append(1)
        else: # 기타 관계 있음
            before_node.append(id1)
            after_node.append(id2)
            relation.append(0)

    edge_array = np.stack([before_node, after_node, relation], axis=0)
    np.save(save_file, edge_array)
        
def solve_duplication_triplets(triplet_file):
    """
    triplet 파일에서 (id1, id2) 기준 중복 관계를 제거합니다.

    Args:
        triplet_file (str): triplet .npy 파일 경로

    Returns:
        None
    """
    # 파일 로드
    triplets = np.load(triplet_file, allow_pickle=True)

    # 중복 제거: (id1, id2) 기준으로만 유일하게 남기기
    seen = set()
    dedup_triplets = []
    for t in triplets:
        id1, id2, rel_id = t
        key = (id1, id2)
        if key not in seen:
            seen.add(key)
            dedup_triplets.append((id1, id2, rel_id))

    # 저장
    np.save(triplet_file, np.array(dedup_triplets, dtype=object))
    print(f"중복 제거 완료: {len(triplets)} → {len(dedup_triplets)}개로 줄었습니다.")

def check_triplets(triplet_file):
    """
    Triplet 파일을 로드하여 개수, 관계 분포, 샘플을 출력합니다.

    Args:
        triplet_file (str): npy 형식의 triplet 저장 파일 경로
    """

    triplets = np.load(triplet_file,allow_pickle=True)
    print("Triplets number :", len(triplets))

    # 고유 관계 타입 개수
    rel_types = {}
    for _, _, r in triplets:
        rel_types[r] = rel_types.get(r, 0) + 1
    print("📊 관계 분포:")
    for rel_id, count in sorted(rel_types.items()):
        print(f"  - rel_id={rel_id}: {count}개")

    # 샘플 몇 개 출력
    print("\n🔍 Triplet 샘플:")
    for i, (h, t, r) in enumerate(triplets[:2]):
        print(f"  {i+1}. ({h}) --[{r}]--> ({t})")



def main():
    #  기존 triplet 결과 간단 확인 (캐싱 여부 확인용)
    check_triplets(filepaths.saved_triplets_np)

    #  기존 파일 제거 (중간 결과들 초기화)
    json_path = 'clause_gpt_top6.json'
    delete_all_created_files()

    #  1. 절 로딩 및 전처리
    with open(json_path, 'r', encoding='utf-8-sig') as f:
        clauses = json.load(f)
    triplets = Triplets()
    _clauses = triplets.preprocessing(clauses)
    _clauses_list = [(id, clause) for id, clause in _clauses.items()]

    print("we are watching", filepaths.db_path, filenames_db.sbert_np)

    #  2. Clause DB 및 임베딩 처리
    with ClauseDB(filepaths.db_path, filenames_db.sbert_np) as db:
        # 임베딩 파일이 없으면 새로 생성
        if not os.path.exists(filenames_db.sbert_np):
            triplets.set_embedding(db, _clauses_list)

        #  3. DB로부터 절 정보 로딩
        clauses_dict = db.get_all_clauses(return_format='clauses', return_id=True)
        clause_ids = list(clauses_dict.keys())
        print("clauses_dict loaded!", len(clause_ids))

        #  4. 유사쌍 추출 및 관계 예측
        print("Pairing Started!!")
        if not os.path.exists(filepaths.similar_np):
            pair_list = triplets.is_same_pair(db, clause_ids)
            triplets.infer_relation_pair(pair_list, clauses_dict)
        print("----------------")

    #  5. Triplet 후처리 (중복/병합/군집화 포함)
    triplet_file = filepaths.saved_triplets_np
    after = AfterProcess(tls=triplets, triplet_file=triplet_file, clause_dict=clauses_dict)

    #  6. GNN 학습용 edge 저장
    prepare_gnn(np.load(filepaths.new_triplet_np, allow_pickle=True), filepaths.final_relation_triplets_np)

    #  7. 최종 Triplet 결과 확인
    check_triplets(filepaths.new_triplet_np)
    check_triplets(filepaths.no_duplicated_triplets_np)
    print("<<<THE END>>>")

if __name__ == "__main__":
    main()
