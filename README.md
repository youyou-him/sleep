# AI 기반 수면 건강 코치 (AI-based Sleep Health Coach)

**웨어러블 데이터와 AI를 통한 개인 맞춤형 수면 건강 관리 및 사고 예방 솔루션**

* **프로젝트 기간:** 2025.06.18 ~ 2025.06.25
* **프로젝트 목표:** 사용자 데이터를 기반으로 수면의 질을 예측하고, 수면 영향 요인을 식별하여 개인 맞춤형 개선 가이드 및 모니터링 기능 제공. '데이터를 가치있는 정보로' 만듦.

---

## 1. 프로젝트 배경 (Problem Definition)

* **사회적 문제:** 2021년 OECD 조사 결과, 한국인의 하루 평균 수면 시간은 7시간 51분으로 회원국 평균(8시간 27분) 대비 최하위 수준.
* **관심 증가:** 수면 부족은 졸음운전 사고 등 다양한 사회적 문제로 이어짐. '슬리포노믹스(Sleeponomics)' 및 '갤럭시 링' 같은 건강 측정 기기에 대한 관심이 높아짐.
* **프로젝트의 필요성:** 수면 부족과 피로로 인한 사고 위험을 감지하고, 개인 맞춤형 수면 개선 가이드를 제공할 필요.

---

## 2. 핵심 기능 (Core Features)

1.  **수면의 질 예측 (Classification)**
    * 사용자가 입력한 건강 데이터(나이, 직업, BMI, 혈압, 스트레스 등)를 기반으로 **LightGBM 모델**을 사용하여 수면의 질 점수 예측.
2.  **사용자 유형 분류 (Clustering)**
    * **K-Means 군집 분석**을 통해 사용자를 4가지 고유 유형('에너지 넘치는 건강인', '균형 추구형', '건강 취약형', '번아웃 경계형')으로 분류.
3.  **개인 맞춤형 리포트 (Web Service)**
    * **Flask** 기반 웹 서비스를 통해 '나의 일일 수면 리포트' 제공.
    * 사용자가 직접 데이터를 입력하거나 기존 사용자 데이터를 선택해 분석 요청 가능.
4.  **맞춤형 솔루션 제공 (Customized Guide)**
    * 분석된 군집 유형과 예측된 수면 점수를 바탕으로, **개인화된 건강 및 수면 개선 가이드**를 텍스트로 제공.

---

## 3. 기술 스택 (Tech Stack)

| 구분 | 기술 |
| :--- | :--- |
| **Language** | Python, JavaScript (JS), HTML, CSS |
| **Web Framework & API** | **Flask** |
| **Data Handling & ML** | Scikit-learn (`learn`), Imbalanced-learn (`imbalanced<br>learn`), Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **IDE** | Jupyter, VS Code |

---

## 4. 분석 프로세스 (Analysis Process)

1.  **데이터 수집 (Data Collection)**
    * Kaggle "Sleep Health Data" (374x13)
    * 국민건강영양조사(KNHANES) (8088x799)
2.  **전처리 및 가공 (Preprocessing & Processing)**
    * 데이터 병합 (최종 7303x10), 결측치 처리, 그룹핑
    * *피처 엔지니어링*: 평균 혈압(MAP) 계산, (주중/주말) 수면시간 계산
    * *인코딩*: 범주형 변수(직업, 성별 등)에 LabelEncoder / One-Hot Encoding 적용
    * *스케일링*: 수치형 데이터에 StandardScaler 표준화
    * *불균형 처리*: **SMOTE** 오버샘플링 기법 사용
3.  **EDA (Exploratory Data Analysis)**
    * 변수별 히스토그램 작성, 상관관계 히트맵 분석
4.  **모델링 (Modeling)**
    * **분류 (Classification)**:
        * *모델*: Random Forest, **LightGBM**
        * *전략*: SMOTE 및 Stratified K-Fold(k=5) 교차 검증
        * *최종 선택*: **LightGBM** (Accuracy: 0.74, Macro F1: 0.74)
    * **군집 (Clustering)**:
        * *모델*: **K-Means**
        * *최적 K 탐색*: Elbow Method 및 Silhouette Score (K=3~4 추천)
        * *결과*: 군집별 특성(직업, 성별, BMI 등) 비율 분석
5.  **웹 서비스 구현 (FLASK Webpage Production)**
    * 사용자 입력값을 받아 실시간 전처리 후 모델 적용
    * 분석 결과를 **JSON** 형식으로 **REST** 기반 비동기 통신
6.  **맞춤 리포트 및 가이드 제공 (User-customized Report & Guide)**
    * JSON 데이터를 받아 웹페이지에 동적 시각화

---

## 5. 향후 계획 (Future Work)

* **모바일 앱 개발**을 통한 접근성 확대
* **실시간 웨어러블 데이터** (스마트 워치, 링 등) 연동
* **딥러닝(Deep Learning)**을 통한 모델 고도화
* 다양한 **수면 장애 유형** (불면증, 무호흡증 등) 예측 및 진단 기능 추가

---

## 6. 팀원 및 역할

* **박예슬**
    * 데이터 수집 & 전처리
    * 분류 및 전체적 모델링
    * PPT 제작 및 발표
* **임규리**
    * 데이터 수집
    * 데이터 전처리 및 가공
    * 군집 분석 모델링
    * 웹페이지 시각화(REST) 구현
