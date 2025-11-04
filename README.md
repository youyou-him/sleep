# AI 기반 수면 건강 코치 (AI-based Sleep Health Coach)

[cite_start]**웨어러블 데이터와 AI를 통한 개인 맞춤형 수면 건강 관리 및 사고 예방 솔루션 [cite: 2]**

* [cite_start]**프로젝트 기간:** 2025.06.18 ~ 2025.06.25 [cite: 20]
* [cite_start]**프로젝트 목표:** 사용자 데이터를 기반으로 수면의 질을 예측하고 [cite: 155][cite_start], 수면 영향 요인을 식별하여 [cite: 156] [cite_start]개인 맞춤형 개선 가이드 및 모니터링 기능을 제공함으로써 '데이터를 가치있는 정보로' 만듭니다[cite: 158, 160, 161].

---

## 1. 프로젝트 배경 (Problem Definition)

* [cite_start]**사회적 문제:** 2021년 OECD 조사 결과, 한국인의 하루 평균 수면 시간은 7시간 51분으로 회원국 평균(8시간 27분) 대비 최하위 수준입니다[cite: 71, 72, 101, 102, 132, 133, 151, 152].
* [cite_start]**관심 증가:** 수면 부족은 졸음운전 사고 [cite: 42, 105, 143, 144] [cite_start]등 다양한 사회적 문제로 이어지며, '슬리포노믹스(Sleeponomics)' [cite: 52, 70, 83, 100, 119, 131, 140, 142, 145] [cite_start]및 '갤럭시 링' 같은 건강 측정 기기 [cite: 45, 75, 108, 136]에 대한 관심이 높아지고 있습니다.
* [cite_start]**프로젝트의 필요성:** 수면 부족과 피로로 인한 사고 위험을 감지하고 [cite: 37][cite_start], 개인 맞춤형 수면 개선 가이드를 제공할 필요가 있습니다[cite: 38].

---

## 2. 핵심 기능 (Core Features)

1.  **수면의 질 예측 (Classification)**
    * [cite_start]사용자가 입력한 건강 데이터(나이, 직업, BMI, 혈압, 스트레스 등)를 기반으로 **LightGBM 모델**을 사용하여 수면의 질 점수를 예측합니다[cite: 306, 317, 576, 577].
2.  **사용자 유형 분류 (Clustering)**
    * [cite_start]**K-Means 군집 분석**을 통해 사용자를 4가지 고유 유형('에너지 넘치는 건강인', '균형 추구형', '건강 취약형', '번아웃 경계형')으로 분류합니다[cite: 352, 522, 523, 524, 525].
3.  **개인 맞춤형 리포트 (Web Service)**
    * [cite_start]**Flask** 기반 웹 서비스를 통해 '나의 일일 수면 리포트'를 제공합니다[cite: 183, 191, 526, 637].
    * [cite_start]사용자가 직접 데이터를 입력하거나 기존 사용자 데이터를 선택해 분석을 요청할 수 있습니다[cite: 531, 533, 536, 563].
4.  **맞춤형 솔루션 제공 (Customized Guide)**
    * [cite_start]분석된 군집 유형과 예측된 수면 점수를 바탕으로, **개인화된 건강 및 수면 개선 가이드**를 텍스트로 제공합니다[cite: 609, 624, 625, 626, 627, 631, 632, 633, 634].

---

## 3. 기술 스택 (Tech Stack)

| 구분 | 기술 |
| :--- | :--- |
| **Language** | [cite_start]Python, JavaScript (JS), HTML, CSS [cite: 165, 166, 168, 169] |
| **Web Framework & API** | [cite_start]**Flask** [cite: 183, 551] |
| **Data Handling & ML** | [cite_start]Scikit-learn (`learn`), Imbalanced-learn (`imbalanced<br>learn`), Joblib [cite: 177, 178, 179] |
| **Visualization** | [cite_start]Matplotlib, Seaborn [cite: 180, 181] |
| **IDE** | [cite_start]Jupyter, (VS Code 로고) [cite: 173, 174] |

---

## 4. 분석 프로세스 (Analysis Process)

1.  [cite_start]**데이터 수집 (Data Collection)** [cite: 188]
    * [cite_start]Kaggle "Sleep Health Data" (374x13) [cite: 199]
    * [cite_start]국민건강영양조사(KNHANES) (8088x799) [cite: 200]
2.  [cite_start]**전처리 및 가공 (Preprocessing & Processing)** [cite: 189]
    * [cite_start]데이터 병합 (최종 7303x10) [cite: 202, 208][cite_start], 결측치 처리 [cite: 202, 288][cite_start], 그룹핑 [cite: 211]
    * [cite_start]*피처 엔지니어링*: 평균 혈압(MAP) 계산 [cite: 290][cite_start], (주중/주말) 수면시간 계산 [cite: 291, 296]
    * [cite_start]*인코딩*: 범주형 변수(직업, 성별 등)에 LabelEncoder / One-Hot Encoding 적용 [cite: 211, 293, 294, 295]
    * [cite_start]*스케일링*: 수치형 데이터에 StandardScaler 표준화 [cite: 298, 300, 302]
    * [cite_start]*불균형 처리*: **SMOTE** 오버샘플링 기법 사용 [cite: 301, 303, 304, 306]
3.  [cite_start]**EDA (Exploratory Data Analysis)** [cite: 190]
    * [cite_start]변수별 히스토그램 작성 [cite: 210][cite_start], 상관관계 히트맵 분석 [cite: 242]
4.  **모델링 (Modeling)**
    * [cite_start]**분류 (Classification)**[cite: 186]:
        * [cite_start]*모델*: Random Forest [cite: 306][cite_start], **LightGBM** [cite: 306]
        * [cite_start]*전략*: SMOTE 및 Stratified K-Fold(k=5) 교차 검증 [cite: 306]
        * [cite_start]*최종 선택*: **LightGBM** (Accuracy: 0.74, Macro F1: 0.74) [cite: 310, 313, 316, 317]
    * [cite_start]**군집 (Clustering)**[cite: 193]:
        * [cite_start]*모델*: **K-Means** [cite: 352, 401]
        * [cite_start]*최적 K 탐색*: Elbow Method [cite: 351, 400, 410, 430] [cite_start]및 Silhouette Score [cite: 353, 395, 402, 418, 473, 474, 475] [cite_start](K=3~4 추천 [cite: 476])
        * [cite_start]*결과*: 군집별 특성(직업, 성별, BMI 등) 비율 분석 [cite: 479, 491]
5.  [cite_start]**웹 서비스 구현 (FLASK Webpage Production)** [cite: 191]
    * [cite_start]사용자 입력값을 받아 실시간 전처리 후 모델 적용 [cite: 531, 562, 576, 577]
    * [cite_start]분석 결과를 **JSON** 형식으로 **REST** 기반 비동기 통신 [cite: 642, 644]
6.  [cite_start]**맞춤 리포트 및 가이드 제공 (User-customized Report & Guide)** [cite: 192]
    * [cite_start]JSON 데이터를 받아 웹페이지에 동적 시각화 [cite: 649, 652]

---

## [cite_start]5. 향후 계획 (Future Work) [cite: 677]

* [cite_start]**모바일 앱 개발**을 통한 접근성 확대 [cite: 673]
* [cite_start]**실시간 웨어러블 데이터** (스마트 워치, 링 등) 연동 [cite: 674]
* [cite_start]**딥러닝(Deep Learning)**을 통한 모델 고도화 [cite: 675]
* [cite_start]다양한 **수면 장애 유형** (불면증, 무호흡증 등) 예측 및 진단 기능 추가 [cite: 676]

---

## 6. 팀원 및 역할

* [cite_start]**박예슬** [cite: 21]
    * [cite_start]데이터 수집 & 전처리함 [cite: 23]
    * [cite_start]분류 및 전체적 모델링함 [cite: 24]
    * [cite_start]PPT 제작 및 발표함 [cite: 25]
* [cite_start]**임규리** [cite: 22]
    * [cite_start]데이터 수집함 [cite: 26]
    * [cite_start]데이터 전처리 및 가공함 [cite: 27]
    * [cite_start]군집 분석 모델링함 [cite: 28]
    * [cite_start]웹페이지 시각화(REST) 구현함 [cite: 29]
