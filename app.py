import os
from flask import Flask, render_template, request
import pandas as pd
import joblib

# --- 1. Flask 애플리케이션 초기화 ---
app = Flask(__name__, template_folder='templates')

# --- 2. 경로 및 파일 정의 ---
MODELS_DIR = 'models'
SIMULATED_DATA_FILE = 'data/simulated_wearable_data.csv'

# --- 3. 전역 변수 및 모델 로드 ---
df_with_cluster = pd.DataFrame()
try:
    # --- 원래 변수명으로 되돌린 코드 ---
    model_pipeline_with_bp = joblib.load(os.path.join(MODELS_DIR, 'model_pipeline_with_bp.pkl'))
    model_pipeline_no_bp = joblib.load(os.path.join(MODELS_DIR, 'model_pipeline_no_bp.pkl'))

    # ★수정★: 원래 변수명인 kmeans_model 로 되돌림
    kmeans_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    # ★수정★: 원래 변수명인 cluster_scaler 로 되돌림
    cluster_scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

    # ★추가★ 혈압 없을 때 사용하는 모델 및 스케일러
    kmeans_no_bp_model = joblib.load(os.path.join(MODELS_DIR, 'kmeans_no_bp_model.pkl'))
    scaler_no_bp = joblib.load(os.path.join(MODELS_DIR, 'scaler_no_bp.pkl'))

    # ★추가★ 결과 비교 차트를 위한 군집별 평균값
    cluster_averages_from_pkl = joblib.load(os.path.join(MODELS_DIR, 'cluster_averages.pkl'))
    cluster_averages_no_bp = joblib.load(os.path.join(MODELS_DIR, 'cluster_averages_no_bp.pkl'))

    # ★추가★ 범주형 데이터 처리를 위한 레이블 인코더 딕셔너리
    le_dict = joblib.load(os.path.join(MODELS_DIR, 'le_dict.pkl'))

    print("모든 모델 및 전처리기 로드 완료.")

except FileNotFoundError as e:
    print(f"CRITICAL ERROR: 필수 모델 파일('{e.filename}')을 찾을 수 없습니다. 'models' 디렉토리를 확인하세요.")

# --- 4. 모델별 피처 목록 정의 (가장 중요한 부분) ★★★ ---
# Last.ipynb 노트북에서 최종적으로 학습에 사용된 피처 목록들을 명확히 정의합니다.

# 스케일러(scaler.pkl)가 학습한 피처
SCALER_FEATURES = ['Age', 'Stress_Level', 'Heart_Rate', 'MAP']
SCALER_FEATURES_NO_BP = ['Age', 'Sleep_Duration', 'Stress_Level', 'Heart_Rate']

# K-Means 군집 모델(kmeans_model.pkl)이 학습한 피처
KMEANS_FEATURES = ['Gender', 'Age', 'Occupation', 'Stress_Level', 'BMI_Category', 'Heart_Rate', 'MAP']
KMEANS_FEATURES_NO_BP = ['Gender', 'Age', 'Occupation', 'Sleep_Duration', 'Stress_Level', 'BMI_Category', 'Heart_Rate']

# 수면의 질 예측 모델(model_pipeline_with_bp.pkl)이 학습한 피처
PREDICTION_FEATURES = ['Gender', 'Age', 'Occupation', 'Sleep_Duration', 'Stress_Level', 'BMI_Category', 'Heart_Rate', 'MAP']
# ★추가★ 수면의 질 예측 모델 - 혈압 없는 버전이 학습한 피처 (MAP 제외)
PREDICTION_FEATURES_WITHOUT_BP = ['Gender', 'Age', 'Occupation', 'Sleep_Duration', 'Stress_Level', 'BMI_Category', 'Heart_Rate']


# UI 및 결과 표시에 사용할 이름 매핑
# ★수정★: le_dict의 올바른 내부 구조로 접근합니다.
le_occupation_inverse_map = le_dict['label_to_string']['Occupation']
occupation_ui_value_to_le_value_map = le_dict['string_to_label']['Occupation']

feature_display_names = {
    'Age': '나이', 'Stress_Level': '스트레스', 'Heart_Rate': '심박수',
    'Systolic': '수축기 혈압', 'Diastolic': '이완기 혈압', 'Sleep_Duration': '수면 시간', 'MAP': '평균 동맥압'
}


# --- 5. 앱 시작 시 전체 데이터 로드 및 군집화 ---
try:
    df = pd.read_csv(SIMULATED_DATA_FILE)
    df.columns = [col.replace(' ', '_') for col in df.columns]

    if 'Blood_Pressure' in df.columns:
        df[['Systolic', 'Diastolic']] = df['Blood_Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop('Blood_Pressure', axis=1)

    # ★★★ 여기를 수정하세요: le_dict의 올바른 구조로 접근 ★★★
    # le_dict 딕셔너리 내부의 'string_to_label' 키를 통해 실제 매핑 정보에 접근합니다.
    df['Gender'] = df['Gender'].map(le_dict['string_to_label']['Gender'])
    df['Occupation'] = df['Occupation'].map(le_dict['string_to_label']['Occupation']).fillna(-1).astype(int)
    df['BMI_Category'] = df['BMI_Category'].map(le_dict['string_to_label']['BMI_Category'])
    df['MAP'] = df['Diastolic'] + (1/3) * (df['Systolic'] - df['Diastolic'])

    df_for_kmeans = df.copy()

    # 스케일러가 학습한 피처(SCALER_FEATURES)만 정확히 선택하여 스케일링
    # 여기서는 원래 변수명인 cluster_scaler 와 SCALER_FEATURES 를 그대로 사용합니다.
    df_for_kmeans[SCALER_FEATURES] = cluster_scaler.transform(df_for_kmeans[SCALER_FEATURES])

    # K-Means 모델이 학습한 피처(KMEANS_FEATURES)를 사용하여 예측
    # 여기서는 원래 변수명인 kmeans_model 과 KMEANS_FEATURES 를 그대로 사용합니다.
    X_for_clustering = df_for_kmeans[KMEANS_FEATURES]
    all_clusters = kmeans_model.predict(X_for_clustering)

    df_with_cluster = df.copy()
    df_with_cluster['Cluster'] = all_clusters
    print("'df_with_cluster' 전역 데이터프레임 생성 완료.")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"앱 시작 시 전체 데이터 처리 중 오류 발생: {e}")

# --- 6. 맞춤형 추천 메시지 함수 (최종 수정본) ---
def recommend_sleep(cluster_id, sleep_quality_score, has_bp):
    if has_bp:
        # 혈압 정보가 있을 때의 그룹별 기본 설명 및 해결 방안
        base_messages = {
            0: "당신은 전반적으로 '균형 잡힌 건강인' 유형에 속합니다. 좋은 생활 습관을 유지하기 위해 주 3회 이상 꾸준한 유산소 운동과 균형 잡힌 식단으로 지금의 건강 상태를 관리해 보세요.",
            1: "당신은 전반적으로 '활동적인 도시의 전문직' 유형에 속합니다. 활동적인 생활은 좋지만 수면의 질이 따라오지 못하는 경향이 있습니다. 잠들기 1시간 전에는 스마트폰 사용을 중단하고, 명상이나 가벼운 스트레칭으로 긴장을 푸는 것이 수면의 질을 높이는 데 도움이 될 것입니다.",
            2: "당신은 전반적으로 '원숙한 실버' 유형에 속합니다. 이 유형은 혈압 관리가 중요합니다. 매일 아침 가벼운 산책으로 신체 리듬을 잡고, 나트륨 섭취를 줄이는 저염식 식단을 실천하여 혈압을 안정적으로 관리하는 것을 추천합니다.",
            3: "당신은 전반적으로 '집중 관리 대상' 유형에 속합니다. 비만도와 혈압 관리가 시급해 보입니다. 식사 일지를 작성하여 칼로리를 조절하고, 주 2회 이상 근력 운동을 시작하여 기초대사량을 높이는 것이 좋습니다. 필요하다면 전문가와 상담하세요."
        }
    else:
        # 혈압 정보가 없을 때의 그룹별 기본 설명 및 해결 방안
        base_messages = {
            0: "당신은 전반적으로 '젊은 건강 그룹' 유형에 속합니다. 좋은 건강 상태를 유지하기 위해 현재의 생활 패턴을 지키는 것이 중요합니다. 여기에 주 1~2회 정도 새로운 취미나 운동을 더해 신체적, 정신적 건강을 더욱 증진시켜 보세요.",
            1: "당신은 전반적으로 '활동적인 사회 초년생' 유형에 속합니다. 활동적인 만큼 충분한 휴식이 필요합니다. 매일 같은 시간에 잠자리에 드는 습관을 만들고, 수면 시간을 최소 7시간 이상 확보하도록 노력하는 것이 컨디션 회복에 큰 도움이 됩니다.",
            2: "당신은 전반적으로 '집중적인 스트레스 관리'가 필요한 유형입니다. 스트레스는 만병의 근원입니다. 하루 10분간의 명상 시간을 갖거나, 좋아하는 음악을 듣는 등 자신만의 스트레스 해소법을 찾아 꾸준히 실천하는 것을 추천합니다.",
            3: "당신은 전반적으로 '수면 부족과 신체 지수 관리'가 필요한 유형에 속합니다. 수면 부족과 높은 BMI는 서로 영향을 줄 수 있습니다. 저녁 식사는 잠들기 3시간 전에 마치고, 가벼운 유산소 운동(예: 빠르게 걷기)을 주 3회 이상 실천하여 건강한 사이클을 만들어보세요."
        }

    # 수면 점수에 따른 추가 조언 (강조 표시 제거)
    additional_advice = ""
    if sleep_quality_score <= 5:
        additional_advice = "<br><br>하지만 현재 당신의 수면 점수는 개선이 필요한 상태입니다. 충분한 휴식을 취하고, 자기 전 스마트폰 사용을 줄이는 등 수면 환경을 개선해 보세요."
    elif sleep_quality_score <= 7:
        additional_advice = "<br><br>현재 수면 건강은 양호한 편입니다. 지금처럼 꾸준히 좋은 습관을 유지해 주세요."
    else:
        additional_advice = "<br><br>현재 당신의 수면 건강은 매우 훌륭합니다! 최상의 컨디션을 계속 유지하세요."

    # 최종 메시지 조합
    base_message = base_messages.get(cluster_id, "분석된 군집 정보가 없습니다.")
    return base_message + additional_advice

# --- 7. Flask 라우트 ---
@app.route('/')
def index():
    user_ids = df_with_cluster['user_id'].unique().tolist() if not df_with_cluster.empty else []
    # 초기 페이지 로드 시 템플릿에서 사용하는 모든 변수들에 대해 기본값을 전달합니다.
    return render_template('index.html',
                           user_ids=user_ids,
                           analysis_type='none',
                           predicted_sleep_quality=None,
                           predicted_cluster_id=None,
                           recommendation_message="",
                           user_latest_features={},
                           cluster_averages={},
                           chart_user_data={},
                           chart_cluster_avg_data={},
                           cluster_size=0)


@app.route('/manual_analyze', methods=['POST'])
def manual_analyze():
    try:
        # 1. 폼 데이터 가져오기 및 타입 변환
        form_data = {
            'manual_height': float(request.form.get('manual_height')) if request.form.get('manual_height') else None,
            'manual_weight': float(request.form.get('manual_weight')) if request.form.get('manual_weight') else None,
            'manual_occupation': int(request.form.get('manual_occupation')) if request.form.get(
                'manual_occupation') else None,
            'manual_systolic': float(request.form.get('manual_systolic')) if request.form.get(
                'manual_systolic') else None,
            'manual_diastolic': float(request.form.get('manual_diastolic')) if request.form.get(
                'manual_diastolic') else None,
        }

        # 필수 필드 확인
        if not all(k in form_data and form_data[k] is not None for k in
                   ['manual_height', 'manual_weight', 'manual_occupation']):
            return "<h1>오류 발생</h1><p>키, 몸무게, 직업은 필수 입력 값입니다.</p>", 400

        # 2. 분석용 데이터프레임 생성 및 피처 엔지니어링
        # UI에서 받지 않은 기본값 설정
        base_profile = {'Gender': 'Male', 'Age': 35, 'Stress_Level': 5, 'Heart_Rate': 70, 'Sleep_Duration': 7.2}
        base_profile.update({k: v for k, v in form_data.items() if v is not None})

        input_df = pd.DataFrame([base_profile])

        # le_dict와 매핑 변수를 사용한 인코딩
        input_df['Gender'] = input_df['Gender'].map(le_dict['string_to_label']['Gender'])
        input_df['Occupation'] = input_df['manual_occupation'].map(occupation_ui_value_to_le_value_map)

        height_m = form_data['manual_height'] / 100
        weight_kg = form_data['manual_weight']
        bmi = weight_kg / (height_m ** 2)
        category_str = 'Normal' if bmi < 25 else ('Overweight' if bmi < 30 else 'Obese')
        input_df['BMI_Category'] = le_dict['string_to_label']['BMI_Category'].get(category_str)

        # 3. ★★★ 핵심 로직: 혈압 정보 유무에 따라 모델 및 피처 선택 ★★★
        has_bp = form_data['manual_systolic'] is not None and form_data['manual_diastolic'] is not None

        if has_bp:
            input_df['MAP'] = form_data['manual_diastolic'] + (1 / 3) * (
                        form_data['manual_systolic'] - form_data['manual_diastolic'])

            # 혈압 있을 때 사용할 변수들 할당
            pred_model, pred_features = model_pipeline_with_bp, PREDICTION_FEATURES
            k_model, k_scaler, k_features, k_scaler_features, avg_dict = kmeans_model, cluster_scaler, KMEANS_FEATURES, SCALER_FEATURES, cluster_averages_from_pkl
        else:
            # 혈압 없을 때 사용할 변수들 할당
            pred_model, pred_features = model_pipeline_no_bp, PREDICTION_FEATURES_WITHOUT_BP
            k_model, k_scaler, k_features, k_scaler_features, avg_dict = kmeans_no_bp_model, scaler_no_bp, KMEANS_FEATURES_NO_BP, SCALER_FEATURES_NO_BP, cluster_averages_no_bp

        # 4. 수면의 질 예측
        predicted_sleep_quality = int(pred_model.predict(input_df[pred_features])[0])

        # 5. 군집 분류
        df_for_kmeans = input_df[k_features].copy()

        # ★★★ 이 부분을 추가하세요: 모델에 입력하기 전 NaN 값을 0으로 채웁니다. ★★★
        df_for_kmeans.fillna(0, inplace=True)

        # 스케일링 적용
        features_to_scale = [f for f in k_scaler_features if f in df_for_kmeans.columns]
        if features_to_scale:
            df_for_kmeans[features_to_scale] = k_scaler.transform(df_for_kmeans[features_to_scale])

        # 군집 예측
        predicted_cluster_id = int(k_model.predict(df_for_kmeans)[0])

        # 6. 결과 데이터 생성 및 템플릿 렌더링
        recommendation_message = recommend_sleep(predicted_cluster_id, predicted_sleep_quality, has_bp)
        cluster_averages = avg_dict.get(predicted_cluster_id, {})

        # 차트 데이터 생성
        chart_user_data = {
            '나이': int(input_df['Age'].iloc[0]),
            '스트레스': int(input_df['Stress_Level'].iloc[0]),
            '심박수': int(input_df['Heart_Rate'].iloc[0]),
            '수면 시간': input_df['Sleep_Duration'].iloc[0]
        }

        # 클러스터 평균에서 비교할 데이터 추출
        chart_cluster_avg_data = {
            feature_display_names.get(k, k): round(v, 1) for k, v in cluster_averages.items()
            if feature_display_names.get(k, k) in chart_user_data
        }

        # 혈압 정보 있을 경우 차트 데이터에 추가
        if has_bp:
            chart_user_data['평균 동맥압'] = round(input_df['MAP'].iloc[0], 1)
            chart_cluster_avg_data['평균 동맥압'] = round(cluster_averages.get('MAP', 0), 1)

        return render_template('index.html',
                               analysis_type='manual',
                               predicted_sleep_quality=predicted_sleep_quality,
                               predicted_cluster_id=predicted_cluster_id,
                               recommendation_message=recommendation_message,
                               chart_user_data=chart_user_data,
                               chart_cluster_avg_data=chart_cluster_avg_data
                               )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"<h1>분석 중 오류가 발생했습니다.</h1><p>{e}</p>", 500


@app.route('/analyze_historical', methods=['POST'])
def analyze_historical():
    return "기존 사용자 데이터 분석 기능은 현재 개발 중입니다."


# --- 8. Flask 앱 실행 ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

