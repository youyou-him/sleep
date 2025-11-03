import pandas as pd

# 상대 경로로 데이터 불러오기
df = pd.read_csv('../data/Sleep_health_and_lifestyle_dataset.csv')

# 상위 5개 데이터 미리 보기
print(df.head())