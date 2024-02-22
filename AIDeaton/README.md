# AIchemist AIDeaton ❤️‍🩹심쿵팀

[Heart Attack Analysis & Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/code?datasetId=1226038&sortBy=voteCount)

# ✅ 데이터 살펴보기

## **About this dataset**

- age : Age of the patient
- sex : Sex of the patient
- cp : Chest Pain type
    - Value 0: typical angina
    - Value 1: atypical angina
    - Value 2: non-anginal pain
    - Value 3: asymptomatic
- trtbps : resting blood pressure (in mm Hg)
- chol : cholestoral in mg/dl fetched via BMI sensor
- fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- rest_ecg : resting electrocardiographic results
    - Value 0: normal
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- thalachh : maximum heart rate achieved
- exng: exercise induced angina (1 = yes; 0 = no)
- caa: number of major vessels (0-3)
- target : 0 = less chance of heart attack, 1 = more chance of heart attack

---

```python
heart_data = pd.read_csv('heart.csv')
heart_data.head()
```
```python
heart_data.info()
```
### → NULL값은 없는 것으로 파악된다!

```python
heart_data.describe( )
```

# ✅ 데이터 전처리

## 시각화

### **히스토그램으로 연속형/범주형 변수인지 확인**

```python
plt.figure(figsize=(15,15))
for i,col in enumerate(heart_data.columns,1):
    plt.subplot(7,2,i)
    plt.title(f"Distribution of {col} Data")
    sns.histplot(heart_data[col],kde=True)
    plt.tight_layout()
    plt.plot()
```


- 연속형: **age, trtbps, chol, thalach, oldpeak**
- 범주형: **sex, cp, fbs, restecg, exng, slp, caa**, thall, output

```python
cat_cols = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
con_cols = ["age","trtbps","chol","thalachh","oldpeak"]
```

### 연속형 변수 boxplot으로 시각화

```python
# 플롯의 행과 열을 설정
n_rows = 3
n_cols = 2

# matplotlib의 subplots를 사용하여 여러 그래프를 그립니다.
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

# 각 변수에 대한 Boxplot을 그립니다.
for i, var in enumerate(con_cols):
    row = i // n_cols
    col = i % n_cols
    sns.boxplot(x='output', y=var, data=heart_data, ax=axes[row, col])
```


- chol은 심장마비 환자군의 수치가 오히려 낮았음
- thakachh은 최솟값, 중앙값, 최댓값 등이 모두 심장마비 환자군에서 높게 나타남

### **범주형 변수 barplot으로 시각화**

```python
# 플롯의 행과 열을 설정
n_rows = 2
n_cols = 4

# matplotlib의 subplots를 사용하여 여러 그래프를 그립니다.
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))

# 각 변수에 대한 Barplot을 그립니다.
for i, var in enumerate(cat_cols):
    row = i // n_cols
    col = i % n_cols
    sns.countplot(x='output', hue=var, data=heart_data, ax=axes[row, col])
```

- cp: type=0인 경우 발병확률 낮음, type=2인 경우 발병확률 높음
- rest_ecg
    - Value 0: normal → 발병확률 낮음
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) → 발병확률 높음
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- exng: 0인 경우(exercise induced angina가 아닌경우) 발병확률 높음
- caa: type=0인 경우 발병확률 높음

## 이상치 제거

- 상관도 확인: `.corr()`

```python
corr = heart_data.corr()

# 상관관계 시각화
# -1 또는 1에 가까울수록 상관관계 높은 것
sns.clustermap(corr, 
               annot = True,      # 실제 값 화면에 나타내기
               cmap = 'RdYlBu_r',  # Red, Yellow, Blue 색상으로 표시
               vmin = -1, vmax = 1, # 컬러차트 -1 ~ 1 범위로 표시
              )
```


- 이상치 제거: IQR 이용

```python
# 1사분위수(Q1) 및 3사분위수(Q3) 계산
Q1 = heart_data.quantile(0.25)
Q3 = heart_data.quantile(0.75)

# IQR 계산
IQR = Q3 - Q1

# 임계값 설정 (일반적으로 1.5 이상을 사용)
threshold = 1.5

# 이상치 제거
clean_data = heart_data[~((heart_data < (Q1 - threshold * IQR)) | (heart_data > (Q3 + threshold * IQR))).any(axis=1)]
```


## 인코딩

사이킷런의 머신러닝 알고리즘은 문자열 값에 대해 적용할 수 없음 

→ 피처들의 자료형을 확인했을 때 문장형, 즉 object형 피처가 있을 경우 이들을 숫자형으로 바꿔주는 작업이 필요함

- 머신러닝의 대표적인 인코딩 방식
    - **레이블 인코딩:** 카테고리형 데이터를 코드형 숫자 값으로 변환해주는 작업
        
        → 문자형으로 되어있는 피처가 숫자형으로 변환됨
        
    - **원-핫 인코딩:** 데이터의 유형에 따라 새로운 데이터를 추가하여 고유 값에 해당하는 칼럼에만 1을 표시하고 나머지 칼럼에는 0으로 표시하는 방법
        
        → 열들의 고유 값을 차원을 변환한 뒤, 해당하는 칼럼에만 1을 표시, 나머지에는 0을 표시(pandas의 `get_dummies` 메서드를 사용하면 숫자형 값 변환없이 원-핫 인코딩을 쉽게 구현 할 수 있음)
        

### **→ 범주형 데이터인 경우 원-핫 인코딩 적용 (이미 범주형 데이터가 숫자형임)**

## 피처 스케일링 (표준화, 정규화)

: 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업

- **표준화:** 평균이 0이고 분산이 1인, ‘가우시안 정규 분포’를 가진 값으로 변환해주는 작업
- **정규화:** 서로 다른 크기의 데이터들이 있을 때, 이들의 크기를 통일하기 위해 크기를 변환해주는 작업 → 개별 데이터의 크기를 모두 똑같은 단위로 변경
- **로버스트(Robust)**: 데이터의 중앙값 = 0, IQR = 1이 되도록 스케일링하는 기법
    - `RobustScaler`를 사용하면 모든 변수들이 같은 스케일을 갖게 되며, `StandardScaler`에 비해 스케일링 결과가 더 넓은 범위로 분포
    - `StandardScaler`**에 비해 이상치의 영향이 적어진다는 장점**
    - 금융 데이터 분석, **생물학적 데이터 처리**, **센서 데이터 분석** 등에서 이상치가 자주 발생하는 경우에 주로 사용

### → `RobustScaler` 사용

```python
# creating a copy of df
df = clean_data

# encoding the categorical columns
df = pd.get_dummies(df, columns = cat_cols, drop_first = True)

# defining the features and target
X = df.drop(['output'],axis=1)
y = df[['output']]

# instantiating the scaler
scaler = RobustScaler()

# scaling the continuous featuree
X[con_cols] = scaler.fit_transform(X[con_cols])
print("The first 5 rows of X are")
X.head()
```


# ✅ 모델링

정답 레이블인 ‘심장마비 발병 여부’ 데이터를 알고 있음

**→ 지도 학습**

지도학습은 회귀문제와 분류문제로 나뉨

<aside>
💡 **회귀 vs 분류**

- 회귀와 분류의 공통점
    - 지도학습: 정답이 있는 데이터를 활용해 모델을 학습시키는 방법
- 회귀와 분류의 차이점
    
    
    | 회귀 ( Regression ) | 분류 ( Classification) |
    | --- | --- |
    | 예측하고자 하는 타겟값이 실수(숫자)인 경우 | 예측하고자 하는 타겟값이 범주형 변수인 경우 |
    | 연속적 예측값 | 이산적 예측값 |
    | ex) 손해액, 매출량, 거래량, 파산 확률 등 예측 | ex) 이진 분류 / 다중 분류 |
</aside>


심장마비 데이터의 경우, 정답 값이 0과 1로 이뤄진 이진분류 문제

### → 분류 모델 사용

## 1. 결정 트리

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

# 학습용과 테스트용으로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 의사결정나무 모델 초기화
model = DecisionTreeClassifier(random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
```

```
Accuracy: 0.6956521739130435
F1 Score: 0.75
```

## 2. 랜덤 포레스트

```python
from sklearn.ensemble import RandomForestClassifier

# 학습용과 테스트용으로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 초기화
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 모델 학습
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
```

```
Accuracy: 0.717391304347826
F1 Score: 0.7719298245614036
```

## 3. XGBoost

```python
import xgboost as xgb

# 학습용과 테스트용으로 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 초기화 및 학습
model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=42)
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 정확도 출력
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
```

```
Accuracy: 0.8043478260869565
F1 Score: 0.8421052631578947
```

## 4. 로지스틱 회귀

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report

# 인코딩된 데이터프레임을 사용하여 X, y 정의
X = df1.drop(['output'], axis=1) #output drop!
y = df1[['output']]

# 데이터를 훈련 및 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 스케일러 초기화 및 연속형 변수 스케일링
scaler = RobustScaler()
X_train[con_cols] = scaler.fit_transform(X_train[con_cols])
X_test[con_cols] = scaler.transform(X_test[con_cols])

# 로지스틱 회귀 모델 초기화 및 학습
log_reg = LogisticRegression(max_iter=1000)  # 반복 횟수 설정
log_reg.fit(X_train, y_train.values.ravel())  # 모델 학습

# 테스트 세트로 예측
y_pred = log_reg.predict(X_test)

# 정확도 계산 및 출력
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# 분류 보고서 출력
print(classification_report(y_test, y_pred))
```

```
Accuracy: 0.90163934426229
F1 Score: 0.90
```

## 평가

|  | Accuracy | F1 Score |
| --- | --- | --- |
| 1. 결정 트리 | 0.6956521739130435 | 0.75 |
| 2. 랜덤 포레스트 | 0.717391304347826 | 0.7719298245614036 |
| 3. XGBoost | 0.8043478260869565 | 0.8421052631578947 |
| 4. 로지스틱 회귀 | 0.90163934426229 | 0.90 |

### → 최적의 모델: 로지스틱 회귀
