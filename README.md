# 이성진 · Lee SeongJin

국민대학교 경영학부 회계학전공, AI빅데이터융합경영 복수전공 · 데이터 분석 / 머신러닝 프로젝트 모음

## 프로젝트

| 프로젝트 | 유형 | 주제 | 결과 |
|---|---|---|---|
| [신용카드 사기 거래 탐지](https://github.com/glendale123-jpg/credit-card-fraud-detection) | 개인 · Dacon | 라벨 30건으로 하는 비지도 이상탐지 | Public 0.9305 / Private 0.9054 (macro-F1) |
| [고객 성별 예측](https://github.com/glendale123-jpg/kml-gender-prediction) | 팀(4인) · 머신러닝 수업 경진대회 | 백화점 거래 → 고객 성별, 거래단위 stacking 설계 | Private 0.72896 (ROC-AUC), 0.73330 제출 시 Public 리더보드 1위 |
| [설비 비정상 작동 분류](https://github.com/glendale123-jpg/dacon-anomaly-classification) | 팀(4인) · 머신러닝 수업 | 52개 센서 → 21개 클래스 분류, 4-모델 앙상블 개선 | Private 0.8598 → 0.8744 |
| [BC카드 시장 침투도 분석](https://github.com/glendale123-jpg/BCCARD-submission) | 팀장(3인) · 공모전 | 1,813개 시장의 기대 결제액 회귀 → 공략 우선순위 발굴 | 제1회 AI금융빅데이터플랫폼 공모전 제출 |
| [유가·전기차 EDA](https://github.com/glendale123-jpg/EV-OIL) | 팀 · EDA | 유가·소득·충전소와 전기차 보급의 관계 | 회귀·이중 축 시각화 |

### 신용카드 사기 거래 탐지
- 거래의 88%가 30차원 공간의 한 평면 위에 있고 사기의 83%는 평면 밖에 있다는 **데이터 구조를 발견**
- Elliptic Envelope가 실패한 원인(MCD의 exact fit)을 규명하고 `support_fraction` 조정으로 최고 모델로 전환
- 공개 코드 ablation으로 오토인코더 성능의 핵심이 L1 손실임을 확인 (MSE로 바꾸면 0.91 → 0.51)
- 9번의 제출로 macro-F1 0.66 → 0.93

### 고객 성별 예측 (팀)
- 고객 단위 피처가 포화된 상황에서 **거래 한 건 단위로 먼저 예측한 뒤 고객별로 다시 집계**하는 2단계 stacking 설계
- 같은 고객의 거래가 학습·검증에 섞이지 않도록 고객 fold 기준으로 OOF를 만들어 누수 차단
- 피처 9종을 제출 전 OOF로 먼저 걸러 제출 횟수 절약, OOF와 LB 개선폭 일치 확인

### 설비 비정상 작동 분류 (팀)
- 피처 엔지니어링 탐색 담당: 행 통계·센서 조합 피처와 변수 제거 실험, Argmax vs Hungarian 후처리 비교
- 팀 최종: LDA 피처 증강 + Fold-safe 검증 + SNN 도입 + OOF 기반 가중치 탐색

### BC카드 시장 침투도 분석 (팀장)
- 업종별 회귀로 "점포 수·인구 대비 나와야 할 결제액"을 추정, 실제와의 차이를 표준화한 지표로 4단 필터 설계
- 재현성 검증(반기 재산출 상관 0.946)과 공공데이터포털 API 수집

## 기술
Python · pandas · NumPy · scikit-learn · PyTorch · LightGBM · statsmodels · SQL
