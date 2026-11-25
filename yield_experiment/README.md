# Lek Tablet Yield Anomaly Detection & RCA

## 1. 프로젝트 개요

Lek Dataset을 활용한 의약품 타정 수율 예측, 통계적 이상 탐지 및 RCA Agent 연동을 위한 PoC 프로젝트이다.

- 제품별 가상 수율 규격 산출
- ML 기반 예상 수율 산출
- I-MR 관리도 및 Nelson Rule 1~3 기반 이상 탐지
- 과거 동일 제품 배치와 공정변수 비교
- 시계열 기반 생산 손실 및 정지·재가동 분석
- RCA Agent 연동용 JSON 생성 및 검증

## 2. 실행 환경

- Python 3.12
- uv
- NumPy 2.3.5
- Pandas 2.2.3
- Scikit-learn 1.8.0
- Matplotlib 3.10.8
- SHAP 0.50.0
- Jupyter Notebook

## 3. 데이터 구성

Lek Dataset 전체 25개 제품, 1,005개 배치를 활용한다.

원본 데이터는 yield_experiment/data/에 별도로 준비한다.

- Laboratory.csv
- Process.csv
- Process.zip

원본 데이터와 가상환경은 Git에서 관리하지 않는다.

## 4. 프로젝트 구성

| 파일 및 폴더 | 설명 |
|---|---|
| config/ | 제품별 가상 규격 설정 |
| notebooks/ | 최종 분석 Notebook |
| experiment.py | 초기 ML 및 SPC 분석 |
| followup_experiment.py | 후속 ML 분석 |
| spec_*.py | 제품별 규격 및 통계 분석 |
| rca_*.py | RCA 분석 및 검증 |
| results/ | 선별된 분석 결과 및 RCA JSON |
| test_rules.py | 통계 규칙 테스트 |
| design_notes.md | 초기 분석 설계 기록 |
| pyproject.toml | Python 실행 환경 설정 |
| uv.lock | 패키지 의존성 관리 |

## 5. 최종 Notebook

| Notebook | 주요 내용 |
|---|---|
| 01_batch_data_analysis.ipynb | 배치 수행시간, timestamp 및 데이터 품질 분석 |
| 02_yield_prediction_ml_v2.ipynb | ML 성능, 재현성 및 민감도 검증 |
| 03_batch790_rca_analysis_v3.ipynb | Batch 790 RCA 및 결측 처리 |
| 04_statistical_anomaly_detection.ipynb | 통계적 이상 탐지 및 관리한계 안정성 검증 |

01 Notebook을 먼저 실행한다.

02 Notebook은 01의 배치 수행시간 분석 결과를 사용한다.

03 Notebook은 기존 제품별 Train/Test 분할 및 Q10 규격 산출 결과를 사용한다.

04 Notebook은 01의 분석 결과와 기존 Q10 규격 산출 결과를 사용한다.

기존 Python 스크립트와 초기 설계 기록은 이전 분석 과정의 재현 및 비교를 위해 보존한다.

## 6. ML 모델 분석 결과

- 분석 제품: Code 13, 17, 23
- 학습 배치: 315개
- 평가 배치: 209개
- 경계 제외: 1개
- 최종 후보: Random Forest
- 입력변수: 제품 정보, 원료 특성 및 과거 배치 수율
- CV Macro MAE: 0.5815%p
- Test MAE: 0.5194%p
- Test R²: 0.005
- 저수율 평가 배치 17개 중 탐지 0개

기존 고정 초기 30분 기반 센서 입력변수를 제거하고 생산 시작 전 확보 가능한 데이터를 기준으로 분석하였다.

ML 모델은 예상 수율 제공을 위한 참고용으로 활용하며, 일탈 자동 판정에는 사용하지 않는다.

## 7. 통계적 이상 탐지 결과

- 가상 규격 검사: 제품별 Q10 기반 LSL 및 기존 가상 USL
- SPC: I-MR 관리도
- Nelson Rule 1: 관리한계 이탈
- Nelson Rule 2: 지속적인 수준 변화
- Nelson Rule 3: 연속 상승 및 하락 추세
- 관리한계 기준 기간 민감도 검증

| 분석 항목 | 결과 |
|---|---:|
| 전체 평가 배치 | 232개 |
| 가상 규격 이탈 | 14개 |
| SPC 이상 신호 | 12개 |
| 중복 탐지 | 5개 |
| 통합 이상 신호 | 21개 |

평가 대상은 Code 13, 15, 17, 23이다.

규격 검사와 SPC를 병행하며, 동일 배치에 발생한 복수의 신호는 하나의 조사 요청으로 통합한다.

## 8. RCA 시나리오

| 제품 | Batch | 실제 수율 | 주요 조사 방향 |
|---|---|---:|---|
| Code 17 | 789 | 87.973% | 공정조건 수준 변화 및 생산 중반부 손실 |
| Code 15 | 799 | 95.739% | 반복 정지 및 생산 초기 안정화 |
| Code 23 | 790 | 93.797% | 반복 정지, 압축력 변동 및 생산 중 손실 |

RCA Agent 연동용 JSON은 다음 경로에 위치한다.

- results/spec_all_products_q10/rca_agent_code17_batch789.json
- results/spec_all_products_q10/rca_agent_code15_batch799.json
- results/revision_20261125/03_batch790_rca/phase2_v3/rca_agent_code23_batch790.json

## 9. 분석 한계

- 모든 규격은 연구용 PoC 기준이며 실제 승인 규격이 아니다.
- 실제 일탈 정답 라벨이 없어 이상 탐지 정확도를 평가할 수 없다.
- ML 모델의 저수율 예측 성능이 제한적이다.
- 원료 검사 및 과거 수율의 실제 확보 시점은 확인이 필요하다.
- 관리한계는 기준 기간의 안정성에 영향을 받는다.
- Batch 790은 기존 학습 기간의 별도 소급 분석 사례이다.
- 시계열 결측값은 설비 정지 상태로 임의 대체하지 않는다.
- 폐기 카운터 증가량은 실제 폐기 수량과 구분한다.
- 기존 RCA와 최신 통계 검증은 일부 데이터 전처리 및 규격 산출 조건이 다르다.
- RCA 결과는 원인 조사 후보이며 확정된 근본 원인이 아니다.
- 실제 Agent 연동 및 호환성 검증은 추가로 필요하다.

최종 판단 및 조치 결정은 담당자의 검토를 통해 이루어져야 한다.