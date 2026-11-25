# 수율 이상 추세 모델 — 실험 결과 및 설계 결정

## 1. 통계적 탐지
- Lek Laboratory.csv / Process.csv 각 1,005배치, 제품 코드 25개, 시계열 ZIP 내 제품별 파일 25개.
- 상위 3개 제품 Code 13(131), 17(207), 23(187) 총 525배치. 로그 종료 시각 기준 2020-10-19 02:16 이전 완료 배치 315개로 관리한계 산출, 이후 시작 배치 209개 평가, 경계 배치 1개 제외.
- I 관리도에는 Nelson Rule 1(±3σ), Rule 2(중심선 한쪽 9개 연속), Rule 3(6개 연속 증가/감소)를 적용. MR은 관리상한 초과 확인. ECA 조사에서 I-MR·Nelson 활용을 확인할 수 있으나 개별 규칙별 채택 빈도는 확인되지 않음. ISPE 전문가 기고의 Rule 1~3 우선 적용 제안을 참고하여 초기 규칙 선정.
- 중심선 = 과거 수율 평균, 추정 σ = 평균 이동범위/1.128, I 관리한계 = 중심선 ± 3σ, MR 상한 = 3.267 × 평균 이동범위.

| Code | 기준 배치 | 평가 배치 | LCL | CL | UCL | MR UCL | 이상 배치 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 13 | 70 | 60 | 97.569 | 98.778 | 99.988 | 1.486 | 2 |
| 17 | 119 | 88 | 96.735 | 98.639 | 100.544 | 2.340 | 4 |
| 23 | 126 | 61 | 96.500 | 98.473 | 100.446 | 2.423 | 4 |

평가 209배치 중 Rule 1 3건, Rule 2 3건, Rule 3 1건, MR 4건. 중복 제거 후 10배치(하방 I 신호 5배치, 연속 하방 추세 2배치). Batch 789(Code 17) 수율 87.973%로 I 관리하한 미만 및 MR 상한 초과. Batch 950(Code 17)은 Rule 3 하방 신호. Batch 939(Code 23)은 Rule 2 하방 신호. 실제 제품 규격 또는 일탈 판정은 확인되지 않았으므로 통계적 이상 신호로만 해석한다. 기준 데이터 공정 안정성 미검증, 제품별 레시피·설비 구분 미확인.

## 2. ML 실험
목표: 최종 타정 수율 회귀 및 하방 이상 신호·추세 직접 분류. 315/209 시간순 분할. 시작 전 원료 특성, 제품 코드, 배치 크기 및 과거 수율을 사용한다는 가정. 초기 30분 센서 평균/표준편차를 추가하는 실험도 수행. 최종 수율, 전체 공정 집계 및 최종 품질 정보는 입력에서 제외. 원료 및 과거 배치 수율의 실제 기록 공개 시점은 확인되지 않았으며 일부 배치가 시간상 겹친다.

| 모델 | 시작 전 MAE | 초기 30분 MAE | 초기 30분 RMSE |
|---|---:|---:|---:|
| 전체 중앙값 | 0.531 | 0.531 | 0.991 |
| 제품별 중앙값 | 0.533 | 0.533 | 1.000 |
| 직전 수율 | 0.576 | 0.576 | 0.973 |
| Random Forest | 0.513 | 0.460 | 0.895 |
| Gradient Boosting | 0.557 | 0.487 | 0.922 |

단위 %p. 초기 30분 RF의 MAE는 전체 중앙값 대비 약 13.3% 감소. 그러나 RF 예측 수율과 기존 Nelson 규칙을 결합한 하방 이상 예측은 TP 1, FP 3, FN 4, TN 201 (Precision 25%, Recall 20%). 연속 하방 추세는 2건 중 1건 검출, 오경보 2건. 직접 분류에서도 초기 30분 RF/GB는 하방 이상 5건을 모두 놓쳤음. 시작 전 GB는 하방 이상 5건 중 1건, 연속 하방 추세 2건 중 1건을 탐지했으나 소표본으로 일반화 불가. 95% 미만 가상 기준은 학습 2건, 평가 1건이며 RF/GB 회귀 예측은 해당 1건을 포착하지 못함. 실제 규격 아님.

## 3. 설계 결정 제안
- MVP: 배치 종료 후 I-MR + Nelson Rule 1~3 기반 이상 신호 감지. 관리한계 이탈·연속 추세·MR 급변을 유형별로 표시하고 알림에 발동 규칙·수율·관리한계·배치 ID를 포함. 상방·하방을 구분. 반복 발동은 중복 알림 방지 정책 별도 설계.
- 규격 일탈: 실제 제품/공정별 승인 기준 등록 시 실측값과 직접 비교. 기준 미등록 시 판정 불가 표시. SPC 신호를 공식 일탈로 자동 분류하지 않음.
- ML: 초기 30분 Random Forest 수율 예측을 PoC/참고 정보로 유지. 이상 추세와 일탈 모두 자동 예측 알림의 정식 운영은 보류. 충분한 양성 사례, 예측 시점별 데이터 가용성, 레시피/설비 단위 분리, 검증용 규격·일탈 라벨, 시간순 반복 검증 및 오경보 허용 수준 확보 후 재평가.
- 오늘 실험으로 통계적 탐지의 구현 가능성과 ML 조기 예측의 한계를 확인함. GMP 적합성 또는 실제 일탈 검출 정확도를 입증한 것은 아님.

## 참고 문헌
- ECA Validation Group, ECA Survey Results II: How is Statistical Process Control used in the Pharmaceutical Industry? https://www.qp-association.eu/news/news-detail/eca-survey-results-ii-how-is-statistical-process-control-used-in-the-pharmaceutical-industry.html
- ISPE Pharmaceutical Engineering, Stop Using All Eight Nelson Rules in SPC Control Charts Used in CPV Monitoring. https://ispe.org/pharmaceutical-engineering/ispeak/stop-using-all-eight-nelson-rules-statistical-process-control-spc
- Žagar & Mihelič (2022), Big data collection in pharmaceutical manufacturing and its use for product quality predictions. https://pmc.ncbi.nlm.nih.gov/articles/PMC8943063/
- NIST, Individuals Control Charts. https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm
