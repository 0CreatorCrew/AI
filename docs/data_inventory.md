# SCRUM-22 — 제공 샘플 원본·데이터 사전 1차 인벤토리

작성 근거: 사용자가 제공한 7개 CSV, Jira SCRUM-22, GitHub Issue #1 (`https://github.com/0CreatorCrew/AI/issues/1`).  
범위: 1차 샘플 검토(도메인별 필드 확장 및 확정 작업은 제외).

## 1. 원본 목록과 정제 범위

| 원본 시트 | 종류 | 처리 결과 |
|---|---|---|
| ① 개요 | 설명 문서 | `docs/reference/overview.csv` 보관 |
| ② 필드 명세 | 데이터 사전 | `docs/reference/field_spec.csv` 보관, `docs/data_dictionary.csv` 추출 |
| ③ 시나리오 | 업무 흐름·SQL 참고 | `docs/reference/scenario.csv` 보관 |
| ④ yield_monitoring | 실제 수율 측정 데이터 | `data/yield_monitoring.csv` (100행 × 17열) |
| ⑤ process_condition | 공정 조건 데이터 | `data/process_condition.csv` (100행 × 17열) |
| ⑥ root_cause_analysis | 원인 후보 데이터 | `data/root_cause_analysis.csv` (100행 × 16열) |
| ⑦ improvement_tracking | 개선 조치 이력 데이터 | `data/improvement_tracking.csv` (100행 × 19열) |

정제 규칙: 모든 원본 시트의 **첫 번째 제목 행을 제거하고 두 번째 행을 컬럼명으로 지정**했다. 특히 ⑥ 데이터의 다섯 번째 열은 원본 두 번째 행이 빈 칸이고 첫 번째 행에 `cause_type`이라고 적혀 있어, ② 필드 명세와 대조하여 `cause_type`으로 복구했다. **레코드 값은 변경하지 않았다.**

## 2. 테이블·키·시점

| 테이블 | PK | 주요 연결키(FK) | 시점 컬럼 | 건수 |
|---|---|---|---|---:|
| yield_monitoring | yield_id | (기준 테이블) | measured_at | 100 |
| process_condition | condition_id | yield_id → yield_monitoring | recorded_at | 100 |
| root_cause_analysis | rca_id | yield_id → yield_monitoring | analyzed_at | 100 |
| improvement_tracking | improve_id | rca_id → root_cause_analysis, yield_id → yield_monitoring | created_at, start_dt, due_dt, completed_dt | 100 |

총 69개 컬럼(17+17+16+19). 개별 컬럼 설명·타입은 `docs/data_dictionary.csv` 참고.

예상 관계(스펙 기준):

```text
yield_monitoring (yield_id)
  ├── process_condition (condition_id, yield_id)
  └── root_cause_analysis (rca_id, yield_id)
        └── improvement_tracking (improve_id, rca_id, yield_id)
```

참고: FK는 형식상 참조할 수 있으나 **장비·공정값의 의미상 일관성이 낮다**. 다대일/일대다 관계가 혼재하므로 여러 테이블을 동시에 JOIN하면 행이 증가할 수 있음. JOIN 후 행 수 확인 필수.

## 3. 수율 산식 및 정의 후보(확정 전)

- `actual_yield` 후보: `output_qty / input_qty × 100` (%). **22/100건**은 표시값과 계산값 차이가 **0.15%p 초과**. 비율 계산·반올림 규칙 또는 샘플 데이터 확인 필요.
- `yield_gap`: `target_yield - actual_yield` (%p). **100/100건 일치**(허용 오차 0.11%p 이내).
- 수량 보존: `input_qty = output_qty + defect_qty + loss_qty`. **100/100건 일치**.
- ② 필드 명세의 `yield_status` 구간: gap ≤1 정상; 1<gap≤3 주의; 3<gap≤7 저하; gap>7 심각 (제공 데이터는 100/100건 일치).
- `yield_improvement` 후보: 개선 후 실제 수율 − 개선 전 실제 수율 (%p). 문서의 '개선량' 표현을 바탕으로 한 가설이므로 확정 전 검증 필요.

## 4. 1차 품질 점검 결과

| 점검 사항 | 결과 | 해석·후속 확인 |
|---|---:|---|
| 네 테이블의 PK 중복 | 각 0건 | 고유 식별자 형식은 정상 |
| FK로 참조하는 ID 자체의 미존재 | 0건 | 참조 대상 ID는 모두 존재 |
| process_condition의 equipment_id / process_id 불일치 | 각 90/100건 | 같은 yield_id의 수율 레코드와 다름 |
| root_cause_analysis의 equipment_id / process_id 불일치 | 93/100건, 89/100건 | 같은 yield_id의 수율 레코드와 다름 |
| improvement_tracking의 equipment_id / process_id 불일치 | 91/100건, 86/100건 | 같은 yield_id의 수율 레코드와 다름 |
| improvement_tracking의 rca_id에 연결된 yield_id 불일치 | **100/100건** | 두 연결 키의 의미/생성 규칙 확인 필요 |
| actual_yield vs 수량 기반 산식 차이 > 0.15%p | 22/100건 | 계산/반올림 정의 확인 필요 |
| process_condition 기록 시각이 수율 측정 시각보다 늦음 | 58/100건 | 시점 의미 확인 필요 |
| RCA 분석 시각이 수율 측정 시각보다 빠름 | 48/100건 | 선행 분석 가능한지 확인 필요 |
| 개선 후 실제 수율(actual_yield_after) 미기록 | 74/100건 | 미완료에 따른 공란 가능; 현재 완료 건수 26 |

※ 위 불일치는 *오류로 확정*하지 않았다. 사용자 제공 샘플의 관계 정의·정합성 확인이 필요하다는 뜻이다. **원본 값을 임의로 수정하거나 조인 기준을 추측해서 확정하지 않는다.**

## 5. 의미·정의 확인이 필요한 항목

1. `yield_id`로 묶인 서로 다른 `equipment_id`, `process_id` 중 어떤 값이 기준인지? (샘플 생성상 불일치인지, 서로 다른 맥락의 장비·공정 식별인지)
2. improvement_tracking의 `rca_id`와 `yield_id`가 서로 다른 수율 건을 가리켜도 되는지? 두 FK의 연계 제약은 무엇인지?
3. `actual_yield`는 수량에서 직접 재계산하는 값인지, 별도 측정·반올림 산식인지?
4. `condition_deviation_score`의 계산식, 0~100 정규화 기준, `out_of_spec_yn` 판정 임계값은?
5. `impact_score`, `confidence_level`, `estimated_yield_gain`, `yield_correlation` 산출 근거는? 합성 데이터의 랜덤값인지?
6. `measured_at`/`recorded_at`/`analyzed_at`의 시간대 및 기록 생성 순서를 어떤 규칙으로 보장하는지?
7. 원본 출처(실제 현업/공개 데이터/생성 데이터), 생성 방법, 갱신 주기, 버전, 데이터 보안/사용 범위는?
8. `yield_improvement` 단위가 %p인지 상대증가율 %인지, 개선 전 기준값은 어느 시점 수율인지?

## 6. 제출/검증 가이드

- **data/**: 4개 정제 CSV. 한 줄로 된 유효한 컬럼 헤더, 100 데이터 행씩.
- **notebooks/**: `sample_EDA.ipynb`. 파일 로딩, 4개 테이블별 EDA, 수율/상태/결측/관계 점검, 기본 시각화 및 핵심 결과 포함.
- **docs/**: `data_inventory.md`(본 문서), `data_dictionary.csv`(컬럼 목록), `data_quality_findings.csv`(점검 결과), `reference/`(설명·필드·시나리오 원본).
- 리뷰에서 먼저 확인할 사항: FK 및 시간 정합성, 수율 산식, 출처 구분. 이슈 범위 외의 도메인 확장 필드나 후속 확정은 새 작업으로 이관.

이 문서는 **샘플의 1차 상태를 기록한 작업용 초안**으로, 팀원/데이터 담당자의 확인 전 확정 데이터 사전이 아니다.
