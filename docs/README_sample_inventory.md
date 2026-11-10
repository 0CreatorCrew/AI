# SCRUM-22 sample inventory

- `data/`: 정제된 4개 CSV (제목 행 제거, 빈 `cause_type` 헤더 복구; 레코드 값 수정 없음)
- `notebooks/sample_EDA.ipynb`: EDA 실습/검증 노트북
- `docs/data_inventory.md`: 1차 인벤토리, 데이터 구조, 계산식 후보, 이슈/미확정 항목
- `docs/data_dictionary.csv`: ② 필드 명세에서 추출한 컬럼 설명
- `docs/data_quality_findings.csv`: 점검 결과 요약
- `docs/reference/`: ①~③ 문서 시트 사본

실행: 프로젝트 루트에서 `jupyter notebook notebooks/sample_EDA.ipynb` 실행 후 Run All. 필요 패키지: `pandas`, `numpy`, `matplotlib`, `jupyter`.

샘플 데이터의 정의가 불명확하거나 참조 간 값이 서로 다른 부분은 변경하지 않고 문서화했습니다.
