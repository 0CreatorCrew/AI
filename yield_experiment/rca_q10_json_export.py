from pathlib import Path
import hashlib
import json
import pandas as pd

OUT = Path("results/spec_all_products_q10")
CONFIG = Path(
    "config/yield_specifications_all_products_q10.json"
)

config = json.loads(
    CONFIG.read_text(encoding="utf-8")
)

SCENARIOS = [
    (17, 789, 120, 180),
    (15, 799, 0, 90)
]


# Convert CSV records to JSON-safe objects.
# Missing values become null instead of NaN.

def records(df):
    return json.loads(
        df.to_json(
            orient="records",
            double_precision=15
        )
    )


def one(df):
    assert len(df) == 1
    return records(df)[0]


def indexed(df, key):
    assert df[key].is_unique
    return {
        row[key]: row
        for row in records(df)
    }


def select_case(df, code, batch):
    return one(df[
        (df["code"] == code) &
        (df["batch"] == batch)
    ])


def hypothesis(
    priority, title, evidence,
    counterevidence, checks
):
    return {
        "id": f"H{priority}",
        "priority": priority,
        "title": title,
        "status": "unconfirmed",
        "evidence_paths": evidence,
        "counterevidence": counterevidence,
        "required_checks": checks
    }


# Investigation priorities, not confirmed causes

HYPOTHESES = {
    17: [
        hypothesis(
            1,
            "재가동 과정의 압축력 불안정",
            [
                "/timeseries/focus_comparison",
                "/timeseries/top_running_segments"
            ],
            [
                "과거 배치에서도 재가동 직후 "
                "폐기 기록 집중 현상이 나타남"
            ],
            [
                "재가동 시 압축력 설정 및 변동 확인",
                "설비 알람 및 작업자 기록 확인"
            ]
        ),
        hypothesis(
            2,
            "평소와 다른 주압축력 조건",
            [
                "/feature_comparison/all_features/"
                "main_CompForce mean"
            ],
            [
                "과거 관측 범위 이탈은 "
                "승인된 공정조건 위반을 의미하지 않음"
            ],
            [
                "승인된 압축력 설정 범위 확인",
                "레시피 및 설정 변경 이력 확인"
            ]
        ),
        hypothesis(
            3,
            "타정 속도 설정 및 변경 조건",
            [
                "/feature_comparison/all_features/"
                "tbl_speed_mean",
                "/timeseries/top_running_segments"
            ],
            [
                "동일한 타정 속도 중앙값 300에서도 "
                "폐기 증가량이 적은 구간이 존재함"
            ],
            [
                "속도 변경 및 재가동 이력 확인",
                "속도와 압축력의 동시 변화 확인"
            ]
        ),
        hypothesis(
            4,
            "배출 과정의 이상 가능성",
            [
                "/feature_comparison/all_features/"
                "ejection_mean",
                "/feature_comparison/all_features/"
                "ejection_max"
            ],
            [],
            [
                "펀치와 다이 상태 확인",
                "윤활 및 배출 과정 점검"
            ]
        ),
        hypothesis(
            5,
            "생산 중 정지 및 대기 과정",
            [
                "/timeseries/historical_comparison/"
                "zero_speed_record_pct"
            ],
            [
                "정지 횟수와 최장 정지 시간은 "
                "과거 95% 분위수 이내"
            ],
            [
                "정지 사유 및 대기 기록 확인",
                "정지 전후 공정조건 확인"
            ]
        )
    ],
    15: [
        hypothesis(
            1,
            "반복적인 정지 및 재가동",
            [
                "/timeseries/historical_comparison/"
                "internal_stop_count",
                "/timeseries/historical_comparison/"
                "zero_speed_record_pct"
            ],
            [
                "정지가 설비 고장 때문인지 "
                "계획된 작업인지 확인되지 않음"
            ],
            [
                "정지 사유 및 설비 알람 확인",
                "재가동 절차 확인"
            ]
        ),
        hypothesis(
            2,
            "주압축력 변동성 증가",
            [
                "/feature_comparison/all_features/"
                "main_CompForce_sd",
                "/timeseries/focus_comparison"
            ],
            [],
            [
                "압축력 제어 및 설정 변경 확인",
                "생산 초기 압축력 안정화 기록 확인"
            ]
        ),
        hypothesis(
            3,
            "타정 속도 변경 및 안정화 문제",
            [
                "/feature_comparison/all_features/"
                "tbl_speed_change",
                "/timeseries/focus_comparison"
            ],
            [],
            [
                "속도 변경 사유 확인",
                "생산 초기 운전조건 점검"
            ]
        ),
        hypothesis(
            4,
            "생산 초기 배출 과정의 불안정",
            [
                "/timeseries/focus_comparison"
            ],
            [
                "배치 평균 배출력은 "
                "과거 관측 범위 이내"
            ],
            [
                "초기 배출력 변동 원인 확인",
                "윤활 및 금형 상태 확인"
            ]
        ),
        hypothesis(
            5,
            "장시간 생산 중단 및 재개 조건",
            [
                "/timeseries/historical_comparison/"
                "longest_stop_span_min"
            ],
            [
                "과거에도 더 긴 정지 구간이 존재함",
                "초기 폐기 기록 집중은 "
                "과거 배치에서도 일반적으로 나타남"
            ],
            [
                "장시간 정지 사유 확인",
                "재개 전후 공정조건 및 "
                "중간제품 보관 기록 확인"
            ]
        )
    ]
}


for code, batch, focus_start, focus_end in SCENARIOS:

    sources = [CONFIG]

    def load(name):
        path = OUT / name
        assert path.exists(), path
        sources.append(path)
        return pd.read_csv(path)

    # 1. Load specification and SPC results

    detection = select_case(
        load("scenario_candidates_q10.csv"),
        code, batch
    )

    product = config["products"][str(code)]

    assert detection["outside_q10"] is True
    assert detection["i_low"] is True
    assert detection["mr_signal"] is True

    actual = detection["tbl_yield"]
    lsl = product["lsl"]
    usl = product["usl"]

    assert actual < lsl

    # 2. Load reference batches

    ref_summary = select_case(
        load("rca_reference_summary.csv"),
        code, batch
    )

    ref = load(f"reference_code_{code}.csv")

    assert len(ref) == ref_summary["reference_n"]
    assert (ref["code"] == code).all()
    assert ref["tbl_yield"].between(lsl, usl).all()

    features = load(
        f"rca_comparison_code_{code}.csv"
    )

    process = features[
        features["category"] == "process"
    ]

    # 3. Load timeseries analysis results

    ts_summary = select_case(
        load("timeseries_summary.csv"),
        code, batch
    )

    waste_summary = select_case(
        load("waste_analysis_summary.csv"),
        code, batch
    )

    focus = load(
        f"focus_comparison_code_{code}.csv"
    )

    segments = load(
        f"operating_segments_code_{code}.csv"
    )

    windows = load(
        f"waste_windows_code_{code}.csv"
    )

    historical = load(
        f"reference_ts_comparison_code_{code}.csv"
    )

    reference_ts = load(
        f"reference_timeseries_code_{code}.csv"
    )

    target_ts = one(
        reference_ts[
            reference_ts["group"] == "target"
        ]
    )

    hist_ref = reference_ts[
        reference_ts["group"] == "reference"
    ]

    running_segments = segments[
        segments["running"] == True
    ]

    focus_row = one(
        focus[focus["period"] == "Focus window"]
    )

    # Check consistency between analysis files

    assert abs(
        windows["waste_increment"].sum()
        - waste_summary["total_positive_increments"]
    ) < 0.0001

    assert abs(
        segments["waste_increment"].sum()
        - waste_summary["total_positive_increments"]
    ) < 0.0001

    assert abs(
        focus["waste_increment"].sum()
        - waste_summary["total_positive_increments"]
    ) < 0.0001

    # 4. Graph paths

    graphs = [
        OUT / f"imr_code_{code}_batch_{batch}.png",
        OUT / f"rca_boxplot_code_{code}.png",
        OUT / f"rca_timeseries_code_{code}_full.png",
        OUT / f"rca_timeseries_code_{code}_focus.png"
    ]

    for path in graphs:
        assert path.exists(), path

    # 5. Build scenario JSON

    result = {
        "schema_version": "0.2.0",
        "case_id": f"code{code}_batch{batch}",
        "scenario_type": "retrospective_demo",

        "batch_info": {
            "code": code,
            "batch": batch,
            "start": ts_summary["start"],
            "end": ts_summary["end"],
            "actual_yield_pct": actual,
            "analysis_phase": "test"
        },

        "specification": {
            "status": "research_poc_only",
            "approved_manufacturing_spec": False,
            "configuration": str(CONFIG),
            "configuration_version": config["version"],
            "lsl_pct": lsl,
            "usl_pct": usl,
            "actual_pct": actual,
            "spec_outside": True,
            "gap_below_lsl_pp": round(lsl - actual, 6),
            "actual_gmp_oos": None,
            "newly_detected":
                detection["newly_detected"]
        },

        "spc": {
            "signals": {
                "i_low": detection["i_low"],
                "i_high": detection["i_high"],
                "mr_signal": detection["mr_signal"]
            },
            "limits": {
                "i_lcl": detection["i_lcl"],
                "i_ucl": detection["i_ucl"],
                "mr_ucl": detection["mr_ucl"]
            },
            "observations": {
                "mr": detection["mr"],
                "yield_z_score": detection["z_score"],
                "training_mean": detection["mean"],
                "training_std": detection["std"]
            },
            "interpretation": (
                "I-MR signals support investigation "
                "but do not establish root cause."
            )
        },

        "reference": {
            "selection": (
                "Same product, historical training batches "
                "within q10 PoC specifications"
            ),
            "count": len(ref),
            "summary": ref_summary,
            "timeseries_reference_count": len(hist_ref),
            "valid_waste_reference_count": int(
                hist_ref["counter_valid"].sum()
            ),
            "approved_normal_status_verified": False
        },

        "feature_comparison": {
            "compared_features": len(features),
            "process_feature_count": len(process),
            "outside_reference_range_count": int(
                process["outside_range"].sum()
            ),
            "process_iqr_outlier_count": int(
                (process["iqr_outlier"] == True).sum()
            ),
            "process_abs_z_ge_3_count": int(
                (process["abs_z_score"] >= 3).sum()
            ),
            "all_features": indexed(
                features, "feature"
            ),
            "warning": (
                "Descriptive statistics are not "
                "causal evidence or approved limits."
            )
        },

        "timeseries": {
            "timestamp_resolution": "minute",
            "summary": ts_summary,
            "target_statistics": target_ts,
            "waste_summary": waste_summary,

            "priority_window": {
                "start_elapsed_min": focus_start,
                "end_elapsed_min_exclusive": focus_end,
                "waste_increment":
                    focus_row["waste_increment"],
                "waste_share_pct":
                    focus_row["waste_share_pct"]
            },

            "focus_comparison": indexed(
                focus, "period"
            ),

            "historical_comparison": indexed(
                historical, "metric"
            ),

            "top_running_segments": records(
                running_segments.sort_values(
                    "waste_increment",
                    ascending=False
                ).head(10)
            ),

            "window_summary": records(windows),

            "counter_warning": (
                "Counter increments are not verified "
                "physical waste event timestamps. "
                "Do not reconstruct final yield from counters."
            )
        },

        "ml_prediction": {
            "status": "not_evaluated_in_q10_scenario",
            "included": False
        },

        "shap": {
            "status": "not_evaluated_in_q10_scenario",
            "included": False
        },

        "uncertainty": {
            "status": "not_evaluated_in_q10_scenario",
            "included": False
        },

        "hypotheses": HYPOTHESES[code],

        "references": [],

        "limitations": {
            "confirmed_root_cause": None,
            "causality_established": False,
            "human_review_required": True,
            "approved_specifications_available": False,
            "autonomous_recipe_change_allowed": False,
            "autonomous_batch_release_allowed": False,
            "automatic_capa_execution_allowed": False,

            "warnings": [
                "Historical reference ranges are not "
                "approved process limits.",
                "Zero-speed records do not necessarily "
                "indicate equipment failure.",
                "Restart-associated waste concentration "
                "also occurs in reference batches.",
                "Missing data reduce reference sample sizes.",
                "The q10 limit is a retrospective PoC criterion.",
                "Statistical significance and causality "
                "have not been established."
            ]
        },

        "artifacts": {
            "graphs": [str(path) for path in graphs]
        },

        "provenance": {
            str(path): {
                "sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            }
            for path in sources + graphs
        }
    }

    # 6. Validate hypothesis evidence paths

    for h in result["hypotheses"]:

        assert h["status"] == "unconfirmed"

        for pointer in h["evidence_paths"]:

            node = result

            for key in pointer.strip("/").split("/"):
                node = node[key]

    # 7. Save and reload JSON

    path = OUT / (
        f"rca_agent_code{code}_batch{batch}.json"
    )

    path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            allow_nan=False
        ),
        encoding="utf-8"
    )

    saved = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert saved == result

    print(f"\n=== CODE {code} / BATCH {batch} ===")
    print("Schema version:", saved["schema_version"])
    print("Actual yield:", actual)
    print("LSL:", lsl)
    print("USL:", usl)
    print("Gap below LSL:", round(lsl - actual, 6))
    print("Reference batches:", len(ref))
    print("Compared features:", len(features))
    print("Process features:", len(process))
    print("Timeseries reference batches:", len(hist_ref))
    print(
        "Valid waste reference batches:",
        saved["reference"]["valid_waste_reference_count"]
    )
    print("Hypotheses:", len(saved["hypotheses"]))
    print("Evidence paths: VALID")
    print("Graphs:", len(graphs))
    print("Provenance files:", len(saved["provenance"]))
    print("Confirmed root cause:", None)
    print("Human review required:", True)
    print("Saved:", path)
    print("File size:", path.stat().st_size, "bytes")

print("\nALL SCENARIO JSON FILES GENERATED")
