from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

from ml_utils import (
    describe_estimator,
    infer_feature_names,
    infer_n_features,
    list_model_files,
    load_pickle,
    predict,
)


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

# Keep the UI clean; users can still open logs if needed.
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names,*")
try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


@st.cache_resource
def load_any(path: Path):
    return load_pickle(path)


def _single_input_form(feature_names: list[str] | None, n_features: int | None) -> dict[str, float]:
    if feature_names is not None:
        cols = feature_names
    elif n_features is not None:
        cols = [f"f{i}" for i in range(n_features)]
    else:
        cols = ["f0", "f1", "f2", "f3"]

    values: dict[str, float] = {}
    for c in cols:
        values[c] = st.number_input(c, value=0.0, format="%.6f")
    return values


def main() -> None:
    st.set_page_config(page_title="分類模型預測", layout="wide")
    st.title("分類模型預測 APP (Streamlit)")

    st.caption("支援：單筆輸入、CSV 批次預測、機率輸出與下載結果。")

    model_files = list_model_files(MODELS_DIR)
    if not model_files:
        st.error(f"找不到模型檔：{MODELS_DIR}/*.joblib")
        st.stop()

    model_choices = [p.name for p in model_files if p.name != "scaler.joblib"]
    scaler_path = MODELS_DIR / "scaler.joblib"
    has_scaler = scaler_path.exists()

    with st.sidebar:
        st.header("設定")
        model_name = st.selectbox("模型檔", options=model_choices, index=0)
        use_scaler = st.checkbox("套用 scaler", value=has_scaler, disabled=not has_scaler)
        st.divider()
        st.caption(f"模型目錄：{MODELS_DIR}")

    model_path = MODELS_DIR / model_name
    model = load_any(model_path)
    scaler = load_any(scaler_path) if (use_scaler and has_scaler) else None

    feature_names = infer_feature_names(model, scaler)
    n_features = infer_n_features(model, scaler)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("模型資訊")
        st.write({"path": str(model_path), "type": describe_estimator(model)})
        classes = getattr(model, "classes_", None)
        if classes is not None:
            st.write({"classes_": [str(x) for x in list(classes)]})
        st.write({"feature_names_in_": feature_names, "n_features_in_": n_features})
        if scaler is not None:
            st.write({"scaler": describe_estimator(scaler)})

    with right:
        st.subheader("輸入方式")
        if feature_names is None and n_features is None:
            n_features = int(
                st.number_input(
                    "無法自動推斷特徵數，請手動指定 n_features",
                    min_value=1,
                    max_value=500,
                    value=4,
                    step=1,
                )
            )

        tab_single, tab_csv = st.tabs(["單筆", "CSV 批次"])

        with tab_single:
            with st.form("single_predict"):
                values = _single_input_form(feature_names, n_features)
                submitted = st.form_submit_button("預測")
            if submitted:
                try:
                    out = predict(
                        model,
                        values,
                        scaler=scaler,
                        feature_names=feature_names,
                        n_features=n_features,
                    )
                except Exception as e:
                    st.error(f"預測失敗：{type(e).__name__}: {e}")
                else:
                    st.dataframe(out, use_container_width=True)

        with tab_csv:
            uploaded = st.file_uploader("上傳 CSV", type=["csv"])
            target_col = st.text_input("（可選）真實標籤欄位名，用於計算 accuracy", value="")
            if uploaded is not None:
                try:
                    df = pd.read_csv(uploaded)
                except Exception as e:
                    st.error(f"讀取 CSV 失敗：{type(e).__name__}: {e}")
                    st.stop()

                st.write("預覽：")
                st.dataframe(df.head(20), use_container_width=True)

                if st.button("開始批次預測"):
                    y_true = None
                    if target_col.strip() and target_col.strip() in df.columns:
                        y_true = df[target_col.strip()].astype(str).reset_index(drop=True)

                    try:
                        out = predict(
                            model,
                            df,
                            scaler=scaler,
                            feature_names=feature_names,
                            n_features=n_features,
                        )
                    except Exception as e:
                        st.error(f"預測失敗：{type(e).__name__}: {e}")
                        st.stop()

                    if y_true is not None:
                        y_pred = out["pred"].astype(str).reset_index(drop=True)
                        acc = (y_true == y_pred).mean()
                        st.info(f"accuracy = {acc:.4f}")

                    st.success(f"完成：{len(out)} 筆")
                    st.dataframe(out, use_container_width=True)

                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "下載預測結果 CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

    st.divider()
    st.caption("提示：如果模型沒有 `feature_names_in_`，CSV 欄位可以用 `f0..f{n-1}` 或提供足夠的數值欄位讓 APP 自動取前 n 個。")


if __name__ == "__main__":
    main()
