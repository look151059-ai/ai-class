from __future__ import annotations

import os
import sys
import warnings
import webbrowser
from pathlib import Path

import pandas as pd
import streamlit as st

from ml_utils import load_pickle, predict


APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"

# Keep the UI clean; users can still open logs if needed.
warnings.filterwarnings("ignore", category=UserWarning, message="X has feature names,*")
try:
    from sklearn.exceptions import InconsistentVersionWarning  # type: ignore

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass


IRIS_CLASS_NAMES = {
    "0": "setosa",
    "1": "versicolor",
    "2": "virginica",
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def _inject_css() -> None:
    st.markdown(
        """
<style>
  :root{
    --bg1:#0b1220;
    --bg2:#101b2f;
    --text:#eaf0ff;
    --muted:rgba(234,240,255,.70);
    --border:rgba(234,240,255,.12);
  }
  .stApp{
    background:
      radial-gradient(1200px 700px at 10% 10%, rgba(45,212,191,.16), transparent 55%),
      radial-gradient(900px 600px at 85% 25%, rgba(96,165,250,.18), transparent 60%),
      linear-gradient(180deg, var(--bg1), var(--bg2));
    color: var(--text);
  }
  h1, h2, h3, p, label, div { color: var(--text) !important; }
  .muted{ color: var(--muted); }
  .card{
    background: linear-gradient(180deg, rgba(255,255,255,.04), rgba(255,255,255,.02));
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 16px 16px;
    box-shadow: 0 18px 60px rgba(0,0,0,.35);
  }
  .badge{
    display:inline-block;
    padding: 8px 10px;
    border-radius: 999px;
    border:1px solid var(--border);
    background: rgba(45,212,191,.12);
    font-weight: 700;
    letter-spacing: .4px;
  }
  .kbd{
    display:inline-block;
    padding:2px 8px;
    border-radius:8px;
    border:1px solid var(--border);
    background: rgba(255,255,255,.05);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.88rem;
  }
  .hr{
    height:1px; background: var(--border); margin: 14px 0;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def _pick_scaler_path() -> Path | None:
    # Prefer user's downloaded scaler_iris.joblib
    candidates = [
        APP_DIR / "scaler_iris.joblib",
        MODELS_DIR / "scaler_iris.joblib",
        MODELS_DIR / "scaler.joblib",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _model_map() -> dict[str, Path]:
    return {
        "KNN": MODELS_DIR / "knn.joblib",
        "羅吉斯迴歸": MODELS_DIR / "logistic_regression.joblib",
        "高斯貝葉斯": MODELS_DIR / "gaussian_naive_bayes.joblib",
        "XGBoost": MODELS_DIR / "xgboost.joblib",
    }


def _iris_sliders() -> dict[str, float]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Iris 特徵輸入")
    st.markdown('<p class="muted">用滑桿輸入 4 個特徵，按下右側按鈕進行預測。</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        sepal_length = st.slider("sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
        petal_length = st.slider("petal length (cm)", 1.0, 7.0, 4.0, 0.1)
    with c2:
        sepal_width = st.slider("sepal width (cm)", 2.0, 4.6, 3.0, 0.1)
        petal_width = st.slider("petal width (cm)", 0.1, 2.6, 1.2, 0.1)

    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "sepal length (cm)": float(sepal_length),
        "sepal width (cm)": float(sepal_width),
        "petal length (cm)": float(petal_length),
        "petal width (cm)": float(petal_width),
    }


def _format_pred_label(x) -> str:
    if x in IRIS_CLASS_NAMES:
        return str(IRIS_CLASS_NAMES[x])
    sx = str(x)
    return IRIS_CLASS_NAMES.get(sx, sx)


def main() -> None:
    st.set_page_config(page_title="Iris 分類預測", layout="wide")
    _inject_css()

    st.markdown(
        """
<div class="card">
  <h1 style="margin:0">Iris 分類模型預測</h1>
  <p class="muted" style="margin:6px 0 0 0">
    下拉選模型 + 4 個特徵滑桿，並使用 <span class="kbd">scaler_iris.joblib</span> 做標準化。
  </p>
</div>
""",
        unsafe_allow_html=True,
    )

    model_paths = _model_map()

    with st.sidebar:
        st.header("模型設定")
        model_label = st.selectbox("分類模型", options=list(model_paths.keys()), index=0)
        st.caption(f"檔案：{model_paths[model_label]}")

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.subheader("Scaler")

        scaler_path = _pick_scaler_path()
        use_scaler = st.checkbox("套用標準化 (scaler)", value=True)

        if scaler_path is None:
            st.warning("找不到 scaler_iris.joblib / scaler.joblib。你可以放在專案根目錄或 models/。")
        else:
            st.caption(f"使用：{scaler_path}")

        uploaded_scaler = st.file_uploader("或上傳 scaler_iris.joblib", type=["joblib"])
        st.caption("優先順序：上傳檔 > 專案根目錄 scaler_iris.joblib > models/scaler_iris.joblib > models/scaler.joblib")

    model_path = model_paths[model_label]
    if not model_path.exists():
        st.error(f"找不到模型檔：{model_path}")
        st.stop()

    model = load_pickle(model_path)

    scaler = None
    scaler_used = None
    if use_scaler:
        if uploaded_scaler is not None:
            # Streamlit uploaded file is a file-like object; persist temporarily then load.
            tmp = APP_DIR / ".tmp_uploaded_scaler.joblib"
            tmp.write_bytes(uploaded_scaler.getvalue())
            scaler = load_pickle(tmp)
            scaler_used = "uploaded"
        elif scaler_path is not None:
            scaler = load_pickle(scaler_path)
            scaler_used = str(scaler_path)

    left, right = st.columns([1.1, 0.9])

    with left:
        features = _iris_sliders()

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("預測")
        st.markdown(
            f'<p class="muted">模型：<span class="badge">{model_label}</span></p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="muted">Scaler：<span class="kbd">{("關閉" if (not use_scaler) else (scaler_used or "找不到"))}</span></p>',
            unsafe_allow_html=True,
        )

        if st.button("開始預測", type="primary", use_container_width=True):
            try:
                X = pd.DataFrame([features])
                out = predict(
                    model,
                    X,
                    scaler=scaler,
                    feature_names=list(features.keys()),
                    n_features=4,
                )
            except Exception as e:
                st.error(f"預測失敗：{type(e).__name__}: {e}")
            else:
                raw_pred = out["pred"].iloc[0]
                pred_name = _format_pred_label(raw_pred)
                st.markdown(
                    f'<p style="margin:10px 0 0 0">預測結果：<span class="badge">{pred_name}</span></p>',
                    unsafe_allow_html=True,
                )

                proba_cols = [c for c in out.columns if c.startswith("proba_")]
                if proba_cols:
                    st.markdown('<p class="muted" style="margin:8px 0 0 0">類別機率</p>', unsafe_allow_html=True)
                    proba = (
                        out[proba_cols]
                        .iloc[0]
                        .rename(lambda c: c.replace("proba_", ""))
                        .rename(_format_pred_label)
                        .astype(float)
                        .sort_values(ascending=False)
                    )
                    st.bar_chart(proba, height=220)

                with st.expander("查看原始輸出"):
                    st.dataframe(out, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("提示：你可以直接執行 `python main.py`，它會自動啟動 Streamlit 並印出可點網址。")


def _running_in_streamlit() -> bool:
    # When executed by `streamlit run`, a ScriptRunContext is available.
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None
    except Exception:
        return False


def _maybe_show_url_popup(url: str) -> None:
    """
    Some IDEs run Python without showing a real terminal. In that case, printing
    the URL is easy to miss. On Windows, optionally show a small dialog.

    Disable via: STREAMLIT_URL_POPUP=0
    """

    if os.name != "nt":
        return

    flag = os.environ.get("STREAMLIT_URL_POPUP", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return

    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(
            None,
            f"Streamlit 已啟動\n\n網址：{url}\n\n（也已寫入 STREAMLIT_URL.txt）",
            "Streamlit URL",
            0x40,  # MB_ICONINFORMATION
        )
    except Exception:
        # If the dialog fails, fall back to stdout only.
        return


if __name__ == "__main__":
    if _running_in_streamlit():
        main()
    else:
        # Allow: `python main.py` and still get the Streamlit app + clickable URL.
        from streamlit.web import cli as stcli  # type: ignore

        port = os.environ.get("STREAMLIT_PORT", "8501")
        addr = os.environ.get("STREAMLIT_ADDR", "localhost")
        url = f"http://{addr}:{port}"
        print(f"Local URL: {url}", flush=True)
        # Also persist the URL so IDEs that hide stdout still have an easy place to find it.
        try:
            (APP_DIR / "STREAMLIT_URL.txt").write_text(url + "\n", encoding="utf-8")
        except Exception:
            pass

        _maybe_show_url_popup(url)

        # Avoid the first-run email prompt and make output less confusing in IDE consoles.
        os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
        os.environ.setdefault("STREAMLIT_SERVER_SHOW_EMAIL_PROMPT", "false")
        os.environ.setdefault("STREAMLIT_GLOBAL_SHOW_WARNING_ON_DIRECT_EXECUTION", "false")

        # Optional: try to open the browser if explicitly requested.
        # (Some machines may have browser issues; keep it opt-in.)
        if os.environ.get("STREAMLIT_OPEN_BROWSER", "").strip() in ("1", "true", "True", "yes", "YES"):
            try:
                webbrowser.open(url, new=2)
            except Exception:
                pass

        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--server.address",
            addr,
            "--server.port",
            port,
            "--server.headless",
            "false",
            "--server.showEmailPrompt",
            "false",
        ]
        raise SystemExit(stcli.main())
