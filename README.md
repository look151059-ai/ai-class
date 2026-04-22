## Streamlit 分類模型預測 APP

這個專案提供一個可直接使用的 Streamlit Web APP，用來載入本機已訓練好的分類模型（`models/*.joblib`）並進行：

- 單筆輸入預測
- 上傳 CSV 批次預測
- 顯示每一類別機率（若模型支援 `predict_proba`）
- 下載預測結果 CSV

> 模型會用 `joblib.load` 載入（必要時才回退到 `pickle`）。

### 1) 建立虛擬環境並安裝套件（建議）

```powershell
cd d:\VS\20260422
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

也可以用腳本：

```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

### 2) 啟動 APP（Iris 4 特徵滑桿 + 下拉選模型）

```powershell
python -m streamlit run main.py
```

更省事的方式（只要貼到終端機或直接執行）：

```powershell
python main.py
```

或使用一鍵腳本：

```powershell
.\run_app.ps1
```

（可選）嘗試自動開瀏覽器：

```powershell
.\run_app.ps1 -OpenBrowser
```

## 上傳到 GitHub

1. 先安裝 Git for Windows，並重新開啟 VS Code/終端機
2. 到 GitHub 建立一個空的 repo（不要勾選新增 README/.gitignore，因為本專案已經有）
3. 在專案根目錄執行：

```powershell
.\scripts\publish_github.ps1 -RepoName "你的repo名稱"
```

### 3) 另一個通用版 APP（單筆 + CSV 批次）

```powershell
python -m pip install -r requirements.txt
```

```powershell
python -m streamlit run streamlit_app.py
```

### 4) 模型與 scaler 放置位置

- 分類模型：`models/<something>.joblib`
- (可選) scaler：`models/scaler.joblib`

Iris 專用 scaler（優先使用你自己下載的）：

- 專案根目錄：`scaler_iris.joblib`
- 或 `models/scaler_iris.joblib`

APP 會在側邊欄讓你選擇要用的模型，以及是否套用 scaler。
