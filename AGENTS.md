# AI 自動下單與回測系統 — Agent 工作規範

## 核心規則（強制遵守）

### 1. 寫完程式必須更新工作進度表
- **檔案路徑**：[`工作進度表.md`](file:///g:/我的雲端硬碟/工具/自動測試/工作進度表.md)
- **執行時機**：每次修改、新增程式碼或完成重要調查/診斷任務後，**必須在「最新紀錄」正下方插入一筆新紀錄**。
- **紀錄格式**：
  ```markdown
  ## YYYY-MM-DD HH:MM — <一句話主題>

  - **狀態**：進行中 / 完成 / 卡住
  - **做了什麼**：
  - **動到的檔案**：
  - **驗證方式**：（單元測試／實際連線測試／終端機輸出結果）
  - **未完成 / 下一步**：
  - **給夥伴的提醒**：
  - **簽名**：Claude、Codex 或 Gemini — YYYY-MM-DD HH:MM
  ```
- **簽名規範**：最新紀錄在最上方，不得覆蓋或刪改過往其他協作者已簽名的歷史紀錄。

### 2. 交易安全性最高原則
- 下單執行前必須通過五道可信度檢核。
- 策略停損為強制品，若無停損設定絕對不可執行。
- 行情模組為純報價接收（Market Data Feed），絕不發送委託下單指令。

### 3. 每次修改程式後必須附上雲端網站更新指令
- 每次修改程式碼並推送至 GitHub (`master` 與 `main`) 後，**回覆使用者時必須貼上包含完整專案 ID (`point-api-512405`) 的 Google Cloud Shell 一鍵更新 Cloud Run 網站指令**：
  ```bash
  cd ~/AI-Trading-WarRoom && git fetch origin master && git reset --hard origin/master && gcloud config set project point-api-512405 && gcloud run deploy ai-trading-warroom --source . --project point-api-512405 --region asia-east1 --allow-unauthenticated
  ```

