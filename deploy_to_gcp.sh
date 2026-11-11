#!/usr/bin/env bash
set -e

# ==========================================================
# Google Cloud Platform (GCP) 一鍵全自動部署腳本
# 專案：AI 智慧看盤與多空決策戰情室 (ai-trading-warroom)
# ==========================================================

PROJECT_ID="point-api-512405"
REGION="asia-east1"   # 台灣彰化機房，台新與台股連線延遲最低
SERVICE_NAME="ai-trading-warroom"
BUCKET_NAME="${PROJECT_ID}-stock-history"

echo "=========================================================="
echo "🚀 開始執行 GCP Cloud Run 一鍵部署 - 專案: ${PROJECT_ID}"
echo "=========================================================="

# 1. 設定 GCP 作用專案
echo "1️⃣ 設定專案為 ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# 2. 開通所需 Google API (Cloud Run, Storage, Cloud Build, Artifact Registry)
echo "2️⃣ 正在確認必要的 Google Cloud API..."
gcloud services enable \
  run.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

# 2.1 設定 Cloud Build 預設服務帳戶權限 (與儲存槽讀寫權限)
echo "🔑 正在配置構建與執行服務帳戶權限..."
PROJECT_NUM=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
COMPUTE_SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"
for ROLE in roles/cloudbuild.builds.builder roles/storage.admin roles/artifactregistry.writer roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="${ROLE}" \
    --condition=None --quiet >/dev/null 2>&1 || true
done

# 3. 檢查並建立 GCS 雲端儲存槽，同步既有台新真實 1m K 線 Parquet
echo "3️⃣ 檢查雲端儲存槽: gs://${BUCKET_NAME}..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET_NAME}" --location="${REGION}"
  echo "✅ 成功建立儲存槽: gs://${BUCKET_NAME}"
else
  echo "ℹ️ 儲存槽 gs://${BUCKET_NAME} 已存在。"
fi

if ls data/taishin_kbars_*.parquet >/dev/null 2>&1; then
  echo "📤 正在將本地台新真實 1m K 線 Parquet 同步至 gs://${BUCKET_NAME}/kbars/ ..."
  gcloud storage cp data/taishin_kbars_*.parquet "gs://${BUCKET_NAME}/kbars/" || true
  echo "✅ K 線快取同步完成！"
fi

# 4. 檢查 env.yaml 環境變數配置檔 (若 Cloud Shell 尚未上傳 env.yaml，自動引導生成)
if [ ! -f "env.yaml" ]; then
  echo "⚠️ 偵測到目錄下尚無 env.yaml，正在為您自動建立..."
  # 嘗試從同專案既有的 taiwan-stock-ai 服務自動帶入 GEMINI_API_KEY
  EXISTING_GEMINI_KEY=$(gcloud run services describe taiwan-stock-ai --region "${REGION}" --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | grep -oP "(?<='GEMINI_API_KEY', 'value': ')[^']+" || true)
  read -r -p "請輸入台新身分證字號 (SOL_USER_ID): " INPUT_SOL_ID
  read -r -s -p "請輸入台新密碼 (SOL_PASSWORD): " INPUT_SOL_PWD
  echo ""
  read -r -s -p "請設定戰情室管理密碼 (WARROOM_ADMIN_PASSWORD，留空則任何訪客都能操作模擬倉): " INPUT_ADMIN_PWD
  echo ""
  if [ -z "${EXISTING_GEMINI_KEY}" ]; then
    read -r -p "請輸入 GEMINI_API_KEY: " EXISTING_GEMINI_KEY
  else
    echo "✅ 已從同專案自動帶入 GEMINI_API_KEY"
  fi
  cat > env.yaml <<EOF
SOL_SYMBOL: "2330"
SOL_USER_ID: "${INPUT_SOL_ID}"
SOL_PASSWORD: "${INPUT_SOL_PWD}"
GEMINI_API_KEY: "${EXISTING_GEMINI_KEY}"
GEMINI_MODEL: "gemini-2.5-flash"
GCS_BUCKET: "${BUCKET_NAME}"
WARROOM_ADMIN_PASSWORD: "${INPUT_ADMIN_PWD}"
EOF
  echo "✅ 已建立 env.yaml！"
fi

# 5. 部署至 Google Cloud Run (min-instances=0 無人看盤時自動縮容至 0、完全不計費)
echo "4️⃣ 正在構建容器並部署至 Cloud Run (${REGION} 台灣彰化機房)..."
gcloud run deploy "${SERVICE_NAME}" \
  --source . \
  --region "${REGION}" \
  --allow-unauthenticated \
  --env-vars-file=env.yaml \
  --timeout 3600 \
  --memory 2Gi \
  --cpu 2 \
  --cpu-boost \
  --session-affinity \
  --min-instances 0 \
  --max-instances 1

# 6. 取得 Cloud Run 服務網址
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format='value(status.url)')

echo "=========================================================="
echo "🎉 恭喜！AI 智慧看盤與多空決策戰情室 Cloud Run 部署完成！"
echo "🌐 專屬戰情室網址: ${SERVICE_URL}"
echo "☁️ 雲端 K 線與籌碼儲存槽: gs://${BUCKET_NAME}"
echo "💤 節費機制: 閒置 30 分鐘休眠 + Cloud Run min-instances=0 (無人使用時 $0 計費)"
echo "=========================================================="
