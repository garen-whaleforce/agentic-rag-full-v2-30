## EarningsCallAgenticRag 依賴設定

本 FastAPI 應用會尋找同層級的 `EarningsCallAgenticRag/` 研究庫做為 Agentic RAG 引擎來源。請先將外部倉庫克隆到此資料夾（與本專案並排）：

```bash
git clone https://github.com/la9806958/EarningsCallAgenticRag.git EarningsCallAgenticRag
```

> 路徑示例：`agentic-rag-full/`（本專案）與 `EarningsCallAgenticRag/`（外部庫）位於同一層。

- 執行時會檢查此資料夾是否存在；未找到時後端會回報缺少依賴。
- 外部庫本身需在其根目錄放置 `credentials.json` 來設定 OpenAI 與 Neo4j，詳細請見該庫的 README。
