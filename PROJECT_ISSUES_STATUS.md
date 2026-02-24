# 專案問題與解決方案狀態追蹤 (Project Issues & Solutions Status)

## 1. 目前問題追蹤 (Current Issues Tracking)

| 問題 (Problem) | 解法 (Solution) | 狀態 (Status) |
| :--- | :--- | :--- |
| **Chat 打到錯誤 upstream（/api/chat 404）** | 改成共用 LLM 抽象層，支援 openai-compatible / ollama / gemini，由環境變數切換。 | ✅ 已完成 (`backend/llm_kg/llm_client.py`, `backend/logic.py`) |
| **Gemini 金鑰未生效（400 INVALID_ARGUMENT）** | 統一讀取 .env.local、補文件與範例 env，明確要求 GEMINI_API_KEY。 | ✅ 已完成 (`.env.local.example`, `README.md`) |
| **抽取常出現 JSON 破損/<think> 導致 parse fail** | 抽取加 5 次重試、JSON 修復 prompt、提高輸出 token 上限。 | ✅ 已完成 (`backend/llm_kg/kg_builder.py`) |
| **前端長時間只顯示 Processing...** | Text/File/URL 改 async job + polling，顯示 chunk 進度與狀態。 | ✅ 已完成 (`backend/main.py`, `frontend/src/components/BuildKG.tsx`) |
| **長文本導致記憶體壓力（OOM 風險）** | Ingest 端改為先切 chunk 再逐塊抽取（provider-aware token/char budget），並支援 `chunk_limit` 限流。 | ✅ 已完成 (`backend/logic.py`, `backend/main.py`) |
| **KG 回答太生硬、常帶「根據知識圖譜」前綴** | /api/query 改成先查 rows，再用 QA LLM 重寫答案，失敗才 fallback。 | ✅ 已完成 (`backend/logic.py`) |
| **UNION/UNION ALL 分支 RETURN 欄位別名不一致，觸發 Neo4j syntax error（500）** | 目前僅在 `Agentic Async (Manual)` 路徑有 UNION 別名一致性修正與 repair loop；`GraphCypherQAChain (Sync)` 尚未套用同等 guard，因此仍可能失敗。 | 🚧 部分完成（Manual：✅，GraphCypherQAChain：待補強） |
| **一句多問（例如「董事長 + 創辦人」）** | 已可直接處理同句多意圖問答（例如同時詢問創辦人與董事長），由 agentic 流程生成與修正查詢後回覆。 | ✅ 已完成 (`backend/llm_kg/nl2cypher.py`) |
| **實體漏抽問題** | Gemini 走兩階段抽取：先盤點 entity 比對 KG 補齊，再第二輪抽 relation。 | ✅ 已完成 (`backend/llm_kg/kg_builder.py`, `GEMINI_TWO_PASS_EXTRACTION=1`) |
| **OpenClaw Skill 安全審查覆蓋不足** | 目前僅有 Regex 靜態檢查（`skill_audit.ts`）；需補 AST 分析、沙箱動態測試、來源簽章與權限審批。 | 🚧 待實作 |
| **OpenClaw 權限模型粒度不足** | 導入 capability-based 權限（檔案/網路/命令/外部 API 分離授權），高風險權限需雙重確認。 | 🚧 待實作 |
| **OpenClaw 機敏憑證治理不足** | 第三方 API Token 改由密鑰管理與靜態加密保存，補 token 輪替、審計與外洩告警。 | 🚧 待實作 |
| **OpenClaw Prompt Injection 風險** | 對 Skill 回傳內容加 trusted/untrusted 標記、模板化包裝與指令隔離，阻斷「工具輸出覆蓋系統指令」。 | 🚧 待實作 |

---

## 2. 開源 LLM 模型佈署詳細說明

### (1) 選擇的開源 LLM 模型
本專案採用 **Hybrid 模式**，主要使用 **Ollama (DeepSeek R1 / Qwen 3 / Ministral 3)** 作為本地開源模型，並支援切換至 **Gemini 3 Pro (Preview)**（或相容新版本）以獲得更高品質的知識抽取。

*   **選擇模型**: `deepseek-r1:8b` / `qwen3` / `ministral-3:14b` (透過 Ollama) 或 `gemini-3-pro-preview`。
*   **優點 (Pros)**:
    *   **DeepSeek R1 / Qwen 3**: 支援 **Chain of Thought (CoT)** 思維鏈，邏輯推理能力顯著增強，適合處理複雜指令，同時保有本地部署的隱私優勢與無 API 成本。
    *   **Ministral 3 (14B)**: 在 **指令遵守** 與 **結構化輸出** 表現穩定，特別適合 **NL2Cypher** 任務（可降低 Cypher 生成偏題與格式漂移）。
    *   **Gemini 3 Pro**: 擁有超長 Context Window (**1,048,576 input tokens**) 與強大的邏輯推理能力，對於複雜的實體關係抽取 (NER/RE) 準確率通常優於小參數模型。
*   **缺點 (Cons)**:
    *   **DeepSeek R1 / Qwen 3**: 在啟用 thinking 或輸出約束不足時，可能混入思考內容/註釋而導致 JSON 解析失敗 (Parse Fail)；且 CoT 可能增加推論時間。需搭配 `think=false`/`hidethinking` 與後處理保護。
    *   **Gemini 3 Pro**: 依賴網路，有 API Rate Limit 限制與成本，且存在數據傳輸隱私考量。

#### 官方 Benchmark 參考（截至 2026-02-15）
> 註：以下為官方模型卡/官方頁面公布的成績，不同推論框架、量化版本、sampling 設定下，實測結果可能浮動。

| 模型 | 官方測試指標 | 分數 |
| :--- | :--- | :--- |
| `deepseek-r1:8b`（Ollama 對應 `DeepSeek-R1-0528-Qwen3-8B`） | AIME24 / AIME25 / GPQA-Diamond / LiveCodeBench | **86.0 / 76.3 / 61.1 / 60.5** |
| `qwen3:8b`（Thinking, bf16） | LiveBench / GPQA / MMLU-Redux / AIME24 | **67.1 / 62.0 / 87.5 / 76.0** |
| `qwen3:8b`（Non-Thinking, bf16） | LiveBench / GPQA / MMLU-Redux | **53.5 / 39.3 / 79.5** |
| `ministral-3:14b`（Reasoning） | AIME25 / AIME24 / GPQA-Diamond / LiveCodeBench | **85.0 / 89.8 / 71.2 / 64.6** |
| `ministral-3:14b`（Instruct） | Arena Hard / WildBench / MATH Maj@1 / MM MTBench | **55.1 / 68.5 / 90.4 / 8.49** |
| `gemini-3-pro-preview` | Humanity's Last Exam / GPQA-Diamond / AIME 2025 / MMMU-Pro | **37.5 / 91.9 / 95.0 / 81.0** |
| `gemini-3-pro-preview` | Token limits | **Input 1,048,576 / Output 65,536** |

**資料來源（官方）**
1. DeepSeek 模型卡（含 `DeepSeek-R1-0528-Qwen3-8B` 成績）：https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
2. Qwen3 8B 模型卡（Thinking/Non-Thinking 對照）：https://huggingface.co/Qwen/Qwen3-8B-AWQ
3. Ministral 3 14B 模型卡（Benchmark Results）：https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
4. Gemini 3 Pro benchmark: https://deepmind.google/technologies/gemini/pro/
5. Gemini 3 Pro token limits: https://ai.google.dev/gemini-api/docs/models/gemini

**解讀方式（對應你的任務）**
*   NER/RE：可優先參考 GPQA、AIME、HLE 這類推理/知識密集指標，作為「複雜關係抽取能力」的近似代理。
*   NL2Cypher：可優先參考 Arena Hard、WildBench、MM MTBench 與模型是否原生支援 function calling / JSON outputs，作為「指令遵守 + 結構化輸出」的近似代理。
*   目前沒有單一官方 benchmark 可以直接等同 NER/RE 或 NL2Cypher，仍建議保留專案內部回歸測試集作最終依據。

### (2) 部署方式
使用 **Docker Container** 進行微服務化部署：
1.  **Ollama Service**: 使用官方 `ollama/ollama` Docker 映像檔，掛載 GPU (如支援) 進行本地推論。
2.  **API Wrapper**: 開發 `backend/llm_kg/llm_deploy.py` (基於 FastAPI)，作為統一人工智慧介面層 (`backend/llm_kg/llm_client.py`)。
    *   此層負責處理不同 Provider (Ollama/OpenAI/Gemini) 的 API 差異。
    *   透過環境變數 (`LLM_PROVIDER`) 動態切換後端模型，無需修改程式碼。
3.  **Orchestration**: 使用 `backend/llm_kg/docker-compose.llmkg.yml` 定義服務依賴與網絡。

### (3) 部署遇到的問題與解決
*   **問題 1: JSON 格式破損 (Malformed JSON)**
    *   **狀況**: 開源模型 (DeepSeek R1 / Qwen 3) 在特定設定下可能輸出思考內容、不完整 JSON 或 Markdown 註釋，導致 `json.loads` 失敗。
    *   **解決**: 在 `backend/llm_kg/kg_builder.py` 實作 **Retry & Repair Loop**。優先透過模型參數關閉/隱藏 thinking，再於解析失敗時回傳錯誤給 LLM 要求修正 JSON，最多重試 5 次。
*   **問題 2: 回應超時 (Timeout)**
    *   **狀況**: 處理長文本輸入時，LLM 生成時間過長導致 HTTP 504 Gateway Timeout。
    *   **解決**: 前端改為 **Async Job + Polling** 機制。上傳文件後回傳 Job ID，前端每隔幾秒查詢進度，避免長連接斷開。
*   **問題 3: 思維鏈 (Chain of Thought) 干擾**
    *   **狀況**: 部分模型在 thinking 模式下可能輸出額外推理內容，干擾 JSON 結構。
    *   **解決**: 先用模型參數關閉/隱藏 thinking（`think=false` 或 `hidethinking`），並在 Prompt 禁止輸出思考過程；後處理 Regex 清洗作為最後保險。
*   **問題 4: 長文件造成記憶體壓力（OOM 風險）**
    *   **狀況**: 單次把整份長文件丟入抽取流程，會放大 token/記憶體負載，在本地機或容器資源有限時容易不穩定。
    *   **解決**: 在 `backend/logic.py` 將 Text/File/URL ingest 改為 **chunk-first pipeline**：
        *   先依 provider 做切塊：Gemini 預設 token budget，其他 provider 預設 char budget（可由 `CHUNK_SIZE_MODE` 強制切換）。
        *   對超長段落做二次切分（token overflow segment split），避免單塊超過上限。
        *   逐 chunk 抽取並寫入 KG，搭配 `chunk_limit`/`INGEST_CHUNK_LIMIT` 控制單次處理量，降低 OOM 與長請求失敗率。

---

## 3. Domain Knowledge Graph 建立詳細說明

### 步驟說明
*   **數據來源**: 企業財務報告 (Financial Reports) 與新聞稿 (範例：鴻海、台積電)。
*   **Graph DB**: 選擇 **Neo4j (Community Edition v5)**。
*   **建置流程**: 使用 `backend/llm_kg/kg_builder.py` 進行自動化建置。
*   **大檔處理策略**: 透過 `backend/logic.py` 先做 chunking（token/char budget），再逐塊呼叫 `backend/llm_kg/kg_builder.py` 抽取與 upsert，避免單次大上下文造成 OOM。

### (0) 實作對應：實體/關係抽取與 Neo4j 寫入
*   **抽取框架**: 本專案不是使用傳統 NER/RE 套件（如 spaCy stanza）做規則式抽取，而是採用 `backend/llm_kg/kg_builder.py` 的 **LLM JSON 抽取流程**。
*   **LLM 呼叫層**: 使用自建 `backend/llm_kg/llm_client.py`（底層套件為 `requests`）統一呼叫 `openai-compatible / ollama / gemini`，再由 Prompt 約束輸出固定 JSON 結構。
*   **抽取策略**:
    *   預設 Single-pass：一次輸出 `entities + relations`。
    *   Gemini 可啟用 Two-pass（`GEMINI_TWO_PASS_EXTRACTION=1`）：先實體盤點，再以 seed entities 抽關係，並先補齊 KG 缺漏實體後再做第二階段關係抽取。
    *   解析失敗時有 Retry & Repair（最多 5 次）以修復 JSON。
*   **抽取後清洗 (Post-processing)**:
    *   依白名單限制 entity/relation type（schema constraints）。
    *   做 alias/canonical name 合併與模糊比對，降低同實體多寫法問題。
    *   檢查關係方向與型別是否符合 ontology 規則；不符合者直接丟棄。
*   **Neo4j 寫入方式**（`neo4j` Python driver）:
    1. `_ensure_constraints()`：建立唯一鍵與索引（`Entity.name` unique、`normalizedName` index）。
    2. `_create_entity()`：使用 `MERGE` upsert 節點，節點同時帶有具體 label（如 `:Organization`）與共通 `:Entity`。
    3. `_create_relation()`：使用 `MERGE (a)-[:REL]->(b)` upsert 關係。
    4. `populate_graph()`：逐筆寫入 entities/relations，回傳統計（upsert 數、drop 數、json retries）。

### (1) 利用 LLM 建立 Knowledge Graph
*   **方法**: 採用 **Two-Pass Extraction (兩階段抽取法)**。
    *   **Phase 1 (Entity Inventory)**: 先讓 LLM 掃描全文，列出所有實體 (Entities)，進行標準化 (Canonicalization)。
    *   **Phase 2 (Relation Extraction)**: 將第一階段的實體清單作為 Context 餵給 LLM，要求其找出實體間的關係 (Relations)。
*   **遇到的問題與解決**:
    *   **實體歧義 (Entity Ambiguity)**: 同一公司有多種寫法 (e.g., "鴻海", "Hon Hai", "Foxconn")。
        *   **解決**: 實作 `_resolve_entity_reference` 與 `SequenceMatcher` 模糊比對，將別名映射到單一標準名稱 (Canonical Name)。
    *   **幻覺關係 (Hallucinated Relations)**: LLM 創造不存在的關係類型。
        *   **解決**: 設定 **Schema Constraints (Ontology)**，僅允許特定的 Node Labels (e.g., `Organization`, `Person`) 與 Relation Types (e.g., `FOUNDED_BY`, `SUPPLIES_TO`)，過濾掉不符合 Schema 的輸出。

### (2) 使用者問題轉 Graph DB 查詢 (NL2Cypher)
*   **方法**: `backend/llm_kg/nl2cypher.py` 提供兩種引擎，並由 `query_engine` 控制執行路徑：
    *   **Path A (GraphCypherQAChain / Sync)**: 使用 `GraphCypherQAChain + Neo4jGraph` 完成 NL -> Cypher -> Query -> Answer；已接入 `POST /api/query`（`query_engine=graph_chain`），支援 `ollama` 與 `gemini` provider（`backend/llm_kg/nl2cypher.py:1041`、`backend/llm_kg/nl2cypher.py:1062`、`backend/llm_kg/nl2cypher.py:1070`）。
    *   **Path B (Manual/Agentic)**: 使用可配置模型搭配 Schema-aware prompt 與多輪 agentic loop；維持預設模式（`query_engine=manual`）。
    *   **Path B 實際執行鏈**: `planner -> reactor -> link_entity_literals -> execute (含必要時放寬組織名稱 exact match) -> critic`；若 verdict 為 `replan` 則進 `replanner` 進入下一輪。另保留 repeated-cypher 防呆與 `max rounds` 上限。
    *   **為何預設 `manual`**: 此路徑支援 async 進度事件（`agentic_progress`）、失敗鏈追蹤（`failure_chain`）、以及重試耗盡後的失敗快照保留與除錯資訊，較適合長任務與除錯（`backend/api/routers/qa.py:142`、`backend/llm_kg/nl2cypher.py:1777`、`backend/llm_kg/nl2cypher.py:2006`）。
    *   **同步/非同步邊界**: `GraphCypherQAChain` 僅支援同步 `/api/query`；若送到 `/api/query_async/start` 會直接回 `400`（`backend/api/routers/qa.py:116`）。
    *   **邊界設計原因**: async 路徑依賴可回傳階段事件的 callback；`graph_chain` 分支在 service 層明確不接受 async callback，因此在 router 提前阻擋，避免建立後必定失敗的 job（`backend/services/qa/service.py:355`、`backend/services/qa/service.py:356`）。
    *   **選擇建議**: 需要單次快速問答可用 `graph_chain`（sync only）；需要可觀測的多輪修復、長查詢進度與重試控制時用 `manual`（sync/async 皆可）。
    *   **前端模式對應**: KG Chat 已新增 Execution Mode 下拉，提供 `Agentic Async (Manual)` 與 `GraphCypherQAChain (Sync)` 兩種選擇（`frontend/src/components/Chat.tsx:261`、`frontend/src/components/Chat.tsx:269`）。
    *   **回應相容性**: GraphChain 路徑維持 `question/cypher/rows/answer`，並附加 `query_engine`、`graph_chain_raw`、`engine_provider`、`engine_model`（`backend/services/qa/service.py:370`-`backend/services/qa/service.py:390`）。

#### 目前 `GraphCypherQAChain` 初始化參數（現況）
*   **Chain 建立**: `GraphCypherQAChain.from_llm(llm=..., graph=..., verbose=True, return_intermediate_steps=True, allow_dangerous_requests=True)`（`backend/llm_kg/nl2cypher.py:1085`-`backend/llm_kg/nl2cypher.py:1091`）。
*   **Ollama LLM**: `Ollama(model=resolved_model, temperature=0, base_url=cfg.ollama_base_url)`（`backend/llm_kg/nl2cypher.py:1065`-`backend/llm_kg/nl2cypher.py:1069`）。
*   **Gemini LLM**: `ChatGoogleGenerativeAI(model=resolved_model, temperature=0, google_api_key=cfg.gemini_api_key)`（`backend/llm_kg/nl2cypher.py:1077`-`backend/llm_kg/nl2cypher.py:1081`）。
*   **目前未設定**: `top_k`、`return_direct`、`validate_cypher`、`include_types`、`exclude_types`、`use_function_response`。

#### 為何目前未開啟 `validate_cypher=True`
*   **策略選擇**: GraphChain 路徑目前採「失敗直接回錯，不自動 fallback」；保留原始錯誤對除錯更直接。
*   **Manual 現況**: Manual/Agentic 主路徑以 `critic/replanner` 迴圈驅動修正，並搭配 repeated-cypher 防呆與 `max rounds`，不依賴 deterministic relation gate。
*   **後續擴充**: 若要啟用 `validate_cypher`，建議改成可配置旗標，並以 A/B 測試評估對成功率與延遲的影響後再預設打開。

#### 使用 `GraphCypherQAChain` 的好處（官方文件對應）
*   **[現況已用] 端到端流程內建**：`GraphCypherQAChain` 可直接完成「自然語言問題 -> 產生 Cypher -> 查 Neo4j -> 生成答案」，目前已接入同步 API 路徑。
*   **[現況已用] Provider 可切換**：目前同一條 GraphChain 路徑可切換 `ollama` / `gemini`，並支援 model override。
*   **[現況已用] 可觀測性高，便於除錯**：已設定 `return_intermediate_steps=True`，可取得中間產物（生成 Cypher 與查詢 context）供除錯與稽核（`backend/llm_kg/nl2cypher.py:1089`）。
*   **[官方可用，本專案未啟用] Schema 導向生成 Cypher**：可配合 `Neo4jGraph.refresh_schema()` 與 `enhanced_schema=True`，降低不存在 label/property/relationship 的生成風險；目前未啟用。
*   **[官方可用，本專案未啟用] 結果可控**：`top_k`、`return_direct=True` 可限制資料量與輸出型態；目前未設定。
*   **[官方可用，本專案未啟用] 可客製 Cypher 生成品質**：可透過 `cypher_prompt` 注入 few-shot，或分離 `cypher_llm` / `qa_llm` 做任務分工；目前未配置。
*   **[本專案刻意未啟用] 關係方向驗證/修正**：官方可用 `validate_cypher=True`，目前保留 fail-fast 行為，避免隱性修正導致可觀測性下降。
*   **[官方可用，本專案未啟用] function/tool response**：`use_function_response=True` 可提升答案與資料列一致性；目前未啟用。
*   **[現況已用] 風險明示同意**：目前 `query_with_graph_chain` 已顯式設定 `allow_dangerous_requests=True`（`backend/llm_kg/nl2cypher.py:1090`）。

#### 常見誤解澄清（新版用語）
*   **`GraphCypherQAChain` 與 `Neo4jGraph` 功能分層不同**：前者負責 NL -> Cypher -> Query -> QA 編排；後者負責圖連線與 schema 提供。
*   **`refresh_schema()` / `enhanced_schema` 屬於 `Neo4jGraph`**：不是 `GraphCypherQAChain` 本身的方法。
*   **本專案現況**：GraphChain 已接 `/api/query` 同步模式；`/api/query_async/*` 僅支援 Manual/Agentic。

#### 新版/舊版功能歸屬對照（含本專案現況）
| 主題 | 功能 | 所屬元件 | 本專案現況 |
| :--- | :--- | :--- | :--- |
| 端到端 NL -> Cypher -> QA | `GraphCypherQAChain.from_llm(...)` | `GraphCypherQAChain` | 已接同步主路徑（`/api/query` + `query_engine=graph_chain`） |
| Provider/Model 覆寫 | `nl2cypher_provider` / `nl2cypher_model` | 本專案 GraphChain wrapper | 已使用（`ollama` / `gemini`） |
| 限制回傳筆數 | `top_k` | `GraphCypherQAChain` | 可用未啟用 |
| 直接回傳資料列 | `return_direct` | `GraphCypherQAChain` | 可用未啟用 |
| 輸出中間步驟 | `return_intermediate_steps` | `GraphCypherQAChain` | 已使用 |
| 關係方向驗證/修正 | `validate_cypher` | `GraphCypherQAChain` | 可用未啟用（目前策略為 fail-fast） |
| 限制 schema 子集 | `include_types` / `exclude_types` | `GraphCypherQAChain` | 可用未啟用 |
| function/tool response | `use_function_response` | `GraphCypherQAChain` | 可用未啟用 |
| 刷新 schema | `refresh_schema()` | `Neo4jGraph` | 可用未啟用 |
| schema 增強（樣本值/分佈） | `enhanced_schema=True` | `Neo4jGraph` | 可用未啟用 |
| 迴圈安全防呆 | repeated-cypher limit + `NL2CYPHER_AGENTIC_MAX_ROUNDS` | 本專案 NL2Cypher agentic loop | Manual 路徑已使用；GraphChain 路徑不適用 |

#### 官方來源（LangChain，最新版）
1. Neo4j provider integration：https://docs.langchain.com/oss/python/integrations/providers/neo4j
2. Neo4j graph/cypher integration：https://docs.langchain.com/oss/python/integrations/graphs/neo4j_cypher
3. LangChain Security：https://docs.langchain.com/oss/python/security-policy

#### Legacy 參考（對照目前 `langchain_community` 匯入路徑）
1. GraphCypherQAChain API Reference（community）：https://api.python.langchain.com/en/latest/community/chains/langchain_community.chains.graph_qa.cypher.GraphCypherQAChain.html

> 安全註記（官方 + 現況）：`GraphCypherQAChain` 需明確 `allow_dangerous_requests=True` 才可執行，且官方要求資料庫帳號必須使用最小權限（narrowly-scoped credentials）。本專案目前在 `query_with_graph_chain` 已顯式設定此參數（`backend/llm_kg/nl2cypher.py:1090`）。

### (3) 選擇 Neo4j 的原因與比較

| 圖資料庫 (Graph DB) | 特點與優勢 (Pros) | 劣勢與限制 (Cons) | 本專案適用性分析 (Verdict) |
| :--- | :--- | :--- | :--- |
| **Neo4j** (本專案採用) | 1. **成熟度高**：市場佔有率最高，社群資源最豐富。<br>2. **查詢語言**：Cypher 直觀且類似 SQL，易於 LLM 學習與生成。<br>3. **視覺化強**：內建 Neo4j Browser 與 Bloom，方便除錯。 | 大規模分散式運算需企業版支援；寫入吞吐量在極端場景下可能不如專門的分散式圖資料庫。 | **最佳選擇**<br>適合本專案的中小型規模，Docker 部署簡單，且 Cypher 對 LLM 最友善。 |
| **TigerGraph** | **分散式運算強**：適合超大規模 (TB級) 數據分析。 | **學習曲線陡**：GSQL 較複雜。<br>**社群版限制**：功能與限制較多。 | **不適合**<br>本專案規模未達 TB 級，且 Neo4j 對於中小型規模更易於上手與部署。 |
| **ArangoDB** | **多模資料庫 (Multi-model)**：同時支援 Document, Key-Value, Graph，靈活性高。 | **非原生圖資料庫**：在純圖演算法與深度遍歷效能上通常不如原生的 Neo4j。 | **不適合**<br>本專案專注於複雜關係鏈查詢，原生圖資料庫 (Native Graph DB) 較為合適。 |
| **NebulaGraph** | **寫入吞吐量高**：針對海量數據設計的開源分佈式圖資料庫。 | **部署複雜**：在單機開發環境的輕量級部署上，配置較 Neo4j 繁瑣。 | **備選**<br>Docker 版的 Neo4j 配置較為簡單，符合目前快速開發需求。 |

### (4) 詳細流程圖（建圖 + 查詢全流程）

以下以 6 張 Mermaid 圖同步目前實作：完整覆蓋 ingest 的 sync + async 路徑；query 同時涵蓋 `Agentic Async (Manual)` 與 `GraphCypherQAChain (Sync)`，並補上 manual 路徑共用回答後處理。

#### 圖 A：Ingest API 與同步/非同步任務協調總覽

```mermaid
flowchart TB
    subgraph API["API｜入口與查詢"]
        A1[POST api process text｜同步文字建圖請求入口]
        A2[POST api process url｜同步網址建圖請求入口]
        A3[POST api process keyword｜同步關鍵字建圖請求入口]
        A4[POST api process text async start｜非同步文字建圖啟動入口]
        A5[POST api process url async start｜非同步網址建圖啟動入口]
        A6[POST api process keyword async start｜非同步關鍵字建圖啟動入口]
        A7[GET api process text async by job id｜非同步文字建圖狀態查詢]
        A8[GET api process url async by job id｜非同步網址建圖狀態查詢]
        A9[GET api process keyword async by job id｜非同步關鍵字建圖狀態查詢]
    end

    subgraph JOB["JOB｜建立與回寫"]
        B1[create job in ingest or keyword store｜建立 ingest 或 keyword job 紀錄]
        B2[run worker thread｜背景執行緒處理作業]
        B3[update progress result error｜更新進度與結果錯誤]
        B4[get job by id｜依 job id 讀取狀態]
    end

    subgraph PIPE["PIPE｜建圖主鏈"]
        C1[logic process to kg facade｜logic facade 轉發建圖]
        C2[ingest service process to kg｜ingest service 主流程]
        C3[build kg from chunks｜逐 chunk 建圖處理]
        C4[chunk update callback｜chunk 進度回呼上報]
        C5[result stats summary chunk progress｜統計與進度摘要輸出]
    end

    A1 --> C1
    A2 --> C1
    A3 --> C1
    A4 --> B1
    A5 --> B1
    A6 --> B1
    B1 --> B2
    B2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C5
    C3 --> C4
    C4 --> B3
    B3 --> A7
    B3 --> A8
    B3 --> A9
    A7 -.-> B4
    A8 -.-> B4
    A9 -.-> B4
    B4 --> A7
    B4 --> A8
    B4 --> A9
```

圖 A 聚焦 API 層與 Job Store 協調：同步請求直接執行；非同步請求會先建立 job（`ingest_job_store` / `keyword_job_store`），再由背景執行與 polling 狀態查詢串起完整流程。節點已改為英文 + 繁中說明，便於程式對照與閱讀。

#### 圖 B：建圖主線（Chunk -> Extract -> Sanitize -> Upsert）

```mermaid
flowchart TB
    B0[logic process text url keyword to kg｜建圖 facade 入口]
    B1[service chunk text by token or char｜依 token 或 char 切塊]
    B2[resolve chunk limit｜套用 chunk_limit 決策]
    B3[build kg from chunks｜逐塊建圖主迴圈]
    B4{more chunks｜是否仍有未處理塊}
    B5[extract entities and relations｜抽取實體與關係]
    B6[sanitize extraction｜抽取結果清洗守門]
    B7[populate graph ensure entity relation｜upsert 節點與關係]
    B8[aggregate stats and chunk status｜聚合統計與狀態]
    B9[emit chunk update callback｜送出 chunk_update 事件]
    B10[return stats summary chunk progress｜回傳最終統計進度]

    B0 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 -- yes/是 --> B5
    B5 --> B6
    B6 --> B7
    B7 --> B8
    B8 --> B9
    B9 --> B4
    B4 -- no/否 --> B10
```

圖 B 對齊實作中的「逐 chunk 抽取與寫入」主線，並明確標示 `_resolve_chunk_limit`、chunk_update 回報與最終聚合輸出。節點已改為英文 + 繁中說明，便於程式對照與閱讀。

#### 圖 C：抽取內核 Retry/Repair 與 Two-pass 分支

```mermaid
flowchart TB
    C0[extract entities relations entry｜抽取流程入口]
    C1{gemini two pass enabled｜是否啟用 Gemini two-pass}
    C2[phase1 entity prompt｜第一階段實體盤點提示]
    C3[retry json phase1｜第一階段 JSON 重試抽取]
    C4[fetch existing and prefill missing｜補齊既有實體與預填]
    C5[phase2 relation prompt｜第二階段關係抽取提示]
    C6[retry json phase2｜第二階段 JSON 重試抽取]
    C7[single pass prompt｜單階段抽取提示]
    C8[retry json single｜單階段 JSON 重試抽取]
    C9[sanitize extraction｜套用 schema 清洗守門]
    C10[output entities relations meta｜輸出 entities relations meta]

    subgraph RETRY["RETRY｜JSON重試"]
        R1[attempt loop｜重試嘗試迴圈]
        R2{json mode｜JSON 模式判斷}
        R3[call chat json or text｜呼叫 chat_json chat_text]
        R4{payload valid｜內容是否有效}
        R5[build repair prompt｜建立 repair prompt]
        R6[return payload with retries｜回傳有效 payload 與重試資訊]
        R7[raise error at max retry｜達上限拋出錯誤]
        R1 --> R2
        R2 --> R3
        R3 --> R4
        R4 -- yes/是 --> R6
        R4 -- no/否 --> R5
        R5 --> R1
        R1 -- max reached/達重試上限 --> R7
    end

    C0 --> C1
    C1 -- yes/是 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5
    C5 --> C6
    C1 -- no/否 --> C7
    C7 --> C8
    C6 --> C9
    C8 --> C9
    C9 --> C10
    C3 -.-> R1
    C6 -.-> R1
    C8 -.-> R1
```

圖 C 把 two-pass/single-pass 分流與 JSON 修復回圈同圖呈現，方便定位是「抽取策略問題」或「格式重試問題」。節點已改為英文 + 繁中說明，便於程式對照與閱讀。

#### 圖 D：同步查詢分流（`/api/query`：Manual vs GraphCypherQAChain）

```mermaid
flowchart TB
    S1[post api query｜同步查詢入口]
    S2[query_sync router｜qa router 同步入口]
    S3[_query_with_overrides to logic query kg｜帶 provider model query_engine 轉呼叫]
    S4{query_engine｜manual 或 graph_chain}

    subgraph MPATH["Manual｜預設同步路徑"]
        M1[answer_with_manual_prompt｜agentic manual 查詢]
        M2[_generate_kg_answer_with_llm｜QA LLM 回答改寫]
        M3[_summarize_query_rows｜模板摘要 fallback]
        M4[return answer cypher rows answer_source query_engine manual｜回傳 manual 結果]
    end

    subgraph GPATH["GraphChain｜同步專用路徑"]
        G1[query_with_graph_chain｜進入 GraphCypherQAChain]
        G2{provider｜ollama 或 gemini}
        G3[build graphcypherqachain from_llm｜return_intermediate_steps true]
        G4[invoke query on chain｜執行 query]
        G5[normalize question cypher rows answer｜標準化相容欄位]
        G5A{answer empty｜answer 是否為空}
        G5B[_summarize_query_rows｜以 rows 產生摘要補值]
        G6[attach graph_chain_raw engine_provider engine_model query_engine graph_chain｜附加欄位]
        G7[error return directly no auto fallback｜錯誤直接回傳]
    end

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 -- manual or empty/預設 --> M1
    M1 --> M2
    M2 --> M4
    M2 -. llm unavailable or invalid .-> M3
    M3 --> M4
    S4 -- graph_chain --> G1
    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> G5
    G5 --> G5A
    G5A -- no/否 --> G6
    G5A -- yes/是 --> G5B
    G5B --> G6
    G4 -. exception .-> G7
```

圖 D 對齊同步 `/api/query` 的 `query_engine` 分流：`manual` 走既有 agentic + QA 回答流程；`graph_chain` 走 `query_with_graph_chain`，成功回傳相容欄位並附加 `graph_chain_raw` 等資訊。若 GraphChain 的 `answer` 為空，會在同一路徑以 `_summarize_query_rows` 補值；執行失敗則直接回錯誤且不自動 fallback 到 manual。

#### 圖 E：非同步查詢與 Agentic Loop（`/api/query_async/*`，Manual 專用）

```mermaid
flowchart TB
    subgraph API["API｜非同步查詢"]
        E1[post api query async start｜非同步查詢啟動入口]
        E1A{query_engine is graph_chain｜是否誤用 graph_chain}
        E1B[return 400 only supported by api query sync｜graph_chain 僅支援同步]
        E2[create query job with planner stage｜建立 query job 與初始階段]
        E3[run worker thread｜背景執行查詢工作]
        E4[get api query async by job id｜依 job id 查詢狀態]
        E5[return progress result error｜回傳進度結果錯誤]
    end

    subgraph LOOP["LOOP｜agentic迴圈"]
        F1[load schema and entity names｜載入 schema 與實體名稱]
        F2[emit agentic progress callback｜發送 agentic progress 事件與 trace 快照]
        F3[planner｜planner 規劃查詢策略]
        F4[reactor｜reactor 生成查詢動作]
        F5[link entity literals｜link_entity_literals 實體字面值連結]
        F6[execute cypher and optional relax exact org match｜執行 cypher 與必要放寬組織名稱精確比對]
        F7[critic｜critic 評估結果品質]
        F8{verdict｜判定結果分支}
        F9[replanner and merge plan｜replanner 產生並合併新計畫]
        F10[next round｜進入下一輪]
        F11[accept rows done｜接受結果並完成]
        F12[fail fast｜收到 fail_fast 提前停止]
        F13[exhausted raise runtime error｜輪次耗盡或提前停止後拋錯]
    end

    E1 --> E1A
    E1A -- yes/是 --> E1B
    E1A -- no/否 --> E2
    E2 --> E3
    E3 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5
    F5 --> F6
    F6 --> F7
    F7 --> F8
    F8 -- accept/接受 --> F11
    F8 -- replan/重規劃 --> F9
    F9 --> F10
    F10 --> F4
    F10 -. max rounds/達上限 .-> F13
    F8 -- fail_fast --> F12
    F12 --> F13
    F2 -.-> E2
    F11 --> E5
    F13 --> E5
    E4 --> E5
```

圖 E 對齊目前 agentic async 實作：入口先拒絕 `query_engine=graph_chain`（400），其餘才進入 `planner -> reactor -> link_entity_literals -> execute(+optional relax) -> critic -> replan`。若 `accept` 則完成；若 `fail_fast` 或達最大輪次，會進入錯誤終止並由 async job 回報 failed，同時保留最後 `agentic_trace` 快照。

#### 圖 F：Manual 路徑共用回答後處理（sync + async）

```mermaid
flowchart TB
    D0[query route or async worker calls logic query kg｜sync router 或 async worker 呼叫 query_kg]
    D1[qa service query kg｜qa service 主流程]
    D2[answer with manual prompt｜manual 路徑執行 answer_with_manual_prompt]
    D3[raw result with cypher rows trace｜取得 cypher rows trace]
    D4{use qa llm｜是否啟用 QA LLM}
    D5[generate answer with llm｜LLM 生成自然語句答案]
    D6{answer valid｜QA LLM 答案是否可用}
    D7[answer source qa llm｜answer_source 設為 qa_llm]
    D8[summarize rows fallback｜rows 模板摘要 fallback]
    D9[answer source template fallback｜answer_source 設為 template_fallback]
    D10[return answer rows cypher query_engine manual｜回傳結果 async 另回寫 job]

    D0 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 -- yes/是 --> D5
    D5 --> D6
    D6 -- yes/是 --> D7
    D6 -- no/否 --> D8
    D4 -- no/否 --> D8
    D8 --> D9
    D7 --> D10
    D9 --> D10
```

圖 F 描述 `query_engine=manual` 的共用後處理：同步 `/api/query` 與 async worker 都會走到，最後輸出 `answer/rows/cypher/query_engine=manual`；若為 async 再回寫 job result。

#### 本節流程圖涉及的 Env 變數作用表（精簡）

| Env 變數 | 主要讀取位置 | 作用說明 | 流程影響點 |
| :--- | :--- | :--- | :--- |
| `CHUNK_SIZE_MODE` | `backend/config/settings.py` -> `backend/services/ingest/service.py` | 切塊策略 `provider|token|char`。 | 圖 A/B |
| `CHUNK_SIZE_TOKENS` / `CHUNK_MIN_TOKENS` | `backend/config/settings.py` -> `backend/services/ingest/service.py` | token 切塊上限與最小門檻。 | 圖 B |
| `CHUNK_SIZE_CHARS` / `CHUNK_MIN_CHARS` | `backend/config/settings.py` -> `backend/services/ingest/service.py` | char 切塊上限與最小門檻。 | 圖 B |
| `INGEST_CHUNK_LIMIT` | `backend/config/settings.py` -> `backend/services/ingest/service.py` | 全域 chunk 上限（與 request 參數合併）。 | 圖 A/B |
| `KEYWORD_SEARCH_MODE` | `backend/config/settings.py` -> `backend/services/ingest/service.py` | 控制 keyword crawl 搜尋策略（html_only/html_first 等）。 | 圖 A |
| `EXTRACTION_JSON_MODE` | `backend/llm_kg/kg_builder.py` | JSON 解析模式。 | 圖 C |
| `EXTRACTION_MAX_JSON_RETRIES` | `backend/llm_kg/kg_builder.py` | 修復重試上限。 | 圖 C |
| `EXTRACTION_NUM_PREDICT` | `backend/config/settings.py` -> `backend/llm_kg/kg_builder.py` | 抽取輸出 token 預算。 | 圖 C |
| `GEMINI_TWO_PASS_EXTRACTION` | `backend/llm_kg/kg_builder.py` | two-pass 分支開關。 | 圖 C |
| `KG_QA_USE_LLM` | `backend/services/qa/service.py` | 是否啟用 QA LLM 改寫。 | 圖 F（manual 路徑） |
| `KG_QA_MODEL` / `KG_QA_TEMPERATURE` / `KG_QA_MAX_TOKENS` / `KG_QA_MAX_ROWS_FOR_PROMPT` | `backend/services/qa/service.py` | QA 回答品質與輸出長度控制。 | 圖 F（manual 路徑） |
| `NL2CYPHER_TIMEOUT_SECONDS` / `NL2CYPHER_NUM_PREDICT` | `backend/llm_kg/nl2cypher.py` | NL2Cypher 呼叫 timeout 與 token 預算。 | 圖 E |
| `NL2CYPHER_PROVIDER` / `NL2CYPHER_MODEL` | `backend/config/settings.py` -> `backend/llm_kg/nl2cypher.py` | 查詢階段 provider/model（可被 QueryRequest 覆寫）。 | 圖 D：sync manual/graph_chain 分流；圖 E：async manual agentic loop；圖 F：manual 共用回答後處理 |
| `OLLAMA_BASE_URL` / `GEMINI_API_KEY` | `backend/llm_kg/llm_client.py` -> `backend/llm_kg/nl2cypher.py` | GraphChain 路徑建立 LLM client（Ollama endpoint / Gemini API key）所需；缺值時直接拋錯。 | 圖 D：`query_with_graph_chain` provider 建鏈與執行 |
| `NL2CYPHER_AGENTIC_MAX_ROUNDS` | `backend/llm_kg/nl2cypher.py` | agentic loop 最大輪數。 | 圖 E |
| `NL2CYPHER_AGENTIC_PLAN_TOKENS` / `NL2CYPHER_AGENTIC_REACT_TOKENS` / `NL2CYPHER_AGENTIC_CRITIC_TOKENS` | `backend/llm_kg/nl2cypher.py` | planner/reactor/critic token 預算。 | 圖 E |
| `CYPHER_REPAIR_RETRIES`（legacy） | `backend/llm_kg/nl2cypher.py` | 舊式修復語境參數；非 agentic 主迴圈控制鈕。 | 圖 E 次要路徑 |

#### 本節流程圖涉及的核心函式作用表（精簡）

| 函式 | 檔案 | 作用 |
| :--- | :--- | :--- |
| `process_text_sync` / `process_url_sync` / `process_keyword_sync` | `backend/api/routers/ingest.py` | Ingest 同步 API 入口。 |
| `process_text_async_start` / `process_url_async_start` / `process_keyword_async_start` | `backend/api/routers/ingest.py` | Ingest 非同步入口（建立 job + 背景執行）。 |
| `logic.process_*_to_kg` | `backend/logic.py` | Facade wrapper，轉呼叫 ingest service。 |
| `chunk_text` / `_resolve_chunk_limit` / `build_kg_from_chunks` | `backend/services/ingest/service.py` | 切塊、限流、逐 chunk 抽取與統計聚合。 |
| `extract_entities_relations` / `_extract_entities_relations_two_pass` | `backend/llm_kg/kg_builder.py` | 抽取主控制器（single/two-pass）。 |
| `_extract_json_with_retry` / `_sanitize_extraction` | `backend/llm_kg/kg_builder.py` | JSON 修復重試與 schema 清洗守門。 |
| `populate_graph` | `backend/llm_kg/kg_builder.py` | Neo4j upsert 協調。 |
| `query_sync` / `query_async_start` / `query_async_status` | `backend/api/routers/qa.py` | Query 同步/非同步 API 與 job 狀態查詢；async 入口拒絕 `query_engine=graph_chain`。 |
| `logic.query_kg` | `backend/logic.py` | QA facade wrapper。 |
| `backend/services/qa/service.py::query_kg` | `backend/services/qa/service.py` | 依 `query_engine` 分流至 manual 或 graph_chain，並做回傳欄位標準化。 |
| `answer_with_manual_prompt` / `_run_agentic_query_loop` / `_emit_progress` | `backend/llm_kg/nl2cypher.py` | agentic NL2Cypher 主流程與進度回報。 |
| `query_with_graph_chain` | `backend/llm_kg/nl2cypher.py` | GraphCypherQAChain 同步路徑：依 provider 建立 chain 並解析 intermediate steps。 |
| `_run_planner_agent` / `_run_reactor_agent` / `_run_critic_agent` / `_run_replanner_agent` | `backend/llm_kg/nl2cypher.py` | planner/reactor/critic/replanner 子代理決策鏈。 |
| `link_entity_literals` / `_relax_exact_organization_name_match` | `backend/llm_kg/nl2cypher.py` | 執行前實體字面值連結與 zero-row 時的組織名稱放寬策略。 |
| `_generate_kg_answer_with_llm` / `_summarize_query_rows` | `backend/services/qa/service.py` | Manual 路徑的 `qa_llm` 與 `template_fallback` 回答分支。 |

---

## 4. OpenClaw 系統架構與技能安全分析

### (1) OpenClaw 系統架構與主要組件
以下依 `ARCHITECTURE.md` 彙整：

*   **Gateway Server (`src/gateway`)**：
    *   以微核心架構為中樞，負責配置載入、Plugin 掛載、HTTP/WebSocket 服務、事件廣播與子服務生命週期。
    *   內含 Discovery、Tailscale、Exec Approval Manager 等關鍵能力。
*   **Channel System (`src/channels`)**：
    *   將 WhatsApp / Telegram / Discord / Slack 等異質訊息標準化，統一輸入事件格式。
*   **Auto-reply Engine (`src/auto-reply`)**：
    *   負責訊息路由、Session 管理、模型選擇、ReAct 工具迴圈執行與最終回覆生成。
*   **Plugin / Skill 生態 (`extensions/`, `skills/`)**：
    *   透過外掛擴充 Channel、Tools、CLI 指令與 HTTP Routes，是平台擴展性核心。
*   **基礎設施層 (`src/cron`, `src/media`)**：
    *   提供排程、媒體轉換、短期檔案服務與清理機制。

**資料流（Data Flow，引用 `ARCHITECTURE.md`）**

#### 4.1 訊息處理流程

```mermaid
sequenceDiagram
    participant User
    participant Channel (Plugin)
    participant Gateway
    participant Dispatcher
    participant AgentRunner
    participant LLM

    User->>Channel (Plugin): 發送訊息
    Channel (Plugin)->>Gateway: 標準化訊息 (Normalized Event)
    Gateway->>Dispatcher: dispatchInboundMessage()
    
    note over Dispatcher: Security Check (Command Gating)
    
    Dispatcher->>Dispatcher: 建立 MsgContext
    Dispatcher->>AgentRunner: 觸發 Agent 執行
    
    loop Thinking Process
        AgentRunner->>LLM: 發送 Prompt + Context
        LLM-->>AgentRunner: 回傳 Thought / Tool Call
        opt Tool Execution
            AgentRunner->>Gateway: 執行工具 (e.g., search, query)
            Gateway-->>AgentRunner: 回傳 Tool Result
        end
    end

    AgentRunner-->>Dispatcher: 生成最終回覆 (Payload)
    Dispatcher->>Channel (Plugin): 發送回覆
    Channel (Plugin)->>User: 顯示訊息
```

#### 4.2 Agent 思考過程詳解 (Agent Execution Loop)

Agent 的執行邏輯位於 `src/auto-reply/reply/agent-runner.ts` 與 `agent-runner-execution.ts`，其核心流程如下：

1. **上下文準備 (Context Prep)**：
   - 從 Session Store 載入對話歷史。
   - 注入系統提示詞 (System Prompt) 與當前可用工具 (Tools)。
2. **權限檢查 (Security)**：
   - **Command Gating**：檢查發送者是否在允許清單 (Allowlist) 中，以決定是否執行敏感指令。
   - **Access Groups**：支援群組層級的權限管理。
3. **模型調用 (Model Invocation)**：
   - 將上下文發送至配置的 LLM (OpenAI, Anthropic 等)。
   - 若配置了 Fallback，當主要模型失敗時會自動切換至備援模型。
4. **推理與工具 (Reasoning & Tools)**：
   - **Thinking**：Agent 產生思考過程 (CoT)。
   - **Action**：若 Agent 決定使用工具，系統會攔截 Tool Call，執行對應的 TypeScript 函數，並將結果 (Observation) 附加回對話歷史。
5. **串流回應 (Streaming)**：
   - 支援 **Block Streaming**，即時將 Agent 的部分思考或回應推送給使用者，減少等待感。
   - 處理 **Typing Indicators**，在 Agent 思考或執行工具時顯示「正在輸入...」。
6. **記憶體管理 (Memory Management)**：
   - 執行後自動檢查 Context Window，必要時觸發 **Compaction**（摘要/壓縮歷史紀錄）。

#### 4.3 媒體處理管道 (Media Pipeline)

位於 `src/media/`，負責處理圖片、音訊與影片檔案。

* **暫存伺服器 (Ephemeral Server)**：啟動一個 Express Server 提供本地媒體檔案的 HTTP 存取。
* **生命週期管理**：上傳的媒體檔案預設有 TTL (Time-To-Live，例如 2 分鐘)，過期後會由 `cleanOldMedia` 定期清理，確保磁碟空間不被佔用。
* **MIME 偵測**：使用 `detectMime` 自動識別檔案類型。

### (2) OpenClaw 安全風險（3 大類 10 項）

#### A. Skill 生態系風險（5 項，最高優先）
1. **任意程式碼執行 (RCE)**：Skill 透過 `eval` / shell pipe / `child_process` 執行惡意指令。
2. **路徑遍歷 (Path Traversal)**：惡意路徑（如 `../../etc/passwd`）讀取超出授權範圍檔案。
3. **資料外洩 (Exfiltration)**：Skill 讀取本機敏感檔後透過網路上傳到外部端點。
4. **供應鏈攻擊 (Supply Chain Poisoning)**：惡意 Skill 偽裝實用工具混入生態。
5. **Prompt Injection via Skill Output**：Skill 輸出夾帶惡意指令，誘導 Agent 覆寫安全邊界。

#### B. 執行環境風險（3 項）
6. **沙箱逃逸 (Sandbox Escape)**：vm/容器隔離不足時，惡意程式突破執行邊界。
7. **過度高權限執行**：Skill 直接取得過大系統權限（檔案、網路、命令同時開放）。
8. **聊天訊息注入向量**：外部訊息直接拼接到高權限工具 prompt/命令，造成間接注入。

#### C. 系統整合與治理風險（2 項）
9. **權限粒度不足**：缺少 capability 級別授權，無法最小權限化。
10. **Token/Secrets 管理不足**：第三方 API Token 若未加密、未輪替、缺稽核，外洩風險高。

**現況觀察（本 repo）**：
*   已有 `genai_project/openclaw/security/skill_audit.ts`，可抓到 `curl|bash`、`eval`、`sudo`、`elevated:true` 等高風險模式。
*   目前屬於 **靜態 Regex 基線檢查**，尚未覆蓋 AST 行為分析、沙箱動態驗證、來源簽章與完整權限政策。

### (3) 技能安全審查機制設計（建議落地版）

#### 3.1 四階段審查總覽（上架前）

| Phase | 名稱 | 核心方法 |
| :--- | :--- | :--- |
| 1 | Manifest Audit | 必要欄位驗證、權限白名單比對、sandbox 強制限制 |
| 2 | Static Analysis | 危險 API Regex/規則掃描 + 混淆偵測（長行、Hex/Unicode 跳脫） |
| 3 | Sandbox Trial | 受控執行、網路攔截、系統呼叫監控、資源上限 |
| 4 | Signature Verification | `SHA-256(manifest + entrypoint)` + 發布者公鑰（RSA/等效）驗證 |

**決策邏輯**
*   有 `CRITICAL/ERROR`：自動拒絕。
*   有 `HIGH/WARN`：轉人工複審。
*   全通過且簽章有效：允許上架。

#### 3.2 Phase 1：Manifest Audit（清單審查）

先審查 Skill 的 `skill.json`（自我聲明），不合規直接擋下，不進下一關。

**必要欄位驗證**
```json
{
  "name": "file_summarizer",
  "version": "1.0.0",
  "author": "demo-author",
  "permissions": ["fs.read.whitelist"],
  "entrypoint": "index.js",
  "sandbox": {
    "exec": false,
    "network": false,
    "fs_read": ["/home/user/documents"],
    "fs_write": []
  }
}
```

**權限白名單比對**
*   允許：`system.time`、`fs.read.whitelist`、`network.outbound.allowlist`
*   拒絕：`system.root`、`fs.read.*`、`exec.shell`

**Sandbox 強制規則**
*   `sandbox.exec=true`：直接 `CRITICAL`（拒絕）。
*   `sandbox.network=true` 但未宣告對應網路權限：`ERROR`（拒絕）。
*   `fs_read/fs_write` 必須是明確白名單路徑。

#### 3.3 Phase 2：Static Analysis（靜態掃描）

不執行程式，直接掃描原始碼模式。

**危險 API 風險分級**
*   `CRITICAL`：`child_process`、`exec()`、`execSync`、`spawn()`、`eval()`、`new Function()`
*   `HIGH`：未授權網路呼叫（`http/https/fetch`）、檔案破壞操作（`fs.writeFile/fs.unlink/fs.rmdir`）
*   `MEDIUM`：`process.env`、`process.exit()`、可疑解碼（`Buffer.from(base64)`）
*   `LOW`：Hex/Unicode 跳脫字串、過度混淆痕跡

**混淆偵測**
*   單行超過 500 字元觸發警示（常見於混淆或惡意 payload）。
*   偵測 `\xNN`、`\uNNNN` 大量跳脫字元。

**限制說明（需搭配 Phase 3）**
```javascript
const mod = "child" + "_" + "process";
require(mod).exec("rm -rf /");
```
上述動態組字可能繞過純字串規則，因此需進入沙箱動態驗證。

#### 3.4 Phase 3：Sandbox Trial（沙箱試跑）

在隔離環境實際執行 Skill，驗證行為是否與宣告一致。

**隔離層級（由弱到強）**
*   Node.js `vm`（不建議單獨依賴，歷史上有逃逸風險）
*   gVisor（syscall 層攔截）
*   Firecracker microVM（高隔離）

**試跑監控**
*   網路：DNS/TCP 連線紀錄，未授權外聯立即標記。
*   檔案：`open/read/write` 存取是否超出白名單。
*   程序：禁止 `fork/exec`；CPU > 5 秒或記憶體 > 128 MB 即終止。
*   行為一致性：以 mock 輸入執行，檢查輸出是否符合 manifest schema。

```javascript
const result = await skill.run(
  { filepath: "/home/user/documents/test.txt" },
  { permissions: mockPermissions, logger: mockLogger }
);
validateAgainstSchema(result, skill.manifest.schema.output);
```

#### 3.5 Phase 4：Signature Verification（數位簽章驗證）

確保「通過審查的內容」在發布與安裝過程中未被竄改。

**發布端**
1. 對 `manifest + entrypoint` 做 SHA-256。
2. 用發布者私鑰簽章，產生 `skill.sig`。

**平台驗證端**
1. 對收到的 `manifest + entrypoint` 重新計算 SHA-256。
2. 用註冊公鑰驗證簽章。
3. 不一致即拒絕上架/安裝。

#### 3.6 兩個 Skill Plugin（落地示例）

1. **`system_info`**
   *   功能：查詢系統時間、時區、hostname。
   *   權限：`network=false`、`exec=false`。
   *   執行期再驗證：以 `context.permissions.require()` 做 runtime gate，避免配置漂移。

2. **`file_summarizer`**
   *   功能：讀取指定目錄內文字檔並摘要。
   *   五層防護：
       1. null byte 防護
       2. `path.resolve()` 正規化
       3. 白名單目錄比對
       4. 副檔名白名單
       5. 1MB 檔案大小限制

#### 3.7 縱深防禦總結

`Phase 1 -> Phase 2 -> Phase 3 -> Phase 4` 分別處理「宣告合規、靜態風險、動態行為、完整性驗證」。  
任一階段命中 `CRITICAL/ERROR` 即中止流程，降低惡意 Skill 進入生態系機率。

---

## 5. OpenClaw 自訂 Skill Plugin 與自動化 Agent 實作說明

### (1) 自訂 OpenClaw Skill Plugin（`todo-helper`）

已實作全新 plugin：`extensions/todo-helper/`，提供三個工具能力：

1. `local_time({ timezone? })`
   * 回傳本地或指定時區時間（ISO、local string、epoch）。
2. `read_and_summarize({ path, maxBytes?, maxSentences? })`
   * 在 allowlist 路徑內讀取檔案並回傳摘要（可關閉）。
3. `read_todo_snapshot({ path?, includeCompleted?, lookaheadHours? })`
   * 讀取 Todo 快照 JSON，驗證欄位，並分類：
     * `overdue`（逾期）
     * `dueToday`（今日到期）
     * `upcoming`（即將到期）
     * `noDueDate`（未設定到期）
   * 另含 stale 檢查，避免 agent 使用過舊資料。

相關檔案：
* `extensions/todo-helper/index.ts`
* `extensions/todo-helper/openclaw.plugin.json`
* `extensions/todo-helper/skills/todo-sync/SKILL.md`
* `extensions/todo-helper/skills/todo-notify/SKILL.md`

### (2) 自動化 Agent 工作流程（至少兩個 skill）

本實作使用 **3 個 skill**（符合「兩個以上 skill」要求）：
1. `todo-sync`
2. `organize-files`
3. `todo-notify`

排程：每日 `09:00`（本地時區）啟動，流程如下：

1. **todo-sync**
   * 先呼叫 `local_time`，再呼叫 `read_todo_snapshot` 取得最新待辦狀態。
   * 做逾期 / 今日 / 即將到期分類，並檢查 snapshot 是否 stale。
2. **organize-files**
   * 整理指定資料夾（例如 `~/Downloads`），輸出搬移/整理結果。
3. **todo-notify**
   * 生成可直接發送的提醒內容（包含下一步行動）。
   * 送到 `last route`（最近互動通道）。

#### 詳細設定步驟（實際可執行）

1. **準備 To-do snapshot 檔案（資料來源）**
   * 在 To-do 前端連接 snapshot 檔，建議使用：
     * Host 路徑：`/Users/silver/.openclaw/workspace/openclaw-data/todo-snapshot.json`
   * 說明：Gateway 容器內對應路徑會是：
     * Container 路徑：`/home/node/.openclaw/workspace/openclaw-data/todo-snapshot.json`

2. **設定 `todo-helper` plugin config**
   * 在 `~/.openclaw/openclaw.json` 確認：
   * `allowedRoots` 包含 `/home/node/.openclaw/workspace/openclaw-data`
   * `defaultSnapshotPath` 設為 `/home/node/.openclaw/workspace/openclaw-data/todo-snapshot.json`

   ```json
   {
     "plugins": {
       "entries": {
         "todo-helper": {
           "enabled": true,
           "config": {
             "allowedRoots": [
               "/home/node/.openclaw/workspace/openclaw-data",
               "/home/node/.openclaw/workspace/automation-demo"
             ],
             "defaultSnapshotPath": "/home/node/.openclaw/workspace/openclaw-data/todo-snapshot.json",
             "maxReadBytes": 262144,
             "staleMinutes": 180,
             "enableReadSummary": true
           }
         }
       }
     }
   }
   ```

   * 套用設定：

   ```bash
   cd /Users/silver/Documents/openclaw-main
   docker compose restart openclaw-gateway
   ```

3. **建立/更新 cron job（固定 skill 順序）**

   ```bash
   AUTOMATION_MESSAGE=$(cat <<'PROMPT'
   You are executing a cron automation verification run.

   Mandatory requirements (do not skip):
   1) Read /app/extensions/todo-helper/skills/todo-sync/SKILL.md.
   2) Call local_time exactly once.
   3) Call read_todo_snapshot exactly once with:
      /home/node/.openclaw/workspace/openclaw-data/todo-snapshot.json
   4) Read /app/extensions/todo-helper/skills/organize-files/SKILL.md.
   5) Use exec to organize /home/node/.openclaw/workspace/automation-demo/inbox:
      - documents: *.pdf, *.docx
      - images: *.jpg, *.png
      - top-level files only (maxdepth 1)
   6) Read /app/extensions/todo-helper/skills/todo-notify/SKILL.md.

   Final output headings:
   Time
   Snapshot Health
   Priority Focus
   Upcoming
   File Organization
   Next Step
   PROMPT
   )

   cd /Users/silver/Documents/openclaw-main
   docker compose run --rm openclaw-cli cron add \
     --name "verify-todo-workflow" \
     --cron "0 9 * * *" \
     --session isolated \
     --message "$AUTOMATION_MESSAGE" \
     --deliver \
     --channel last \
     --best-effort-deliver
   ```

   * 若 job 已存在，改用：

   ```bash
   docker compose run --rm openclaw-cli cron edit <jobId> \
     --message "$AUTOMATION_MESSAGE" \
     --deliver \
     --channel last \
     --best-effort-deliver
   ```

4. **手動觸發與驗證（含 log 取證）**

   ```bash
   cd /Users/silver/Documents/openclaw-main
   docker compose run --rm openclaw-cli cron list
   docker compose run --rm openclaw-cli cron run <jobId> --force --timeout 180000
   docker compose run --rm openclaw-cli cron runs --id <jobId> --limit 5
   ```

   ```bash
   docker compose exec openclaw-gateway sh -lc '
   S=$(ls -t /home/node/.openclaw/agents/main/sessions/*.jsonl | head -n 1)
   echo "$S"
   grep -n "todo-sync\|organize-files\|todo-notify\|local_time\|read_todo_snapshot\|\"name\":\"exec\"" "$S" | sed -n "1,120p"
   '
   ```

5. **驗收重點**
   * session log 可看到 `todo-sync -> organize-files -> todo-notify` 對應讀取/工具呼叫。
   * `read_todo_snapshot` 使用的是 `openclaw-data/todo-snapshot.json` 路徑。
   * `organize-files` 只整理 top-level 的目標副檔名檔案。
   * 重跑後檔案結構維持不變（idempotent）。
   * 若未設定 `--to` recipient，`cron runs` 可能顯示 `Delivery skipped`，此時以 session log 驗證流程是否完整執行。

錯誤分支設計：
* 若 snapshot 讀取失敗或 stale，回覆必含「失敗原因 + 修復建議」。

完整流程與命令展示文件：
* `docs/automation/todo-web-agent-workflow.md`

### (3) To-do 前端作為自動化資料來源

To-do 專案（`/Users/silver/Documents/To-do`，compose 服務 `todo-app`）負責輸出 snapshot：
* CRUD、優先級、標籤、到期時間、篩選。
* 使用 `localStorage` 保存本地資料。
* 透過 File System Access API 連接 `todo-snapshot.json` 後，待辦變更即自動同步。

---

## 6. 附件：如何啟動（Frontend + OpenClaw + Docker）

### (1) 啟動 OpenClaw Gateway（Docker）

```bash
cd /Users/silver/Documents/openclaw-main
docker compose up -d openclaw-gateway
docker compose ps openclaw-gateway
```

### (2) 開啟 Control UI（帶 Token）

```bash
cd /Users/silver/Documents/openclaw-main
source .env
open "http://127.0.0.1:18789/?token=$OPENCLAW_GATEWAY_TOKEN"
```

若出現 `pairing required`：

```bash
docker compose exec openclaw-gateway node dist/index.mjs devices list --json
docker compose exec openclaw-gateway node dist/index.mjs devices approve <requestId> --json
```

### (3) 啟動 To-do 前端

```bash
cd /Users/silver/Documents/To-do
docker compose up -d --build todo-app
docker compose ps todo-app
```

瀏覽器開啟：`http://localhost:8080`
（To-do 專案 compose 服務：`todo-app`，定義於 `/Users/silver/Documents/To-do/docker-compose.yml`）

### (4) 安裝與啟用 `todo-helper` plugin

```bash
cd /Users/silver/Documents/openclaw-main
docker compose run --rm openclaw-cli plugins install ./extensions/todo-helper
docker compose run --rm openclaw-cli plugins enable todo-helper
docker compose restart openclaw-gateway
```

### (5) 驗證排程流程（手動觸發）

```bash
cd /Users/silver/Documents/openclaw-main
docker compose run --rm openclaw-cli cron list
docker compose run --rm openclaw-cli cron run <jobId> --force
docker compose run --rm openclaw-cli cron runs --id <jobId> --limit 20
```
