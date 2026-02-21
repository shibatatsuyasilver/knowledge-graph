const fs = require('fs');
const path = require('path');
const PptxGenJS = require('pptxgenjs');
const html2pptx = require('/Users/silver/.codex/skills/pptx/scripts/html2pptx.js');

const ROOT = '/Users/silver/Documents/鴻海';
const SLIDES_DIR = path.join(ROOT, 'pptx_build', 'slides');
const OUTPUT = path.join(ROOT, 'genai-graphdb-openclaw-briefing.pptx');

const baseCss = `
html { background: #F4F5F7; }
body {
  width: 720pt;
  height: 405pt;
  margin: 0;
  padding: 0;
  display: flex;
  background: #F4F5F7;
  font-family: "Trebuchet MS", "Arial", sans-serif;
  color: #1f2937;
}
.slide {
  box-sizing: border-box;
  width: 720pt;
  height: 405pt;
  padding: 16pt 20pt;
  display: flex;
  flex-direction: column;
  gap: 8pt;
}
.theme-1 { background: #F4F5F7; }
.theme-2 { background: #FFF7ED; }
.theme-3 { background: #EFF6FF; }
.theme-4 { background: #F0FDF4; }
.kicker {
  margin: 0;
  font-size: 9pt;
  color: #9a3412;
  text-transform: uppercase;
  letter-spacing: 0.8pt;
  font-weight: bold;
}
.title {
  margin: 0;
  font-size: 22pt;
  line-height: 1.2;
  color: #0f172a;
}
.subtitle {
  margin: 0;
  font-size: 11pt;
  line-height: 1.3;
  color: #334155;
}
.section-title {
  margin: 0;
  font-size: 13pt;
  color: #1d4ed8;
}
.two-col {
  display: flex;
  gap: 8pt;
  flex: 1;
  min-height: 0;
}
.col {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 7pt;
  min-height: 0;
}
.card {
  background: #ffffff;
  border: 1pt solid #d1d5db;
  border-radius: 6pt;
  padding: 7pt 8pt;
}
.card h3 {
  margin: 0 0 3pt 0;
  font-size: 12pt;
  color: #111827;
}
.card p {
  margin: 0;
  font-size: 9.6pt;
  line-height: 1.28;
}
ul {
  margin: 0;
  padding-left: 14pt;
}
li {
  margin: 1pt 0;
  font-size: 9.6pt;
  line-height: 1.26;
}
.pipeline {
  display: flex;
  align-items: center;
  gap: 5pt;
  flex-wrap: wrap;
}
.pipe {
  background: #ffffff;
  border: 1pt solid #9ca3af;
  border-radius: 6pt;
  padding: 4pt 6pt;
  min-width: 94pt;
}
.pipe p {
  margin: 0;
  font-size: 8.8pt;
  line-height: 1.2;
}
.arrow {
  margin: 0;
  font-size: 11pt;
  color: #6b7280;
  font-weight: bold;
}
.matrix {
  display: flex;
  flex-direction: column;
  gap: 3pt;
}
.row {
  display: flex;
  gap: 3pt;
}
.cell {
  flex: 1;
  background: #ffffff;
  border: 1pt solid #d1d5db;
  border-radius: 4pt;
  padding: 3pt 4pt;
}
.cell.head {
  background: #111827;
  color: #ffffff;
}
.cell p {
  margin: 0;
  font-size: 8.2pt;
  line-height: 1.2;
}
.code {
  background: #0f172a;
  border-radius: 6pt;
  padding: 6pt;
}
.code p {
  margin: 0;
  font-family: "Courier New", monospace;
  font-size: 7.8pt;
  line-height: 1.18;
  color: #e2e8f0;
}
.small {
  margin: 0;
  font-size: 8.8pt;
  line-height: 1.2;
  color: #475569;
}
.footer {
  margin-top: auto;
  font-size: 8.6pt;
  color: #64748b;
}
`;

function wrap(content, theme = 'theme-1') {
  return `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>${baseCss}</style>
</head>
<body>
<div class="slide ${theme}">
${content}
</div>
</body>
</html>`;
}

const slides = [
  {
    name: '01-cover.html',
    theme: 'theme-2',
    content: `
<p class="kicker">Gen-AI Technical Briefing</p>
<h1 class="title">Gen-AI 知識問答系統與 AI Agent 平台技術簡報</h1>
<p class="subtitle">LLM + Graph DB 知識圖譜 QA Chatbot ｜ OpenClaw AI Agent 安全與實作</p>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>主題一：Knowledge Graph QA</h3>
      <ul>
        <li>Ollama 本機多模型部署</li>
        <li>Neo4j 知識圖譜建構</li>
        <li>自然語言轉 Cypher 查詢</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>主題二：OpenClaw</h3>
      <ul>
        <li>系統架構與核心元件</li>
        <li>安全風險與審查機制</li>
        <li>自訂技能與多技能工作流</li>
      </ul>
    </div>
  </div>
</div>
<p class="footer">日期：2026 年 2 月</p>
`,
  },
  {
    name: '02-agenda.html',
    theme: 'theme-1',
    content: `
<p class="kicker">Slide 2</p>
<h1 class="title">議程總覽</h1>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>主題一：Gen-AI 知識問答 Chatbot</h3>
      <ul>
        <li>開源 LLM 實測與部署（Ollama 本機模型）</li>
        <li>領域知識圖譜建構（Neo4j）</li>
        <li>自然語言轉 Graph DB 查詢語法</li>
        <li>Graph DB 比較（Neo4j/TigerGraph/ArangoDB）</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>主題二：OpenClaw AI Agent 平台</h3>
      <ul>
        <li>系統架構與核心元件</li>
        <li>安全風險深度分析</li>
        <li>技能安全審查機制設計</li>
        <li>自訂技能插件與自動化工作流</li>
      </ul>
    </div>
  </div>
</div>
`,
  },
  {
    name: '03-llm-selection.html',
    theme: 'theme-3',
    content: `
<p class="kicker">Slide 3</p>
<h1 class="title">1. 實際模型使用紀錄（本機 Ollama）</h1>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>已下載與切換測試（實機）</h3>
      <ul>
        <li>deepseek-r1:8b（5.2GB）：KG 抽取早期測試</li>
        <li>gpt-oss:20b（13GB）：已下載做切換評估</li>
        <li>gemma3:12b（8.1GB）：目前整合驗收主模型</li>
      </ul>
    </div>
    <div class="card">
      <h3>目前正式設定</h3>
      <ul>
        <li>docker-compose backend：OLLAMA_MODEL=gemma3:12b</li>
        <li>/api/process_text、/api/process_keyword、/api/query、/api/chat_general 共用</li>
        <li>抽取品質波動時可回切 deepseek-r1:8b 對照測試</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="matrix">
      <div class="row">
        <div class="cell head"><p>項目</p></div>
        <div class="cell head"><p>deepseek-r1:8b</p></div>
        <div class="cell head"><p>gpt-oss:20b</p></div>
        <div class="cell head"><p>gemma3:12b</p></div>
      </div>
      <div class="row">
        <div class="cell"><p>本機狀態</p></div>
        <div class="cell"><p>已下載</p></div>
        <div class="cell"><p>已下載</p></div>
        <div class="cell"><p>已下載</p></div>
      </div>
      <div class="row">
        <div class="cell"><p>專案用途</p></div>
        <div class="cell"><p>抽取基線</p></div>
        <div class="cell"><p>切換評估</p></div>
        <div class="cell"><p>目前主路徑</p></div>
      </div>
      <div class="row">
        <div class="cell"><p>整合狀態</p></div>
        <div class="cell"><p>曾作主模型</p></div>
        <div class="cell"><p>非最終配置</p></div>
        <div class="cell"><p>compose 預設</p></div>
      </div>
    </div>
  </div>
</div>
`,
  },
  {
    name: '04-deploy-practice.html',
    theme: 'theme-1',
    content: `
<p class="kicker">Slide 4</p>
<h1 class="title">2. 部署實戰：本機 Ollama + Docker Backend</h1>
<div class="two-col">
  <div class="col">
    <div class="code">
      <p>curl -fsSL https://ollama.com/install.sh | sh</p>
      <p>ollama pull deepseek-r1:8b</p>
      <p>ollama pull gpt-oss:20b</p>
      <p>ollama pull gemma3:12b</p>
      <p>ollama serve &amp;</p>
      <p>export OLLAMA_BASE_URL=http://host.docker.internal:11434</p>
      <p>export OLLAMA_MODEL=gemma3:12b</p>
    </div>
    <div class="card">
      <h3>FastAPI 封裝</h3>
      <ul>
        <li>POST /api/chat 作為應用層入口</li>
        <li>設定 timeout 與錯誤轉換（504/500）</li>
        <li>低溫度提高結構化輸出穩定度</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>常見問題與解法</h3>
      <ul>
        <li>keyword crawl 504：前端 proxy / axios timeout 提高到 15 分鐘</li>
        <li>抽取變 0：固定 schema prompt + temperature 0 + 回歸測試</li>
        <li>回應過慢：控制 num_predict + 分 chunk 逐批處理</li>
        <li>JSON 失敗：format=json + retry/repair 機制</li>
      </ul>
    </div>
    <div class="card">
      <h3>部署結論</h3>
      <p>Ollama 放在本機、backend 放在 Docker，能保留模型切換彈性，同時維持 API 穩定。</p>
    </div>
  </div>
</div>
`,
  },
  {
    name: '05-kg-pipeline.html',
    theme: 'theme-4',
    content: `
<p class="kicker">Slide 5</p>
<h1 class="title">3. 領域知識圖譜建構流程（台灣半導體）</h1>
<div class="card">
  <div class="pipeline">
    <div class="pipe"><p>資料下載</p><p>新聞/公司介紹</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>前處理</p><p>清洗/斷句/chunk</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>LLM 抽取</p><p>Entity + Relation</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>三元組驗證</p><p>(S,P,O)</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Neo4j 寫入</p><p>MERGE 去重</p></div>
  </div>
</div>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>資料品質挑戰</h3>
      <ul>
        <li>輸出格式不穩：加 JSON schema 約束</li>
        <li>實體別名不一致：台積電/TSMC 正規化</li>
        <li>關係類型發散：預定義關係字典</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>超長來源文章處理（避免記憶體爆量）</h3>
      <ul>
        <li>HTML 清洗後再分塊：chunk_text(max_chars=900, min_chars=120)</li>
        <li>每塊產生 chunk_id=sha1(source_url|text) 去重</li>
        <li>逐 chunk 呼叫 LLM（不把整篇一次塞進 context）</li>
        <li>受限 num_ctx/num_predict，降低 OOM 與 timeout 風險</li>
        <li>單頁失敗記錄 failed_urls，其餘頁面繼續處理</li>
      </ul>
    </div>
  </div>
</div>
`,
  },
  {
    name: '06-neo4j-implementation.html',
    theme: 'theme-3',
    content: `
<p class="kicker">Slide 6</p>
<h1 class="title">4. Neo4j 知識圖譜建構實作（真實抽取 Prompt）</h1>
<div class="two-col">
  <div class="col">
    <div class="code">
      <p>def extract_entities_relations(text):</p>
      <p>  prompt = """你是知識圖譜抽取器，只能輸出合法 JSON"""</p>
      <p>  # entities: Person|Organization|Location|Technology|Product</p>
      <p>  # relations: FOUNDED_BY|HEADQUARTERED_IN|PRODUCES|SUPPLIES_TO|USES|COMPETES_WITH</p>
      <p>  # 規則：關係方向必須符合規範；無法抽取就回空陣列</p>
      <p>  rsp = ollama_chat(model=OLLAMA_MODEL, format="json", temp=0.0)</p>
      <p>  return json.loads(clean(rsp))</p>
      <p></p>
      <p>MERGE (e:Organization {name:$name})</p>
      <p>MERGE (a)-[:RELATION {type:$rel}]-&gt;(b)</p>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>抽取後處理（真實流程）</h3>
      <ul>
        <li>JSON parse 失敗時自動 retry/repair（最多 2 次）</li>
        <li>entity alias merge（例：TSMC ↔ 台積電）</li>
        <li>relation type 與方向驗證，不合法就丟棄</li>
        <li>relation fingerprint 去重，避免重複寫入</li>
      </ul>
    </div>
    <div class="card">
      <h3>工程建議</h3>
      <ul>
        <li>使用 MERGE + index 避免重複</li>
        <li>先寫 node 再寫 edge</li>
        <li>批次匯入並記錄 dropped_relations/json_retries</li>
      </ul>
    </div>
  </div>
</div>
`,
  },
  {
    name: '06b-extraction-prompt-full.html',
    theme: 'theme-1',
    content: `
<p class="kicker">Slide 7（新增）</p>
<h1 class="title">4.1 Entity/Relation 抽取 Prompt（完整版本）</h1>
<div class="two-col">
  <div class="col">
    <div class="code">
      <p>f"""</p>
      <p>你是知識圖譜抽取器。請從文本抽取實體與關係，只能輸出合法 JSON。</p>
      <p></p>
      <p>允許的實體類型:</p>
      <p>&#8203;- Person</p>
      <p>&#8203;- Organization</p>
      <p>&#8203;- Location</p>
      <p>&#8203;- Technology</p>
      <p>&#8203;- Product</p>
      <p></p>
      <p>允許的關係:</p>
      <p>&#8203;- FOUNDED_BY</p>
      <p>&#8203;- HEADQUARTERED_IN</p>
      <p>&#8203;- PRODUCES</p>
      <p>&#8203;- SUPPLIES_TO</p>
      <p>&#8203;- USES</p>
      <p>&#8203;- COMPETES_WITH</p>
      <p></p>
      <p>方向規則:</p>
      <p>&#8203;- Organization -[FOUNDED_BY]-&gt; Person</p>
      <p>&#8203;- Organization -[HEADQUARTERED_IN]-&gt; Location</p>
      <p>&#8203;- Organization -[PRODUCES]-&gt; Technology|Product</p>
      <p>&#8203;- Organization -[SUPPLIES_TO]-&gt; Organization</p>
      <p>&#8203;- Organization -[USES]-&gt; Technology|Product</p>
      <p>&#8203;- Organization -[COMPETES_WITH]-&gt; Organization</p>
    </div>
  </div>
  <div class="col">
    <div class="code">
      <p>輸出格式:</p>
      <p>{{</p>
      <p>  "entities": [</p>
      <p>    {{"name": "台積電", "type": "Organization"}}</p>
      <p>  ],</p>
      <p>  "relations": [</p>
      <p>    {{"source": "台積電", "relation": "FOUNDED_BY", "target": "張忠謀"}}</p>
      <p>  ]</p>
      <p>}}</p>
      <p></p>
      <p>規則:</p>
      <p>1. 只能使用上面的 type 與 relation。</p>
      <p>2. relation 方向必須符合方向規則。</p>
      <p>3. 不要輸出任何額外文字、markdown、註解。</p>
      <p>4. 若文本無法抽取，回傳空陣列，不要猜測。</p>
      <p></p>
      <p>文本:</p>
      <p>"""</p>
      <p>{text}</p>
      <p>"""</p>
      <p>""".strip()</p>
    </div>
    <p class="small">來源：genai_project/llm_kg/kg_builder.py:124-170</p>
  </div>
</div>
`,
  },
  {
    name: '07-nl2cypher.html',
    theme: 'theme-1',
    content: `
<p class="kicker">Slide 7</p>
<h1 class="title">5. 自然語言轉 Cypher 查詢</h1>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>推薦方法：GraphCypherQAChain</h3>
      <ul>
        <li>自動讀取 Neo4j schema</li>
        <li>自動生成 Cypher 並執行查詢</li>
        <li>可回傳 intermediate steps 便於除錯</li>
      </ul>
    </div>
    <div class="code">
      <p>chain = GraphCypherQAChain.from_llm(...)</p>
      <p>result = chain.invoke({"query":"台積電創辦人是誰"})</p>
      <p>print(result["intermediate_steps"][0]["query"])</p>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>常見錯誤與修正</h3>
      <ul>
        <li>語法錯：prompt 加 Cypher 速查規則</li>
        <li>方向錯：schema 明確標注箭頭方向</li>
        <li>查無結果：用 CONTAINS + toLower 正規化</li>
        <li>多跳困難：few-shot 範例引導</li>
      </ul>
    </div>
    <p class="small">流程：User NL → 注入 Schema → LLM 產生 Cypher → 驗證 → 執行 → 回答</p>
  </div>
</div>
`,
  },
  {
    name: '08-graphdb-compare.html',
    theme: 'theme-3',
    content: `
<p class="kicker">Slide 8</p>
<h1 class="title">6. Graph DB 比較：Neo4j vs TigerGraph vs ArangoDB</h1>
<div class="matrix">
  <div class="row">
    <div class="cell head"><p>維度</p></div>
    <div class="cell head"><p>Neo4j</p></div>
    <div class="cell head"><p>TigerGraph</p></div>
    <div class="cell head"><p>ArangoDB</p></div>
  </div>
  <div class="row">
    <div class="cell"><p>查詢語言</p></div>
    <div class="cell"><p>Cypher（直觀）</p></div>
    <div class="cell"><p>GSQL（進階）</p></div>
    <div class="cell"><p>AQL（多模型）</p></div>
  </div>
  <div class="row">
    <div class="cell"><p>LLM 整合</p></div>
    <div class="cell"><p>★★★★★</p></div>
    <div class="cell"><p>★★★</p></div>
    <div class="cell"><p>★★</p></div>
  </div>
  <div class="row">
    <div class="cell"><p>學習曲線</p></div>
    <div class="cell"><p>低</p></div>
    <div class="cell"><p>高</p></div>
    <div class="cell"><p>中</p></div>
  </div>
  <div class="row">
    <div class="cell"><p>適用情境</p></div>
    <div class="cell"><p>知識圖譜/中型規模</p></div>
    <div class="cell"><p>大規模即時圖分析</p></div>
    <div class="cell"><p>圖+文件混合場景</p></div>
  </div>
</div>
<div class="card">
  <p class="small"><b>選型結論：</b>本題以 Neo4j 最佳，因 LangChain 整合完整、Cypher 易被 LLM 穩定生成、開發週期最短。</p>
</div>
`,
  },
  {
    name: '09-openclaw-architecture.html',
    theme: 'theme-4',
    content: `
<p class="kicker">Slide 9</p>
<h1 class="title">7. OpenClaw 平台架構解析（基於原始碼）</h1>
<div class="card">
  <div class="pipeline">
    <div class="pipe"><p>openclaw.mjs</p><p>CLI 入口</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>entry.ts</p><p>Runtime setup</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>gateway/server.impl.ts</p><p>核心路由服務</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>channels / agents / plugins / cron</p><p>功能模組</p></div>
  </div>
</div>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>核心元件</h3>
      <ul>
        <li>Gateway：WebSocket + HTTP + lifecycle</li>
        <li>Channels：多平台訊息標準化</li>
        <li>Agents：session/model/sandbox 管理</li>
        <li>Plugins/Skills：工具與能力擴充</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>程式碼觀察重點</h3>
      <ul>
        <li>協定版本化（request/response/event frame）</li>
        <li>通道能力抽象（polls/reactions/media）</li>
        <li>plugin registry 含 tool/channel/provider/hook</li>
      </ul>
    </div>
  </div>
</div>
`,
  },
  {
    name: '10-openclaw-security-risk.html',
    theme: 'theme-2',
    content: `
<p class="kicker">Slide 10</p>
<h1 class="title">8. OpenClaw 安全風險分析</h1>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>原始碼內建安全機制</h3>
      <ul>
        <li>SandboxConfig + ToolPolicy（open/restricted/closed）</li>
        <li>Exec approvals（always/once/never）</li>
        <li>DANGEROUS_HOST_ENV_VARS 防注入</li>
        <li>設備身份驗證（P-256 + token）</li>
      </ul>
    </div>
    <div class="card">
      <h3>外部曝露風險（社群觀測）</h3>
      <ul>
        <li>大量公開實例增加攻擊面</li>
        <li>惡意技能與供應鏈風險</li>
        <li>prompt injection + bash 濫用</li>
        <li>憑證外洩與資料外傳</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>四大風險類別</h3>
      <ul>
        <li>1) Prompt Injection</li>
        <li>2) ClawHub 供應鏈攻擊</li>
        <li>3) 憑證與資料外洩</li>
        <li>4) bash 工具提權濫用</li>
      </ul>
    </div>
    <p class="small">重點：OpenClaw 已有安全底座，但預設值與技能生態治理仍需強化。</p>
  </div>
</div>
`,
  },
  {
    name: '11-skill-security-review.html',
    theme: 'theme-3',
    content: `
<p class="kicker">Slide 11</p>
<h1 class="title">9. 技能安全審查機制設計（四層防禦）</h1>
<div class="card">
  <div class="pipeline">
    <div class="pipe"><p>Layer 1</p><p>SKILL.md 靜態審查</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Layer 2</p><p>沙箱強制執行</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Layer 3</p><p>LLM 語意審查</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Layer 4</p><p>運行監控 + 回滾</p></div>
  </div>
</div>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>實作原則</h3>
      <ul>
        <li>沿用既有 sandbox/exec 元件再增強</li>
        <li>預設改為 restricted + requireApproval</li>
        <li>掃描危險模式（curl|bash、eval、base64 decode）</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>落地策略</h3>
      <ul>
        <li>安裝前掃描 + 簽章版本控管</li>
        <li>運行時異常檢測（新網域/新路徑）</li>
        <li>社群回報觸發重新審查與緊急下架</li>
      </ul>
    </div>
  </div>
</div>
`,
  },
  {
    name: '12-custom-skills.html',
    theme: 'theme-1',
    content: `
<p class="kicker">Slide 12</p>
<h1 class="title">10. 自訂技能插件開發（SKILL.md）</h1>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>技能一：system-info</h3>
      <ul>
        <li>查詢本機時間、uptime、磁碟狀態</li>
        <li>frontmatter 宣告 requires.bins=[date]</li>
        <li>限制長時間網路診斷指令</li>
      </ul>
    </div>
    <div class="code">
      <p>name: system-info</p>
      <p>metadata.openclaw.requires.bins: [date]</p>
      <p>date '+%Y-%m-%d %H:%M:%S %Z'</p>
      <p>df -h / | tail -1</p>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>技能二：file-digest</h3>
      <ul>
        <li>讀取指定檔案並產生結構化摘要</li>
        <li>禁止敏感檔（.env、id_rsa）</li>
        <li>大檔案只讀前 200 行防資源濫用</li>
      </ul>
    </div>
    <div class="code">
      <p>name: file-digest</p>
      <p>head -200 /path/to/file</p>
      <p>file /path/to/file</p>
      <p>python3 -m json.tool data.json</p>
    </div>
  </div>
</div>
`,
  },
  {
    name: '13-workflow-agent.html',
    theme: 'theme-4',
    content: `
<p class="kicker">Slide 13</p>
<h1 class="title">11. 多技能自動化工作流程 Agent</h1>
<div class="card">
  <div class="pipeline">
    <div class="pipe"><p>Cron 09:00</p><p>daily workflow trigger</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Skill A</p><p>daily-collect</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Skill B</p><p>daily-summarize</p></div>
    <p class="arrow">→</p>
    <div class="pipe"><p>Skill C</p><p>daily-report + notify</p></div>
  </div>
</div>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>資料流</h3>
      <ul>
        <li>.local/collect.md：檔案清單 + 系統狀態</li>
        <li>.local/summaries.md：逐檔摘要</li>
        <li>.local/daily-report-YYYY-MM-DD.md：最終報告</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="code">
      <p>Cron: 0 9 * * *</p>
      <p>Message:</p>
      <p>請依序執行 /daily-collect、/daily-summarize、/daily-report</p>
      <p>完成後通知使用者</p>
    </div>
  </div>
</div>
`,
  },
  {
    name: '14-summary.html',
    theme: 'theme-2',
    content: `
<p class="kicker">Slide 14</p>
<h1 class="title">12. 總結與未來展望</h1>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>主題一結論（LLM + Graph DB）</h3>
      <ul>
        <li>Neo4j + Cypher 對知識圖譜 QA 最快落地</li>
        <li>本機 Ollama（deepseek / gpt-oss / gemma3）可快速切換驗證</li>
        <li>未來可升級 GraphRAG（向量 + 圖遍歷）</li>
      </ul>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>主題二結論（OpenClaw）</h3>
      <ul>
        <li>原始碼已有企業級安全元件基礎</li>
        <li>高風險在技能供應鏈與預設配置鬆散</li>
        <li>建議四層防禦 + sandbox 預設啟用</li>
      </ul>
    </div>
    <div class="card">
      <h3>下一步</h3>
      <ul>
        <li>技能簽章與版本治理</li>
        <li>行為監控與異常回滾</li>
        <li>多 Agent 分層權限設計</li>
      </ul>
    </div>
  </div>
</div>
`,
  },
  {
    name: '15-qa.html',
    theme: 'theme-1',
    content: `
<p class="kicker">Slide 15</p>
<h1 class="title">Q&amp;A</h1>
<div class="card">
  <h3>感謝聆聽，歡迎提問與討論。</h3>
  <p>可討論方向：模型選型、Cypher 生成品質、OpenClaw 安全治理與企業落地流程。</p>
</div>
<div class="two-col">
  <div class="col">
    <div class="card">
      <h3>附註</h3>
      <p>本簡報內容以技術原型與原始碼分析為主，實際上線需再做壓測、資安滲測與維運設計。</p>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h3>交付檔案</h3>
      <p>/Users/silver/Documents/鴻海/genai-graphdb-openclaw-briefing.pptx</p>
    </div>
  </div>
</div>
`,
  },
];

async function build() {
  fs.mkdirSync(SLIDES_DIR, { recursive: true });
  for (const slide of slides) {
    fs.writeFileSync(path.join(SLIDES_DIR, slide.name), wrap(slide.content, slide.theme), 'utf8');
  }

  const pptx = new PptxGenJS();
  pptx.layout = 'LAYOUT_16x9';
  pptx.author = 'Codex';
  pptx.company = 'Foxconn';
  pptx.subject = 'Gen-AI Knowledge Graph and OpenClaw Security';
  pptx.title = 'Gen-AI 知識問答系統與 AI Agent 平台技術簡報';

  for (const slide of slides) {
    await html2pptx(path.join(SLIDES_DIR, slide.name), pptx, {
      tmpDir: path.join(ROOT, 'pptx_build', 'tmp'),
    });
  }

  await pptx.writeFile({ fileName: OUTPUT });
  console.log(`Created: ${OUTPUT}`);
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
