# 重构整个项目



## 启动与测试

### 启动
运行dist\AskMyZotero.exe

（在发布里面下载）


### 模拟新用户（需要清空“缓存”）

全新用户本机没有写入过的**运行时配置**和**文献索引工作目录**。要复现“第一次使用”，建议删掉下面 **两类位置**（按你当前是脚本还是 exe、以及是否改过路径为准）：

1. **运行时配置文件**  
   默认：`%APPDATA%\AskMyZotero\config.yaml`（默认在这里 e.g  C:\Users\用户名\AppData\Roaming\AskMyZotero）（若设置了 `ASKMYZOTERO_CONFIG`，则删你指向的那份）。  
   也可删除整个 `%APPDATA%\AskMyZotero` 文件夹以彻底清空本轮用户配置。

2. **索引与切块工作目录**（默认相对 `work_dir` + `index_name`，见 `config.yaml` / `config.dist.yaml`）  
   - **脚本启动**：一般在项目根目录下 **`.askmyzotero\default\`**（内含 `faiss_index`、`zotero_splits_cache.pkl`、`manifest.json` 等）。  
   - **exe启动**：一般在 **`AskMyZotero.exe` 同目录下** 的 `.askmyzotero\default\`（因 frozen 时相对路径相对 exe 目录）。  
   删除整个 **`default`** 子目录或整个 **`.askmyzotero`** 即可清空向量库与切块缓存。


### 首次保存配置与向量化（注意终端）

模拟新用户并填写设置点击 **保存** 后：

- 后端会写入 `config.yaml` 并初始化 Agent；**若没有本地 FAISS**，会调用嵌入模型建库，**终端会持续输出进度**（如「向量化进度」等）。
- **请稍等向量与索引构建完成**（时间与文献量、网络、模型有关），完成后才会进入可用状态；设置页在成功保存后会 **弹出浏览器提示**。
- **后续可在前端增加进度条/轮询 `/health` 等**（当前以终端日志为准，避免误以为保存后立刻可搜）。



### 测试效果
<img width="1574" height="1219" alt="20260402011048_146_1" src="https://github.com/user-attachments/assets/e8dee2ca-81a8-4c06-a397-1e83200d4112" />



### 本地开发测试

1. 激活你的 Python 环境后，在项目根目录执行：
   python server.py
2. 浏览器打开 **`http://127.0.0.1:8000/`**（主界面），设置页为 **`http://127.0.0.1:8000/settings.html`**。  
   与 exe 行为一致，推荐统一用上述地址；若从磁盘双击打开 `.html`，仅适合临时查看静态页，页面间跳转与接口可能不一致。

### 一键打包方式

按 `build_exe.bat` 说明在已激活环境中执行清理旧包，重新打包，得到 `dist\AskMyZotero.exe`。


## 2.0更新明细

### 架构与后端
- `server.py` 经 **`ZoteroAgent`** 对外；接口含 `GET/POST /api/config`、`/health`（含 `ready` / `init_error`）、`POST /api/init`；配置写入后可初始化或自动尝试初始化（与保存逻辑、首屏检测的配合详见 **「启动与测试」**）。

### Agent 与检索流程
- 初始化：Manifest 快照 → 向量库 → LLM；请求可带 **`top_k`** 覆盖默认值；提示词倾向于有检索结果时先列相关论文再总结。
- 检索链路逐步说明见下文 **「当前检索结果链路」**。

### 文档视图、Metadata 与卡片数据结构
- **`document_view.py`**：按 `source_path` 聚合成论文卡片，输出 **`evidence_snippets`**（逐条证据，不再做服务端论文级摘要压缩）；（合并/排序功能与 **`agent` 编排** 分离，便于维护。）
- **`parser.enrich_split_metadata`** 补齐 `source_path`、`file_name`、`page_1based`、`chunk_id`、`chunk_rank` 等标准字段。
- **`ReferenceSnippet`**：`evidence_snippets` + 预留 `abstract` / 书目字段（书目待 Zotero 联动）；列表页以 **标题 + 证据块** 为主，HTML 转义防注入。

### 前端阶段性优化：配置持久化
- **`settings.html`** 写入后端配置项；页面均通过 **`http://127.0.0.1:8000/`** 
- 运行时配置默认 **`%APPDATA%\AskMyZotero\config.yaml`**，可用 **`ASKMYZOTERO_CONFIG`**；**`GET /api/config`** 返回 **`api_key_set`**（不明文）；保存时 **API Key 留空 = 不覆盖**；**`localStorage` 不存 Key**，非敏感字段可缓存，以服务端为准。

### 打包
- **Exe**：**`launcher.py`** 拉起服务并打开浏览器；**`build_exe.bat`** 需自行先激活环境；内置配置来自 **`config.dist.yaml`**；打包收集 **`tiktoken`**、**`certifi`** 等；**相对 `work_dir`** 在 frozen 下相对 **exe 目录** 解析。


---

## 当前检索结果链路（简要）

1. **切块**：PDF → 多段 `chunk`，每段带路径、页码、chunk 标识。  
2. **检索**：问题 embedding → FAISS 取 Top-K 相关 **正文 chunk**。  
3. **聚合**：`document_view` 按论文合并 chunk，生成 `evidence_snippets`。  
4. **回答**：LLM 使用完整检索片段拼成的 `context` 生成答案。  
5. **前端**：标题 + 每条 `evidence_snippets` 全文展示；书目信息待 Zotero。


## 1.0更新明细

- 新增**config.yaml配置文件**方便修改测试信息，比如api和pdf文件地址，同步修改了config.py中的resolve_config函数用于读取配置文件
- 新增**server.py**脚本用于前后端交互，目前直接调用底层功能indexer.py中的函数进行测试
- 新增indexer.py中**api_ask_once函数**用于一次性获取回答
- 新增**schemas.py**脚本进行前后端的数据结构设定和检查，ps：除了目前仅有问询和回答，其他根据前端渲染文献卡片的信息设置，但后端还并未同步得到这些信息的功能。
- 修改zotero_rag_ui.html中的runSearch函数，新增**renderResults函数**用于展示回答
- 测试方法：
- - 配置好config.yaml
  - 运行server.py脚本
  - 进入网页输入在你的文件里面能搜到的问题

- 测试结果：
  
  <img width="482" height="411" alt="image" src="https://github.com/user-attachments/assets/9803f85b-0455-4ef4-825f-17fe25c0a756" />




