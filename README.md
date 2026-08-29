# KuaiRand-Pure Starter Kit

## 依赖

Python 3.9+ 和 numpy。**没有别的。** 不需要 torch、pandas、sklearn。

## 数据

从 https://kuairand.com 下载（Zenodo 直链，无需注册）：

```bash
# 在 Starter Kit 目录下执行，解压后得到 ./KuaiRand-Pure/
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
```

## 运行

```bash
python3 baseline.py --model fm
```

`--data_dir` 默认 `./KuaiRand-Pure/data`；数据放在别处时显式指定。

`--model` 可选 `fm`（官方 baseline）/ `pop`（trivial baseline）/ `random`（下界，用于自检评测代码）。
FM 全程约 40 秒（CPU，单核）。

## 任务定义（口径已写死，不要改）

| | |
|---|---|
| 任务 | **用户内排序** —— 每个用户只对其在评测集中的曝光排序，不做全库检索 |
| 相关性标签 | `long_view`（原生列，0/1） |
| 指标 | `GAUC`、`nDCG@5`；**主分 = 两者平均** |
| 数据划分 | train `20220408–20220421` / valid `20220422–20220428` / test `20220429–20220508` |
| 零正例用户 | nDCG 记 0.0 并计入平均；GAUC 只统计 `0 < 正例数 < 曝光数` 的用户，按正例数加权 |
| nDCG gain | `2^rel − 1`（二元标签下等价于 identity） |

实现见 `evaluate.py`，全部约定写在文件头注释里。

## Baseline 阶梯

test 集上的分数。**要打败的是 FM 这一行。**

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random（下界，自检用） | 0.4996 | 0.4511 | 0.4753 |
| item popularity（trivial） | 0.6308 | 0.5121 | 0.5715 |
| **FM（官方 baseline）** | **0.6610** | **0.5282** | **0.5946** |

### ⚠️ 指标的真实区间：nDCG@5 的天花板是 0.729，不是 1.0

test 集 23,875 个用户里：

| | 占比 | 对指标的影响 |
|---|---|---|
| 全负用户（该用户所有曝光都不是 long_view） | **27.1%** | nDCG 恒为 **0**，任何模型都救不了；不计入 GAUC |
| 全正用户 | **9.2%** | nDCG 恒为 **1**；不计入 GAUC |
| 有区分度的用户 | **63.7%** | GAUC 的实际样本 |

所以用真实标签当预测分（oracle，完美排序）也只能拿到：

| | random | FM baseline | **oracle 上限** | FM 已吃掉的区间 |
|---|---|---|---|---|
| GAUC | 0.4996 | 0.6610 | **1.0000** | 32.3% |
| nDCG@5 | 0.4511 | 0.5282 | **0.7289** | 27.8% |
| **primary** | 0.4753 | **0.5946** | **0.8645** | **30.7%** |

**评估进展请以 oracle 为分母。** 看到 0.5946 就以为「离满分 1.0 还很远」是误判——
baseline 已经吃掉可用区间的三成，剩余 headroom 是 0.27 而不是 0.41。

FM 在 5 个随机种子上的 std 均为 **0.0008**。据此收敛判据取 **ε = 0.002（≈2.5σ）, N = 3**：
连续 3 轮迭代 validation 主分提升不超过 0.002 即判定收敛。

> 自检：如果你的评测代码跑 `--model random` 得不到 primary ≈ 0.475（±0.001），说明 harness 有问题，先修它。

## 提交格式

CSV，含表头，一行对应评测集的一行：

```
row_id,user_id,video_id,score
0,0,7531,-3.34176
1,0,4214,-1.4955
...
```

| 字段 | 说明 |
|---|---|
| `row_id` | 0 起连续递增，对应 `data.load()[split]` 的行序（确定性：先读 `log_standard_4_08_to_4_21_pure.csv` 再读 `log_standard_4_22_to_5_08_pure.csv`，按 date 过滤后保持原文件顺序） |
| `user_id` / `video_id` | 冗余字段，仅用于校验对齐 |
| `score` | 你的模型给该行打的分，任意实数，只用相对大小；不允许 NaN / Inf |

> **为什么必须带 `row_id`：** `(user_id, video_id)` 在评测集里**不唯一** ——
> test 集有 3.06% 的重复对，最多重复 12 次。所以它不能作为主键。

生成与校验：

```bash
python3 submit.py --make  --split test  submission.csv    # 用官方 FM baseline 生成一份示例提交
python3 submit.py --check --split test  submission.csv    # 校验格式与对齐
python3 submit.py --score --split valid submission.csv    # 校验并打分（本地 valid 可用）
```

`--check` 会拒绝：表头错误、行数不符、`row_id` 跳号、`user_id`/`video_id` 与评测集不对齐、
`score` 非数字或为 NaN/Inf。**提交前请自行跑一遍 `--check`。**

## 从哪里开始改

下面的排序是**实测过的**，不是猜的。组委会已经试过的死路直接标出来，别重复踩。

### 已实测：这两条没有收益，不要浪费迭代

| 试过的 | 结果 |
|---|---|
| **加静态特征** —— 把 CWM 的 13 个特征域全接进来（+`music_id`/`video_type`/`upload_type` + 6 个用户侧粗桶） | primary **0.5940** vs 5 域的 **0.5950**，噪声内无差别，甚至略降 |
| **加模型容量** —— embedding 维度 k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887，几乎不动 |

原因：`user_id × video_id` 的交叉已经吃掉了大部分可学的信号。`follow_user_num_range` 这类粗桶
在 `user_id` 面前是冗余的；而 114 万行数据也撑不起更大的容量。**瓶颈不在特征和容量。**

⚠️ 另外注意：**纯用户侧特征的一阶项对分数贡献恒为 0。** 因为排序在用户内部做，任何在用户内为常数的项
都不改变组内顺序（实测：`item_pop × 用户偏置` 和纯 `item_pop` 的分数一位不差）。用户侧特征只能通过
**与物品侧的交叉项**起作用。

### 未探索：headroom 应该在这里

按我们判断的可能性排序（**这几条组委会没测过，是留给你们的**）：

1. **换损失函数。** 现在是 pointwise logloss，但指标（GAUC / nDCG）是**排序指标**。
   换成 pairwise（BPR）或 listwise（对该用户的曝光做 softmax）—— 目标函数和评测口径对齐，
   这是我们认为最可能有效的一条。
2. **用户历史序列。** 现有特征**完全没用到行为序列**。KuaiRand 每用户在 train 里有上百到上千条交互，
   DIN / SIM 那一类的兴趣建模是完全空白的方向。
3. **多目标。** 日志里还有 `is_click`、`is_like`、`is_follow`、`is_comment`、`is_forward`、`play_time_ms`，
   可以做多任务辅助 `long_view` 主任务。
4. **观看时长的建模。** [CWM](https://github.com/hyz20/CWM) 的贡献正是这条：它把观看时长做**删失回归**
   （视频播完时真实观看时长被截断，所以用单侧损失而非平方误差）。这是个有研究深度的方向。
5. **换模型。** DeepFM / DCN / xDeepFM。鉴于容量实测不是瓶颈，**优先级放在 1-4 之后**。
6. **时间特征与分布漂移。** `hourmin`、`date`，以及 train 与 test 之间的漂移。
7. **无偏验证（进阶）。** `log_random_4_22_to_5_08_pure.csv` 是随机曝光日志（118 万行），
   可作为额外的无偏验证集，检查模型是否只在有偏流量上过拟合。

## 用你自己的模型（包括 CWM）

`evaluate.py` 与模型完全解耦，它只要三个等长数组：

```python
from evaluate import evaluate
print(evaluate(user_ids, labels, scores))   # scores 可以来自任何模型
```

- `user_ids`：评测集每一行的 user_id
- `labels`：该行的 `long_view`（0/1）
- `scores`：你的模型给该行打的分（任意实数，只用相对大小）

所以你可以完全不用 `baseline.py`，换成 PyTorch、LightGBM 或 [CWM](https://github.com/hyz20/CWM) 的 xDeepFM，
只要最后把 `scores` 交给 `evaluate()` 即可。**评分口径由 `evaluate.py` 唯一决定。**

> 用 CWM 需注意：它依赖 `torch==1.6.0`（2020 年版本，新 GPU 上大概装不上），
> 且它的损失优化的是 counterfactual watch time、评测标签是自己重建的 `long_view2`。
> 它是一篇时长纠偏论文的研究代码，可以当**进阶参考**，不建议作为起步点。

## 自主研究循环（改编自 autoresearch-win-rtx）

`program.md` 把 [autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx)
（karpathy/autoresearch 的 Windows/消费级 GPU fork，让 agent 在固定 5 分钟 GPU 时间预算内
自主迭代 nanochat 的 `train.py`，以 `val_bpb` 为准绳，git 分支 + `results.tsv` 记录每次
keep/discard）改编到了这个仓库：agent 编辑的对象换成 `baseline.py`/`data.py`，准绳换成
`evaluate.py` 算出的 `valid` primary（GAUC 与 nDCG@5 均值，越高越好），`test` 只用来汇报、
不参与调参决策。跑一轮 FM baseline 在 CPU 上只要 ~40 秒，比原仓库 GPU 上的 5 分钟预算快得多，
所以循环预算按 5 分钟/次的上限给了很大余量。

把 `program.md` 交给一个 agent（比如把它整段贴给 Claude Code，或者 `/loop` 一个引用它的
prompt）即可启动：agent 会开一个 `autoresearch/<tag>` 分支，反复「改代码 → 跑
`baseline.py` → 读 `valid` primary → 达标 keep / 否则 discard」，把每一轮记到
`results.tsv` 里，直到人工打断为止。起步方向见上面「从哪里开始改」——`program.md` 里也
重复了一遍这份清单，避免 agent 重新踩已经踩过的坑（加静态特征、加 FM 容量）。

### LangGraph harness（`autoresearch_lg/`）

同一个循环的**代码版**：`program.md` 是喂给 Claude Code 的一段 prompt，
`autoresearch_lg/` 是一个真正的 Python 程序，用 [LangGraph](https://github.com/langchain-ai/langgraph)
把它编成一个 concept 驱动的研究 agent —— 三个子图（各自独立编译、可单独测试），加一个
决定 micro/macro、explore/exploit 走向的路由：

```
eda → propose → experiment → critic → router
        ↑                                │
        │        ┌───────────────────────┤
        │        │ error（retries 未耗尽）→ 直接回 experiment（同代码重跑，不经过 propose）
        │        ▼
        │   check_convergence ── done ──→ finalize（写 submission.csv + 资源报告）→ end
        └────────── continue（router 设好 mode 后）
```

- `propose`（brain，子图 read_mode → build_context → retrieve_options → llm_generate →
  validate_diff）：调用 LLM（默认 `gpt-5.5`；`claude-*` 名字会走 Anthropic），根据 router
  设定的 **mode** 生成下一轮实验。`mode=tune` 延续当前 active concept（换超参/修 bug）；
  `mode=expand` 是这个 concept 已经 tune 够了（`--tune-cap`，默认 3 次）后的宏观
  exploit，开一个相邻的新 concept；`mode=pivot` 是这个 concept 失败或跑不通
  （`--retry-cap`，默认 3 次错误后）之后的宏观 explore，换一个完全不同的方向。
  **mode 由 router 决定，具体提案永远由 LLM 决定** —— 硬编码"pivot 就必须做
  BPR"会把 agent 变成脚本化流程，这条边界是故意留着的。`validate_diff` 会真的
  `ast.parse()` 检查一遍生成的代码，语法错就重新生成一次。system prompt 刻意写得很短
  （核心规则 + baseline/oracle 数字，~600 token）——**不内嵌整份 README**：候选方向
  由 `retrieve_options` 每轮单独给，而不是把 190 行 README 复制进每一次调用里。
- `experiment`（do it safely，子图 apply_diff → run_and_evaluate → collect_metrics，
  每一步都有失败分支到 emit_failure）：**每轮实验都是全新的 `runs/exp_NNNN/` 文件夹**——
  写入提议的 `baseline.py`/`data.py`，外加一份固定的 `evaluate.py` 副本（Python 按脚本
  所在目录解析 import，所以文件夹要自成一体），然后**在这个文件夹里**跑
  `baseline.py --model fm`。仓库根目录的 `baseline.py`/`data.py` 从 `setup` 之后**再也
  不会被写入**——不是"写了再撤销"，是从来没被碰过。同一 concept 的 error 重试会换
  seed（0/1/2...）而不是原样重跑——固定种子下原样重跑一个确定性 bug 只会一直失败，
  换 seed 才让重试机制真正有意义。
- `critic`（judge it + log it，子图 compare_to_best → keep_or_revert → classify_outcome
  → update_counters → write_log）：跟当前最好成绩比较。`keep_or_revert` 不做任何文件
  操作——因为每轮实验本来就在自己的文件夹里，"revert"就是下一轮 propose 不再读这个
  文件夹而已，没有东西需要撤销或覆盖。分类成 `improved` / `failed` / `error` 三种结果，
  写日志。**`write_log` 是全部结构化日志唯一的写入点**：这一轮的 `exp_dir` 路径存进
  `checkpoints.db`（**SQLite，零新依赖的"tiny db"**，只是个可查询的索引——`iteration`/
  `concept`/`metrics`/`outcome` → 哪个文件夹，文件内容本身在文件夹里，不在数据库里）
  成为一行新记录，`runs.jsonl`（AIDE 风格，一行一条完整记录：concept、hypothesis、
  metrics、outcome、mode、error、tokens、wall-clock、exp_dir）、`results.tsv`（兼容旧
  格式）、`concepts.json`（每个 concept 的状态与尝试记录）都从这里落盘，随后重新生成
  `results_dashboard.html`。

**整个循环里没有任何 git commit / git reset。** 早期版本每轮实验一个 git commit，
discard/crash 就 `git reset --hard` 回上一个最好的 commit——这在实测中出过事故：
`reset --hard` 是对整个工作区生效的，如果工作区里刚好还有别的没提交的改动（哪怕跟
实验完全无关），会被一起吃掉；后来改成 SQLite 存文件内容、revert 时写回磁盘，仍然是
在原地覆盖 `baseline.py`/`data.py`。现在的版本更彻底：每轮实验从一开始就写在自己独立
的文件夹里，根目录的 `baseline.py`/`data.py` 除了 `setup` 那一次只读性质的 baseline
训练之外，**全程不会被这个循环写入**。`finalize` 收敛时也不做任何 git 操作——只是把
`submit.py` 复制一份到最好的那个实验文件夹里，在那里跑 `--make` 生成
`submission.csv`，写到仓库根目录（普通文件写入，不是 git 操作）。

`router`（读 outcome，决定 mode，`retry_count`/`tune_count` 这两个计数器驱动升级：
一直 error 就升级成 pivot，一直 tune 就升级成 expand）和 `check_convergence`
（`no_improve_count >= N`，或迭代/时间到上限）在图里各是独立的节点——不是为了好看，
是为了让 LangGraph Studio 里能看到这两次决策分别在哪一步发生，而不是折叠成一条不透明的分支。

换来三样东西：

- **图可视化，含子图内部**：`compiled.get_graph(xray=True).draw_mermaid()`
  （`python -m autoresearch_lg.cli graph`）连 propose/experiment/critic 内部的每一步
  都画出来，Studio 里可以直接点进子图看。
- **假设（concept）可追踪**：每个 concept 的状态（`active`/`closed`）、关闭原因
  （"expanded (maxed out after 3 tunes)" / "pivoted (no improvement)" 等）、每次尝试
  都在 `concepts.json` 里，跨 `cli.py run` 的多次调用持久化，恢复时会重建
  `retry_count`/`tune_count`/`no_improve_count`，不会因为重新起进程就悄悄放宽升级条件。
- **执行过程可视化 + 收尾**：`results_dashboard.html` 每轮实验后自动重生成——valid
  primary 曲线、concept 列表、完整明细表；跑到收敛时 `finalize` 节点自动调用
  `submit.py --make` 生成 `submission.csv` 和 `resource_report.json`（迭代数、耗时、
  token 用量、concept 统计）。**注意**：`submit.py` 是不让 agent 碰的固定文件，硬编码
  用 `baseline.FM` 类重新训练——只要 agent 的改动保留了 `FM` 的构造签名和
  `.step()`/`.predict()` 接口（loss/超参/特征类改动都满足），`finalize` 就能正常生成提交；
  换掉整个模型类（DeepFM 之类，headroom 里本来就排最后）会让这步失败，`finalize`
  会把失败原因写进报告而不是假装成功。

代价：需要自己的 `ANTHROPIC_API_KEY`（Claude Code 会话本身的凭据不能被子进程里的
Anthropic SDK 直接复用），多了 `langgraph` + `anthropic` 两个依赖 —— 注意这是
**编排层**的依赖，不是 `baseline.py`/`data.py` 的；"numpy + stdlib only" 那条约束
仍然只管评测用的模型代码本身没变。

**一键脚本**（`setup.sh` / `setup.ps1`，仓库根目录）：建虚拟环境、装依赖、从
`.env.example` 生成 `.env`（已存在则跳过，不会覆盖你的 key）、检查 KuaiRand-Pure
数据在不在。幂等，重复跑无害。

```bash
./setup.sh          # bash / git-bash / Linux / macOS
```
```powershell
.\setup.ps1          # PowerShell
```

跑完自己 activate 虚拟环境（脚本会打印确切命令），然后按脚本末尾提示继续。

**或者手动**（不想装到系统 Python 里 —— 会跟其它全局包版本冲突，而且 Windows 上
系统 Python 的 `Scripts/` 目录多半没在 PATH 上，装完 `langgraph` 命令直接
`command not found`）：

```powershell
# PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e . "langgraph-cli[inmem]"
```

```bash
# bash / git-bash
python3 -m venv .venv
source .venv/Scripts/activate   # Linux/macOS 用 .venv/bin/activate
pip install -e . "langgraph-cli[inmem]"
```

`pip install -e .` 装的是 `pyproject.toml` 里声明的依赖（`langgraph`、`anthropic`、
`python-dotenv`，Windows 上还有 `colorama` —— 没装的话 `langgraph dev` 会在启动时炸
`ValueError: Unable to configure formatter 'simple'`，因为它的日志渲染器在 Windows
上要靠 colorama 上色）。装进虚拟环境后 `langgraph`/`python` 等命令都在
`.venv\Scripts\`（或 `.venv/bin/`）下，激活虚拟环境后直接用命令名即可，不用管全局 PATH。

API key 只需要在仓库根目录放一个 `.env` 文件（复制 `.env.example`，填一行
`ANTHROPIC_API_KEY=...`）——`autoresearch_lg.cli` 和 `langgraph dev` 都会自动读它
（前者用 `python-dotenv`，后者是 `langgraph.json` 的 `env` 字段指的）。`.env` 已经在
`.gitignore` 里，不会被提交。

```bash
python -m autoresearch_lg.cli setup --tag aug29   # 开分支、跑一次 baseline 打底
python -m autoresearch_lg.cli run   --tag aug29    # 跑到收敛为止（默认 50 轮 / 6 小时上限）
python -m autoresearch_lg.cli dashboard            # 不跑实验，只重新生成看板
python -m autoresearch_lg.cli graph                # 不需要 API key，只打印图结构（含子图）
```

`run` 的循环现在真的在图里面，不是外面套一层 Python `for`（早期版本是后者——图结构
简单，但 LangGraph Studio 里看到的只是一条直线到 end，看不出「循环」在哪；`keep_or_revert`
之后的判断——同一个 concept 继续 tune，还是关掉它开一个新的——现在是图自己的环）。
`cli.py` 只对整个图做一次 `.stream()` 调用，跑到收敛（或 `--max-iterations`/
`--max-wall-hours`）图自己停并触发 `finalize`；Ctrl+C 依然安全，因为每轮实验完成时
`results.tsv` / `runs.jsonl` / `concepts.json` / `checkpoints.db` 已经落盘，中断只是
丢失还没跑完的那一轮，重新 `run` 会重建计数器接着跑——而且因为循环本身不再碰 git，
中断也不可能像早期版本那样连累到工作区里其它无关的改动。

**用 LangGraph Studio 交互式可视化**（比静态 mermaid 图更进一步 —— 能点进
propose/experiment/critic 三个子图内部看每一步、手动 invoke、查看每轮的输入输出）：

```bash
langgraph dev --no-browser        # 读 langgraph.json，起本地 dev server（虚拟环境已激活）
# 打开 https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

`langgraph.json` 里的 `graphs` 指向 `autoresearch_lg/graph.py` 里的模块级 `graph`
变量（用 `autoresearch_lg.graph:graph` 这种点号路径，而不是文件路径 ——
`graph.py` 内部用了包内相对 import，文件路径形式的 spec 会因为脱离包上下文而
`ImportError: attempted relative import`）。起 dev server 本身不需要
`ANTHROPIC_API_KEY`（只是导入、编译图）；在 Studio 里点 invoke 跑 `propose` 子图才需要。

**从终端跑，同时在 Studio 面板里看**：`cli.py run` 是直接在自己的 Python 进程里调
`compiled.stream()`，跟 dev server 完全无关，Studio 看不到它。`run_via_api.py` 走的是
dev server 的 REST API（`langgraph_sdk`），所以从终端起的这次跑会变成一个真实的 Studio
thread，可以打开链接实时看：

```bash
python -m autoresearch_lg.run_via_api --tag aug29 --model claude-sonnet-5
# 打印出 thread id 和一个可以直接打开的 Studio 链接
```

需要先起好 `langgraph dev`（上面那步）。

## 文件

| | |
|---|---|
| `evaluate.py` | 指标实现 + 全部口径约定。**不要改。** |
| `data.py` | 数据加载、官方划分、特征编码。加特征改这里。 |
| `baseline.py` | 三个 baseline。FM 是要打败的那个。 |
| `baseline_scores.json` | 官方发布的分数 + 种子方差 + 收敛参数。 |
| `submit.py` | 生成 / 校验提交文件。agent 不能改这个文件。 |
| `ablation_features.py` | 特征消融实验，可复现「加特征没有收益」那组数字。 |
| `program.md` | 自主研究循环的 agent 指令（改编自 autoresearch-win-rtx），配合 `/loop` 或长会话 agent 使用。 |
| `autoresearch_lg/` | 同一个循环的 LangGraph 代码实现（propose/experiment/critic 三个子图 + 主循环），需要 `ANTHROPIC_API_KEY` 或 `OPENAI_API_KEY`（看 `--model`）。 |
| `setup.sh` / `setup.ps1` | 一键建虚拟环境 + 装依赖 + 生成 `.env` + 检查数据，幂等。 |
| `runs/` | 每轮实验自己的文件夹（`exp_0001/` 等），各自一份 `baseline.py`/`data.py`/`evaluate.py`。根目录的 `baseline.py`/`data.py` 不会被这个循环写入。gitignored。 |
| `concepts.json` | `autoresearch_lg` 跑起来后生成——每个 concept 的状态/重试次数/最好成绩，`run` 命令之间持久化。 |
| `runs.jsonl` | `autoresearch_lg` 的结构化日志（AIDE 风格，一行一条完整记录），required deliverable。 |
| `checkpoints.db` | `autoresearch_lg` 的 SQLite 索引——`iteration`/`concept`/`metrics`/`outcome` → `runs/` 里对应的文件夹，不存文件内容本身。 |
| `resource_report.json` | `autoresearch_lg` 收敛后生成——迭代数、耗时、token 用量、concept 统计。 |
