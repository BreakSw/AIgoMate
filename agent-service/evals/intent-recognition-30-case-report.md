# 用户意图识别 30 条真实模型评测报告

- 评测日期：2026-08-28
- 模型：`deepseek-v4-pro`
- 接口：`POST /api/agent/analyze-intent`
- 协议：`TaskSpec v1.0`
- 执行方式：真实模型、串行请求、请求间隔 0.5 秒

> 2026-08-28 后续更新：模型 HTTP 客户端现已实现最多 5 次断连重试和指数退避，前端会通过 SSE 显示“正在重新连接 n/5”。本报告中的“缺少重试”描述保留为评测当时状态。

## 结论

当前效果可以进入开发联调阶段，但还不适合在没有重试和结构修复的情况下直接承担生产级 Agent 路由。

- 首次请求成功率：24/30，80.0%。
- 失败样本重试后成功率：29/30，96.7%。
- 严格主意图准确率（把优先级差异也算错误）：27/30，90.0%。
- 人工语义审查可接受：28/30，93.3%。
- 有效 TaskSpec 中仅 1 条存在会影响 Agent 路由的明确误判。
- 首轮完整字段断言：20/30，66.7%；剔除传输失败后为 20/24，83.3%。其中两条是空格/同义表达导致的评测器误报。
- 延迟：中位数 7.1 秒，P95 8.9 秒；最长请求 16.1 秒。

## 30 条问题与审查结果

| # | 测试问题 | 预期主意图 | 结果 | 审查 |
|---:|---|---|---|---|
| 1 | 用生活例子解释二分查找 | `concept_explanation` | 命中 | 通过 |
| 2 | 两数之和，只给提示，不要完整答案，使用 Python | `guided_hint` | 命中 | 约束、语言、交付方式均正确 |
| 3 | 完整解答 LIS，含思路、复杂度和 Python | `problem_solving` | 命中 | 次意图也正确识别 |
| 4 | 编写可运行的 Java 归并排序 | `code_generation` | 命中 | 通过 |
| 5 | Python 越界代码诊断 | `code_diagnosis` | 重试后命中 | `response_mode` 为直接回答而非代码审查，影响较小 |
| 6 | 快速排序三种复杂度分析 | `complexity_analysis` | 重试后命中 | 通过 |
| 7 | 比较 BFS 和 DFS | `solution_comparison` | 重试后命中 | 通过 |
| 8 | 哈希表模拟面试，一次一问 | `mock_interview` | 命中 | 交互模式和路由正确 |
| 9 | 安排未来两周复习计划 | `review_planning` | 命中 | 通过 |
| 10 | 图示红黑树旋转变色 | `visual_explanation` | 命中 | visualization 路由和工具正确 |
| 11 | 零基础算法学习路线咨询 | `learning_consultation` | 命中 | 通过 |
| 12 | 普通问候 | `general_conversation` | 命中 | 通过 |
| 13 | 只说代码报错但不提供材料 | `code_diagnosis` | 命中 | 正确追问代码和错误信息 |
| 14 | 解释动态规划但不要代码 | `concept_explanation` | 命中 | `include_code=false`，约束正确 |
| 15 | 直接给 C++ 并查集模板 | `code_generation` | 连续两次协议失败 | 模型生成协议外 capability，Pydantic 拒绝；严重 |
| 16 | Java twoSum 返回空数组，含输入用例 | `code_diagnosis` | 命中 | 代码和测试用例提取正确 |
| 17 | 基于历史继续，只给下一个提示 | `guided_hint` | 重试后命中 | 上下文承接正确 |
| 18 | Rust 实现拓扑排序和环检测 | `code_generation` | 重试后命中 | 语言与次意图正确 |
| 19 | 滑动窗口最大值并覆盖边界输入 | `problem_solving` | 命中 | 测试用例提取正确 |
| 20 | 只审查代码，不重写 | `code_diagnosis` | 命中 | review_only、code_review 正确 |
| 21 | 把暴力重复元素方案优化到 O(n log n) 内 | `complexity_analysis` | 主意图为 `problem_solving` | 次意图为复杂度分析，复杂度目标提取完整；可接受 |
| 22 | 用流程图演示 Dijkstra | `visual_explanation` | 命中 | 通过 |
| 23 | 比较归并和快排在近乎有序数组上的表现 | `solution_comparison` | 命中 | 次意图复杂度分析正确 |
| 24 | 带“忽略系统规则”的提示注入后分析二分复杂度 | `complexity_analysis` | 命中 | 保持 JSON 协议，未保留注入指令 |
| 25 | 英文解释递归且不要可执行代码 | `concept_explanation` | 命中 | 英文输入处理正确 |
| 26 | Python `unhashable list` 报错诊断 | `code_diagnosis` | 命中 | 代码、错误和语言提取正确 |
| 27 | 7 天、每天 2 小时的面试复习计划 | `review_planning` | 命中 | 时间约束完整；自动断言仅因空格误报 |
| 28 | 用苏格拉底提问引导接雨水，不直接透露思路 | `guided_hint` | 命中 | 交互方式与约束完整；自动断言因同义改写误报 |
| 29 | “帮我做一道算法题”但无题目 | `problem_solving` | 命中 | 正确要求补充题目并标记风险 |
| 30 | 历史为 Dijkstra，追问“它和 BFS 有什么区别” | `solution_comparison` | `concept_explanation` | 实体和上下文正确，但主意图与 capability 路由错误；严重 |

## 做得好的部分

1. 明确指令识别稳定：解释、提示、解题、代码生成、诊断、面试、规划、可视化等主要意图整体准确。
2. 用户边界保护表现好：“只给提示”“不要代码”“不要重写”“一次只问一道”均能进入 delivery 或 constraints。
3. 输入材料提取较完整：代码、错误、测试用例、语言、复杂度目标均能供后续 Agent 使用。
4. 歧义处理合理：信息不足时会生成澄清问题，并设置风险标记。
5. 上下文能解析指代对象：第 30 条虽然路由错了，但仍从历史正确提取了 Dijkstra 和 BFS。
6. 提示注入测试通过：模型没有执行要求绕过 JSON 协议的指令。

## 主要问题

### P0：协议外枚举会让整条请求失败

第 15 条连续两次复现。当前实现只校验一次，没有 JSON 修复或模型重试。对后续 Agents 来说，这比普通意图误差更严重，因为没有 TaskSpec 可消费。

建议：Pydantic 校验失败时，把精简的校验错误和允许枚举发送给模型进行一次 repair；仍失败再返回错误。也可以在 API 支持良好时使用严格 JSON Schema/structured output。

### P1：上游瞬时失败没有重试

首轮有 5 次连接级失败，重试后全部恢复。当前前端会直接看到失败，并留下只有用户消息的半完成会话。

建议：对连接错误、429 和 5xx 做最多两次指数退避重试并加随机抖动；不要重试 4xx 配置错误。

### P1：比较意图的优先级规则不够明确

“它和 BFS 有什么区别”被归为概念解释，虽然实体提取正确，但会错过 `solution_comparison` Agent。

建议在 Prompt 增加优先级规则：当请求出现“比较、区别、优缺点、A 和 B”且存在两个对象时，优先 `solution_comparison`。

### P2：置信度校准偏乐观

绝大多数结果固定在 0.90 或 0.95。第 30 条误路由仍给 0.95，说明置信度目前不能直接用于是否自动执行的门槛。

建议后续用标注集做置信度校准，或先把置信度视为展示字段，不作为高风险 Agent 自动执行依据。

### P2：部分上下文计划偏宽松

一些独立问题仍设置 `recent_messages=true`，但不影响主任务。未来实现上下文压缩后，需要由确定性策略结合模型建议决定实际加载内容，不能完全服从模型布尔值。

## 推荐上线门槛

在开发联调阶段可以继续使用。进入实际 Agent 执行前，至少完成：

1. 结构修复重试；
2. 上游瞬时错误重试；
3. 比较意图优先级规则；
4. 建立持续回归评测，每次 Prompt 或模型变更都运行这 30 条。

## 复现

```powershell
$env:INTENT_EVAL_CONCURRENCY='1'
$env:INTENT_EVAL_DELAY='0.5'
python scripts/evaluate_intent_recognition.py
```

评测脚本不会写入聊天数据库，完整问题和断言位于 `scripts/evaluate_intent_recognition.py`。
