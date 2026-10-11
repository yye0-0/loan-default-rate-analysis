# Loan Default Rate Analysis — Spark & Survival Extension

这是根据原 2024 年项目记录与上传原型重建的 **2026 年扩展版**。原型原样保留在 `legacy_v1/`；本目录的 `src/` 才是当前入口。它不是遗失源码的逐行恢复，也没有复现旧报告的 94.8%、99.71% 或“提升 35%”。

**当前状态：实际运行过 Spark、竞争风险建模、Spark AFT、Cox 诊断、原分类分支消融、ARIMA/LSTM，并完成边界测试。所有成绩来自模拟数据，真实贷款实验尚未完成。** 具体数量、测试日志与运行环境见 `VERIFICATION.json`；业内可用性评审见 `INDUSTRY_REVIEW.md`；结果解释见 `RESULTS.md`。

## 先运行

推荐 Python 3.12、Java 17。先将 `JAVA_HOME` 指向 Java 17；建立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-diagnostics.txt -r requirements-forecast.txt
python -m unittest discover -s tests -v
python run_project.py --demo --n 6000 --output results_demo --diagnostics --forecast
```

核心流程只需要 `requirements.txt`，不使用 `--diagnostics --forecast` 即可。CatBoost 可选：安装 `requirements-catboost.txt` 并加 `--catboost`。初次本地环境因缺少 `/proc/self/statm`，CatBoost 训练失败；该历史失败仍保留在 `verified_results/`。后续标准 Linux 运行已经完成 CatBoost 训练、评估和保存重载核验，见 `CI_VERIFICATION.json` 与 `ci_verified_results/`。

若实验要求 CatBoost 必须成功，使用 `--require-catboost`，它自动启用 CatBoost，并核对 12/24 月验证及测试指标和保存模型。任一期限跳过训练会以非零退出码结束；重新运行会先移除旧 `run_complete.json`，避免失败后留下上次成功的标记。进度和失败阶段记录在 `run_status.json`。

## GitHub 自动复现

`.github/workflows/verify.yml` 在 Pull Request、main 提交或手动触发时，使用 Ubuntu 24.04、Python 3.12、Java 17 安装锁定依赖，运行边界测试，重建 6,000 笔模拟贷款的完整 Spark、竞争风险、AFT、Cox 和 ARIMA/LSTM 流程，并要求 CatBoost 实际训练及评估完成。随后对四个阶段的完整 Spark 标签做独立核验，并验证保存模型在全部测试贷款上的推理。

每次运行的结果和日志保存为 GitHub Actions artifact，保留 7 天。运行失败同样保留已生成日志。该流程只使用仓库内生成的模拟数据；不会读取真实数据、账号或外部数据下载凭证。自动复现的成功状态以实际 Actions 记录为准，加入工作流本身不代表它已经通过。

首次标准 Linux 复现发现 CatBoost 1.2.7 缺少 scikit-learn 1.8 管道要求的 estimator tags。`src/catboost_compat.py` 用官方 `ClassifierMixin/BaseEstimator` 接口包装原生模型；样本权重、拟合和概率预测仍由 CatBoost 执行，未更换算法或训练参数。接口回归测试覆盖 Pipeline、权重传递、未拟合拒绝及保存重载。

标准 Linux 完整验证已通过：[GitHub Actions 运行记录](https://github.com/yye0-0/loan-default-rate-analysis/actions/runs/38113215709)。25 项测试成功；四阶段完整标签一致，联合模型及 CatBoost 两期限模型在全部 733 笔测试贷款上的保存重载预测通过核验。`ci_verified_results/` 保留此次关键报告、冻结选择、版本、测试日志和 CatBoost 模型，属于精选证据目录；完整新运行可用上述命令重新生成。CatBoost 在这次模拟测试中没有优于简单 FICO Logistic，详细对照见 `RESULTS.md`。

```bash
python -m unittest discover -s tests -v
python run_project.py --demo --n 6000 --diagnostics --forecast --require-catboost --output results_ci
python src/verify_outputs.py --results results_ci --require-catboost
```

`verified_results/` 是本次实际完整运行的结果；不要在这份证据目录上覆盖运行。请为每次新实验选择新输出目录。训练和测试样本数量由输出日志确定，不保证改变随机生成器后与旧结果相同。

## Spark 到底做了什么

Spark 是用于并行处理数据的计算引擎。本项目确实调用 Spark 执行了下列步骤，而不是仅保留一个 import：

1. 读取放款属性和月度表现；检查列宽、ID、月份和重复记录，规范缺失值。
2. 使用连接和窗口函数确定首次观察、首次事件、观测缺口和删失时间。
3. 按每个历史观察截止日重新生成标签，输出全量 Parquet 和样本审计表。
4. 用 Spark `sequence/explode` 生成贷款—月份风险集；为本地模型导出每阶段最多 6,000 笔贷款。
5. 另外实际训练 Spark MLlib 的 Weibull AFT，并在 Spark 中构造日历月风险率。

本次运行是 `local[2]`，不是多机集群部署。6,000 笔贷款主要用于验证实现，不能证明 Spark 对这个规模比 pandas 更快。真实价值是同一 ETL 可扩展到完整月表；完整规模的耗时、分区、内存和集群性能尚未测量。竞争风险与 Cox 模型在本地训练，不称为分布式模型训练。

## 两条分支与原项目对应

| 分支 | 保留的方法 | 解释边界 |
|---|---|---|
| 原分类分支 | 还款压力、FICO、信用历史、Logistic、RF、可选 CatBoost、PCA/KMeans、组合风险分数和 CAPM 风格等级代理的消融 | 原 `isDefault` 是最终状态，观察期限未知；只能做回顾性批次外推，不能称为 12/24 月部署 PD |
| 月度风险分支 | Spark ETL、竞争风险、12/24 月校准及 IPCW 评价、Spark AFT、Cox/PH 诊断、ARIMA/LSTM 对照 | Freddie 为抵押贷款，与原消费贷产品不同，不能跨数据任意拼接或比较成绩宣称业务提升 |

原组合分数不使用目标 `isDefault`，也不叫标准 Altman Z-score。等级映射代理不是以收益协方差估计的 CAPM beta。默认分类模型不纳入这两个启发式分数；通过独立消融检验是否值得加入。PCA 按训练集实际计算保留 95% 方差所需维度。PPT 中地理展示、Q-learning 和团队个人分工仅作为历史记录，不宣称此次已经重建。

```bash
python src/classify.py --csv demo_data/tabular_synthetic.csv \
  --train-end 2014-12-31 --valid-end 2015-12-31 \
  --output results_ablation --synthetic --ablation
```

## 预测时点与事件合同

时间零是贷款 **第一条可观察月度记录**，且当时逾期状态为 0/00、无退出、无已有修改标志；不是贷款发放时点，也不是后来挑选一个恢复正常月份重新入组。预测之后的月度区间，使用原始属性和入组贷款年龄。第一条记录已经逾期、退出或修改的贷款有明确排除记录。首付月份前的记录允许进入，入组年龄下限为 0。

| event_type | 首次结局 | 处理 |
|---|---|---|
| 0 | 行政截止、未知状态、观测缺口、尚未发生终止事件的修改 | 右删失；缺口或未知月之前的有效月份保留 |
| 1 | 数字逾期档位 ≥3，即首次观察到 D90+ | 主目标；同月与退出并存时优先 D90 |
| 2 | 零余额码 01 | 提前结清／到期，作为竞争事件 |
| 3 | RA，或零余额码 02/03/09 | 另一个信用困境退出目标 |
| 4 | 其他退出，包括 15/16/96 | 其他竞争事件；15/16 不直接等于违约 |

修改标志本身不是信用违约。D90 之后出现 RA 或修改不能抹掉此前事件。缺口处不会补造未观测月份，模型不会把最后观测时仍正常的贷款当成永远不会违约。

`cif_1` 是现实竞争结局下的累计 D90 概率；`credit_stress_proxy = cif_1 + cif_3` 是更广的信用困境代理，不是法律违约或最终损失。其余 CIF 和事件未发生概率共同加总为 1，各 CIF 的 24 月概率不小于 12 月。Spark AFT/Cox 的 `net_risk` 是竞争结局删失后的净风险，不可直接当作 CIF。

月份均为月初日期标签，例如 `2020-01-01` 表示报告月份，不代表数据在该日已发布。当前验证使用报告月份的观察截止，尚未模拟发布延迟或历史数据修订；需要实际版本快照后才能称为严格的实时回测。

## 时间切分和评估

| 阶段 | 首次观察月份 | 可用结果的报告月份截止 | 用途 |
|---|---|---|---|
| train | 2007–2015 | 2015-12 | 预处理和模型拟合 |
| calibration | 2016 | 2019-12 | 拟合概率校准 |
| validation | 2017 | 2019-12 | 冻结模型选择 |
| test | 2020–2021 | 2023-12 | 一次留出评价 |

2018/2019 不进入个体模型训练或测试；这些月可用于独立的日历序列分析。每阶段独立按截止重建标签，跨阶段贷款 ID 不重叠。中位数填补、缩放、月龄编码只在训练集拟合。

主模型使用五类月度 multinomial hazard，逐月累计竞争风险。联合校准在月度 logits 上拟合一个正温度和四个原因偏移，再以验证集 12/24 月平均 IPCW Brier 选择一条完整曲线；不会分别校准两个累计 PD 而破坏期限一致性。

对照包含常数、FICO Logistic、完整 Logistic、HistGradientBoosting 和可选 CatBoost。固定期限对照可以独立 Platt 校准，因此会报告期限交叉审计；用户推理入口使用联合曲线。评价包括 IPCW Brier、加权 AUC/AP、校准分箱、前 5% 容量的事件捕获、FICO/入组年龄子组、100 次贷款 bootstrap 区间。该 bootstrap 条件于已拟合模型，不含重新训练不确定性。

删失 KM 对同月先处理终止事件、后处理行政删失；观测至第 h 月的正常贷款在 h 月仍是已知结局。估计 G 低于 .05 时停止该期限的评估，不用无支持区间选模型。当前使用阶段内总体删失模型，依赖独立删失假设；缺口和修改可能造成信息性删失。

## 保存模型后的推理

```bash
python src/predict_risk.py \
  --model verified_results/risk/joint_model.joblib \
  --csv verified_results/risk/inference_example.csv \
  --output predictions.csv
```

输入每贷款一行：`loan_id, fico, dti, ltv, interest_rate, loan_amount, entry_age_months`。只用于上述首次正常观察的预测时点；年龄为已知非负月数。输出 12/24 月四个 CIF、事件未发生概率、信用困境代理和数据标记。本次已用保存后模型核验所有测试贷款的概率守恒与期限单调性。仅加载自己信任的 joblib 模型文件。

## 接入真实数据

官方入口：https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset 。完整 Standard 数据需要注册登录；本包不附真实原始数据或账号。当前公开样本 ZIP 请求返回 HTTP 403，记录见 `DATA_READINESS.json`，没有完成真实匹配月表的训练。

先在本地下载同一来源批次的 origination 与 monthly performance 文件，核对 July 2026／旧版数据字典。`--freddie-version july2026` 检查每行 31/35 列；`legacy` 只支持这里使用的早期基础位置，须对照实际文件字典。Jul 2026 适配已用供应商格式模拟文件实际验证，不等于真实数据业务验证。

复制 `verified_results/synthetic_inputs/config.json`，将 `data_kind` 改为 `USER_SUPPLIED_DATA`，按实际覆盖区间调整日期；保持训练截止早于校准期、选择截止早于测试期。调整切分前先确认各组事件、竞争结局、观察期及 G 支持。原始 ID 必须一对一连接；不同年份文件可先按官方同模式合并为对应的两个输入文件。

```bash
python run_project.py \
  --orig /absolute/path/orig.txt --monthly /absolute/path/perf.txt \
  --config /absolute/path/config.json --freddie-version july2026 \
  --output results_real --diagnostics
```

也可直接提供 canonical CSV：放款表字段为 `loan_id, first_payment_month, fico, dti, ltv, interest_rate, loan_amount`；月表为 `loan_id, report_month, dpd, zero_code, modification`。ID 保持字符串；比例和利率用百分数，例如利率 5；日期为月初标签；DPD 的 0/1/2/3 分别为当前/30+/60+/90+。未知状态用 XX，不转换成 0。默认规范 FICO 9999、DTI 999、LTV 999 等为缺失，由训练期填补。

Spark 全量标签和审计保留在 `spark/`；本地建模每阶段上限 6,000，按 ID 确定性选取以限制内存，真实扩展应改为可审计分层抽样或分布式模型，不能把局部模型结果称为全组合训练。实际规模扩展和历史版本发布时点核验属于真实实验的下一阶段。

## 日历序列

Spark 聚合首次正常观察后的 D90 事件／月初风险集，并以训练截止前风险暴露确定固定贷款年龄权重。至少 100 笔风险集、四个年龄组各至少 10 笔；只允许裁去开头无支持月份，不跨内部缺口训练。比较最后值、季节朴素、ARIMA(1,0,1)、ARIMA 残差 LSTM，使用两个滚动 12 月验证窗口选择，随后评价最后 12 月。该序列是有条件的观察风险率，不等于整体资产池违约率。

## 文件与归属

`tests/verification_run.log`、`verified_results/logs/`、`verified_ablation_final/` 和 `VERIFICATION.json` 对应本次执行。`legacy_v1/` 为历史原型，原始说明及 demo 分数不可混作当前结果。当前扩展独立编写，未复制 GitHub 代码或借用别人的模型分数。旧原型中的 GitHub 链接只是参考方向，不能据此宣称复现其仓库。简历个人贡献必须符合原团队真实分工；2026 扩展不可回写成 2024 已完成。

官方方法／字段参考：

- https://spark.apache.org/docs/3.5.7/ml-classification-regression.html
- https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf
- https://www.freddiemac.com/fmac-resources/research/pdf/disclosure-changes-summary.pdf
