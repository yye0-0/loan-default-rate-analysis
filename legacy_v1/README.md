# Loan Default Rate Analysis 重建与扩展

本包根据你提供的 2024 年 PPT 与 Word 记录，重新实现原项目主线，并增加 Spark 与生存分析入口。它不是遗失代码的逐行恢复，也没有复现原始成绩。代码由本次任务重新编写，公开 GitHub 项目仅用于参考架构与研究方向，没有复制其源代码或借用其成绩。

**版本：2026 年 10 月重建原型。当前可验证的是合成数据上的功能流程。真实数据实验尚未运行；Spark、CatBoost、Cox、LSTM 和 ARIMA 入口当前未做运行验证。**

## 1 保留同一个项目的两条数据分支

| 分支 | 数据 | 研究问题 | 原项目内容与扩展 |
|---|---|---|---|
| A 原项目重建 | 找回天池风格 train.csv 或重新下载同来源数据 | 贷款是否发生违约 | 还款压力、FICO、信用历史特征，组合风险分数，CAPM 风格等级代理，Logistic / Random Forest / CatBoost，PCA / KMeans，示意信用评分 |
| B 月度表现扩展 | Freddie Mac Standard Dataset 的放款属性与月度表现 | 首次严重逾期发生在第几个月 | Spark 读取、连接、月度聚合、Parquet、Weibull AFT；可选 Cox；提前结清作为竞争事件的经验累计发生率；日历月风险率的 ARIMA + LSTM |

这两条分支共享信贷风险主题和验证原则。不同产品、样本、目标和时间口径的结果分别报告，不能把两套数据按任意 ID 拼成训练集，不能直接以指标差异宣称技术改进。Freddie Mac 是抵押贷款，与 Monee 的消费信贷产品不同；项目展示的是可迁移方法，不能声称已有钱包交易或催收建模经验。

原始字段 loanAmnt、ficoRangeLow、earliesCreditLine、isDefault 和约 80 万样本，与天池金融风控学习赛的数据风格高度吻合。这是依据记录的推断，附件没有明确给出下载来源，因此不能将其写成已确认事实。优先找原数据下载记录；若找不回，应记为同题材重建。

## 2 对历史记录的核对

| 附件记录 | 重建处理 |
|---|---|
| Z-Score 中 X5 为是否违约 | 原特征包含目标信息，不能在申请时使用。本包完全移除该项，另写 risk_composite，并声明新权重属于启发式选择，不称为 Altman Z-score |
| CAPM 将 subGrade 映射成 beta，Rf=3%，Rm=10% | 保留风险溢价启发式思路。新代码使用 grade 的固定映射，命名 capm_grade_proxy。这不是由资产收益协方差估计出的 CAPM beta，不是经过验证的贷款定价模型 |
| fico_default_rate 与 target encoding | 最小原型不使用这些特征。以后加入时，只能以训练折的历史标签计算，并在验证/测试上应用冻结映射，不能全表预计算 |
| Word 的 94.8% accuracy | 来自 Random Forest 段落。属于历史报告值，原数据与代码缺失，且特征设计存在泄漏，需要重新验证 |
| PPT 第 40 页 accuracy=0.9971、precision=0.9967、recall=0.9888、F1=0.9928 | 属于 CatBoost 历史记录，与 94.8% 不是同一个模型结果；本包不承诺复现这些数字 |
| 简历所称风险检测提升 35% | 未在附件可提取文字中找到比较基准或计算口径。图中也可能有额外信息；在补齐证据前，不作为已验证指标 |
| PCA 五维保留 95% 方差，随后列出的五项比例 | 五项求和为 39.56%，与文字矛盾。新版按训练数据计算实际维度与累计解释方差 |
| KMeans 四类客户 | 保留四类作为与原记录对应的研究设定。类别是描述性分群，不能预先认定聚类编号具有稳定风险顺序 |
| 原 300–850 信用评分 | 保留示意映射 850−550×PD，与 FICO 不等价，也没有外部行业校准 |
| LSTM + ARIMA | 保留独立时间序列研究分支，以月度首次严重逾期事件数/风险集人数构造序列，避免把最终违约状态按放款月份聚合后当作实时风险率 |
| PPT 的地理可视化、Q-learning 定价 | 保留为历史模块。当前代码没有重建地图或强化学习，不应在此次扩展中宣称已实现。Q-learning 收益验证需要可辩护的模拟环境与约束，不宜为了凑模块直接添加 |

PPT 按成员标注了不同模块负责人。本包描述的是团队项目整体，简历中个人贡献应按你真实负责的部分写。附件能证明这些方法曾出现在团队记录中，不能自动证明你个人实现了全部模块。

## 3 数据和标签合同

### 原项目 CSV

必需列：id、isDefault、issueDate、loanAmnt、term、interestRate、annualIncome、dti、ficoRangeLow、ficoRangeHigh、grade。

term 使用年数 3/5，interestRate 与 dti 使用百分数数值，例如 12 表示 12%。36/60 月期限必须先转换。earliesCreditLine 请先规范为日期，不能把无法解析的文本直接当数值。可选列：revolUtil、homeOwnership、verificationStatus、purpose、applicationType。

isDefault 来自原表的最终状态，没有事件日期和观察截止日时，只能报告该数据集的分类任务，不能声称它是 12 个月 PD 或真实部署的回测。按 issueDate 切分可验证放款批次外推，但如果训练标签使用了测试时间之后才发生的结局，仍不是严格的 as-of 回测。后续应补齐标签可用日期、固定预测期限和成熟期。

### 生存分析标准 CSV

orig.csv 每笔贷款一行：

| 字段 | 定义 |
|---|---|
| loan_id | 同一来源中的贷款唯一 ID |
| cohort_date | 风险区间第 1 月的日历月份，ISO 日期；Freddie 分支取 first payment month，不能写成准确放款日 |
| fico | 预测时点可用的原信用分 |
| dti、ltv、interest_rate | 原始比例或利率，使用百分数数值 |
| loan_amount | 原贷款金额 |

monthly.csv 每笔贷款每个月一行：loan_id、month_index、delinquency_months、exit_type。

month_index 为从 cohort_date 开始的 1-based 月数。delinquency_months 为数值逾期档位：0/1/2/3 对应未满30/30+/60+/90+天。exit_type 为 active、payoff、other。

目标明确为 **首次观测到 90 天及以上逾期**，不是已核定损失、破产或所有法律定义的违约。每笔贷款在首次该事件、全额结清、其他退出或最后观测月停止。行政截止时仍正常的贷款记为删失，不当成永远不会违约。默认同月首次严重逾期与 payoff 同现时事件优先；真实数据中须检查这类冲突并做敏感性分析。

原型剔除未从第 1 月连续观测的贷款，并输出排除名单。这样避开未经处理的延迟入组和月度缺口，但可能导致样本选择偏差。正式分析应评估剔除比例，或对 Cox 使用 entry 参数处理左截断，不能隐瞒排除后样本结构改变。

Cox/AFT 把 payoff 当作删失时估计的是原因特异/净风险，1−S(t) 不能直接解释为现实中包含提前还款的违约累计概率。panel.py 的离散 Aalen–Johansen 表同时保留事件 1（严重逾期）与事件 2（结清），提供经验累计发生率；它是描述性估计，不是个体竞争风险预测模型。

### Freddie 原始文件入口

先从官方下载 Standard Dataset 的同一批次 origination 与 monthly performance 文件。需要注册下载，不包含在本包中。不要发布受数据条款约束的原始数据。

spark_panel.py 的 --freddie 模式使用 2026 User Guide 的列位置，读取不带表头的 | 分隔文件。**运行前请对照你下载版本的数据字典**：origination 中信用分第1列、首次支付月份第2列、DTI第10列、UPB第11列、LTV第12列、利率第13列、贷款ID第20列；monthly 中贷款ID第1列、月度月份第2列、逾期档位第4列、修改标志第8列、零余额退出代码第9列。

本入口将零余额代码 01 记为 payoff，非空且非00的其他退出记为 other，不将其直接当作90天逾期。RA/未知逾期或修改贷款目前保守剔除。信用分9999、DTI999等哨兵不能按正常值处理。代码输出的是严格D90目标，因此未经历观测D90便发生其他信用退出的贷款不会计为D90事件；若研究更广的信用违约，应新增独立事件定义而不是默默改变标签。

## 4 代码入口

| 文件 | 功能 | 当前验证状态 |
|---|---|---|
| src/features.py | 防泄漏白名单特征、还款压力、风险组合、CAPM风格代理 | 合成数据与单元测试通过 |
| src/classify.py | Logistic / RF，可选 CatBoost；验证集选阈值；时间留出；PCA / KMeans；示意评分 | Logistic / RF 与分群流程通过；CatBoost未执行 |
| src/panel.py | 月度事件聚合、删失、竞争事件累计发生率、日历风险率 | 合成数据与边界测试通过 |
| src/survival.py | SciPy Weibull AFT，原因特异风险；可选 lifelines Cox | AFT通过；Cox未执行 |
| src/spark_panel.py | Spark读取和清洗、月表连接、窗口函数、Parquet、Spark AFT | 已完成语法检查，未运行 |
| src/forecast.py | ARIMA基线与ARIMA残差LSTM，独立未来12月留出 | 已完成语法检查，未运行 |
| src/make_demo.py | 固定随机种子的合成数据 | 已执行；不代表真实贷款数据 |
| tests/test_core.py | 泄漏不变性、事件/结清/缺口处理、AFT方向与C-index | 3项测试通过 |

Spark 与 Python AFT 的事件列都是1表示已发生、0表示删失；Spark 参数虽名为 censorCol，语义不能反写。Spark 内置模型为 Weibull AFT，本包不会称它为 Spark Cox。当前C-index采用小样本Harrell C并记录样本数，不是IPCW C-index。

## 5 如何运行

推荐 Python 3.11、Java 17、PySpark 3.5.x。新建虚拟环境后安装 requirements.txt。依赖文件给出兼容区间，尚未形成全环境锁定；可先在个人环境验证，再冻结版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python src/make_demo.py --output demo_data --n 2400
python src/classify.py --csv demo_data/tabular_synthetic.csv --train-end 2014-12-31 --valid-end 2015-12-31 --output demo_results/classification --synthetic
python src/survival.py --orig demo_data/orig_synthetic.csv --monthly demo_data/monthly_synthetic.csv --train-end 2014-12-31 --valid-end 2015-12-31 --output demo_results/survival --synthetic
spark-submit --master local[2] src/spark_panel.py --orig demo_data/orig_synthetic.csv --monthly demo_data/monthly_synthetic.csv --train-end 2014-12-31 --valid-end 2015-12-31 --output demo_results/spark
```

真实数据：将原分类CSV传给 classify.py，移除 --synthetic，加入 --catboost。按实际放款月份选择训练/验证边界。生存标准CSV可交给 survival.py，加入 --cox。

Freddie原始文件：

```bash
spark-submit --master local[2] src/spark_panel.py --freddie --orig data/origination.txt --monthly data/performance.txt --train-end 2016-12-31 --valid-end 2017-12-31 --output results/freddie
```

以上2016/2017只是边界示例，实际样本年份要覆盖三个分区并保证事件和足够观察期。避免一次收集全部贷款月度数据到 pandas。先处理样本批次，再扩大规模；Spark输出保留为Parquet。survival.py是本地参考实现，不能把其to pandas的能力说成分布式Cox。

如果标准月表包含足够连续日历月份，可以运行：

```bash
python src/forecast.py --csv demo_results/survival/calendar_hazard.csv --output results/forecast
```

合成数据的末端可能样本稀少，这条命令仅说明接口。正式预测须过滤小风险集、说明截止时点与数据修订。原型固定ARIMA(1,0,1)，训练残差LSTM，并与最后值基线比较；未做滚动调参、置信区间或模型优越性结论。

## 6 实验完成的证据要求

1. 记录真实数据来源、版本、下载日期、原始文件哈希、样本数量和产品类型。原数据来源若未确认，应保留推断标记。
2. 对Spark与本地标签聚合做逐贷款一致性检查，人工抽查D90、删失、结清、修改和缺口样本。再扩大到实际规模，记录运行时间、机器、分区数与内存。不把local[2]演示称为集群生产部署。
3. 分类报告ROC-AUC、Average Precision、Brier、召回率与阈值。准确率只能辅助报告。使用成熟的固定期限目标后再比较12月PD模型。
4. 比较加入与移除风险组合分数/CAPM代理的消融版本；真正进行训练内OOF编码后再加target encoding。模型成绩可能下降，这能反映移除泄漏后的真实难度。
5. 生存分析检验Cox比例风险假设、删失比例、风险集数量，补IPCW C-index、时间依赖指标、12/24月校准和bootstrap置信区间。当前入口没有这些完整评价，不能宣称已完成。
6. 竞争风险建模进一步训练严重逾期与结清的两个原因模型，计算个体累计发生概率，或使用适当多状态模型。不要把当前净风险预测更名为实际累计PD。
7. LSTM/ARIMA按过去训练、未来评价并与简单基线比较。短序列可能不支持LSTM，模型名称多不等于效果好。
8. 输出研究报告：数据、目标、特征可用时点、方法、真实实验表、限制。GitHub归属说明、贡献边界和AI辅助重建说明应保留。

## 7 简历与面试表述

项目可继续叫 Loan Default Rate Analysis，不必另造一个无关项目。日期最好写成两阶段：2024 Course Project / 2026 Independent Extension，或项目标题下明确注明2026扩展。不要让招聘方理解Spark和生存分析在2024已经完成。

当前只能说“正在重建原课程项目，并加入Spark与生存分析实验”。完成真实数据清洗、Spark运行与模型验证后，才可使用下面的未来版本：

> Extended a 2024 loan-default study with PySpark processing of monthly loan performance data and survival modeling of time to serious delinquency, using temporal holdouts and explicit censoring and prepayment rules.

这句话仍需你实际完成并能解释代码后才能使用。若保留Z-Score/CAPM，应写成“evaluated custom borrower-risk composites and CAPM-inspired grade proxies”，不能写成已验证的标准破产预测或资本市场定价方法。

真实实验结果出来后，简历填实际贷款数、loan-month规模、评价指标和运行环境。本包的所有demo成绩必须留在合成测试标记下，不能转成简历数据。94.8%只可注明旧报告来源与未复现状态；35%的基准找不到时，应删除或补证据。

面试主线可以是：原课程研究识别贷款风险；代码遗失后依据报告重建；复查发现目标泄漏并修正；用月度记录扩展到“何时出现风险”；通过Spark ETL和时间留出评估验证。研究纠错与可复现的过程本身很有价值。

## 8 参考来源

- 原团队记录：Data Mining Group 3(1).pptx、part2+3(1).docx。附件中的模型和个人分工属于历史证据。本包未复制整份附件，以免混同重建代码与旧源码。
- https://github.com/yatshunlee/lending-club-credit-risk-analysis ：参考PySpark ETL与放款时间外验证的项目结构；README可读，notebook正文此次未成功解析。
- https://github.com/thesandipan/credit-risk-modelling ：README列有04_pd_survival.ipynb的Cox分支；此次未完整读取其notebook，不声称已复现或逐行整合。
- https://github.com/BDEvan5/credit-risk-ml ：Home Credit信贷分类与历史贷款严重逾期时间分析的另一条参考路线，未在本包中实施该数据适配。
- https://github.com/yanqiangmiffy/Loan-Default-Prediction ：原字段风格的参考；不能仅据字段相似确认原项目来源。
- https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset ：官方数据入口，注册获取。
- https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf ：字段、逾期和退出码定义。
- https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.AFTSurvivalRegression.html ：官方AFT模型与事件指示列语义。

GitHub参考仓库的代码许可在本次浏览中未逐一确认。本包以独立编写避免复制不明许可代码；未来如直接移植代码，应先读其许可证，保留作者归属、必要声明与修改记录。
