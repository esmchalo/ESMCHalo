# R5：Mixup＋KD与Focal＋KD分类头训练

## 当前状态

本包已完成本地语法、实际选择函数、校准与指标检查。当前本地没有PyTorch，因此不声称完成真实训练或PyTorch运行测试。启动器会先用服务器已验证的esmc_latest解释器执行7项CPU测试（损失、梯度、混合目标、原H0目标一致性、小型两分支训练与checkpoint恢复），通过才进入正式GPU训练。测试失败会直接打印错误，不启动训练。

## 实际任务

- 复用服务器现有ESMC canonical mean嵌入、10,567条开发记录、教师OOF监督、原分组折号和数据接口。
- 原H0的OOF作为固定参考，其哈希和既有H0复现报告必须匹配；不悄悄用K2替代H0，也不重新训练H0。
- 两个新候选各5折内层选轮+5折外层重训：共20次训练过程、10个最终分类头。
- 与原流程使用对应相同种子、标准化、优化器、学习率、batch size、80轮上限和10轮耐心值。Mixup额外随机量使用独立NumPy RNG，不消耗初始化与dropout的torch RNG；训练算法不同后数值路径仍不同，不宣称随机性被完全消除。
- 各候选及H0分别用开发OOF进行五折Platt交叉拟合，保留未校准及校准后表现；不使用旧Platt系数变换新模型。
- 根据预先固定的开发规则选择，只有选中一个新候选时才读取R3已有ESMC嵌入，完成五头logit均值、该候选全开发校准器、0.5阈值推理。
- 不重新编码、不重训教师、不下载模型、不改原文件、不进行新一轮效率测量。

## 两个候选

Mixup：每batch一个lambda~Beta(0.2,0.2)，批内配对；标准化表征、硬标签和温度2教师概率分别线性混合。0.5硬BCE+0.5×T²软BCE。混合的是表征，不代表新真实蛋白。

Focal：硬分支改为gamma=2的Focal，软分支仍是原温度2教师BCE；无类别权重、无静态1.5加权、不与Mixup叠加。Focal也改变硬/软项的有效量级，因此不能将候选差异完全归因于某一个单独机制。

## 开发选择规则（本项目本轮规则，不是文献或期刊门槛）

相对H0，新候选必须同时满足：
1. 原始与交叉拟合校准后的AP/AUROC下降均不超过0.001。
2. 校准后ACC、MCC不下降，Brier、log loss不升高。
3. 校准后log loss至少下降0.005，或MCC至少增加0.010。
4. 至少3/5折的校准后log loss或MCC改善；各折全部指标一并报告，不用该条件冒充稳健性显著检验。
5. strict25和clean-core的原始AP/AUROC下降均不超过0.001。

若两者都通过，依次按校准log loss低、校准MCC高、原AP高选择；完全相同时选Mixup。若均未通过，保留完整两候选结果、输出NO_CANDIDATE_SELECTED，不启动挑战。它只表示这两候选本轮未达到规则，不表示所有改进途径已穷尽。

挑战阶段只评选定的新候选，同时列出原冻结ESMCHalo、原DSP、R2校准DSP；两份数据与历史子集分别报告。先保存不含标签的预测，再关联原记录标签。R3原结果已被看过，必须称回顾性评价。此包先交付点估计、逐条输出和模型，不先运行耗时bootstrap；后续是否需要区间由具体结果决定。

## 最小代码变更

legacy_v205.py为原V2-05A代码副本，原样保留，不调用其main或旧候选选择流程；复用其输入绑定、Student、loader、optimizer、predict等接口。
training.py从原训练函数派生，只接入两种loss/batch变换、改候选checkpoint元数据、增加逐轮日志。run.py实现H0绑定、新候选循环、校准、固定选择与选定候选评价。calibration_core.py复用已有R2/R4中的Platt和指标函数，移除不用的EM代码。原历史协议与服务器文件均不修改。

## Mac上传

```bash
cd ~/Downloads
scp -P 2222 ESMCHalo_R5_student_training_v1.zip lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/
ssh -p 2222 lvfang@10.8.144.45
```

## 服务器首次启动（不依赖unzip）

```bash
(
set -e
cd /home/lvfang/ESMC_halophile
test ! -e R5_student_training_v1_output
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lvfang/miniconda3/envs/esmc_latest/bin/python -s -m zipfile -e ESMCHalo_R5_student_training_v1.zip .
nohup bash R5_student_training_v1/run.sh > R5_student_training_v1_console.log 2>&1 < /dev/null &
echo "R5 PID: $!"
)
```

不主动终止GPU上的其他任务。训练只涉及轻量分类头，但仍需可用GPU显存；不能根据编码时长推断训练时长。日志逐轮输出，可以观察真实进度，不预先承诺运行分钟数。

## 查看进度

```bash
tail -n 25 /home/lvfang/ESMC_halophile/R5_student_training_v1_console.log
```

CPU运行测试结束前，正式输出目录可能尚未出现；之后可查看：

```bash
cat /home/lvfang/ESMC_halophile/R5_student_training_v1_output/STATUS.json
```

每折显示START、内层每轮AP/loss、外层每轮完成、DONE。结束时显示DEVELOPMENT SELECTION和RESULTS。状态COMPLETED意味着实验完成，不意味着新方法胜出；实际选择看SELECTION.json。

如果测试阶段失败，主日志包含异常，训练不会开始。正式阶段失败时STATUS有traceback且结果包保留已完成折。已完成折的checkpoint/表/元数据通过哈希验证后可续跑，勿删除输出或反复新建任务。仅在确认进程已结束并处理具体异常后，用run.sh --resume恢复；不会由失败自动切换参数或方法。

## Mac取回

主日志显示RESULTS后：

```bash
cd ~/Downloads
scp -P 2222 lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/R5_student_training_v1_output_RESULTS.zip .
```

结果包包含全部新checkpoint、逐折训练历史、OOF概率、原始/交叉校准指标、开发子集结果、选择记录；选中时另有候选全开发Platt参数、五成员挑战logit、原记录恢复及比较表。无需补跑收集程序。

## 科学范围

这是冻结ESMC表征上的训练方式比较。旧教师—学生折间依赖、校准OOF到集成的分布差异、已知挑战结果带来的回顾性性质仍然存在。数据不支持的独立性与蒸馏因果结论不能因新训练而重命名。保持网络结构并不等于已重新验证新版本的具体耗时，R1数值属于原冻结实现。

参考：Mixup https://arxiv.org/abs/1710.09412；Mixup校准 https://arxiv.org/abs/1905.11001；Focal校准 https://arxiv.org/abs/2002.09437。候选参数为本轮固定实施选择，非文献证明的本任务最优参数。
