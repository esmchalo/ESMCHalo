# R6：补齐监督方式与Mixup的四组对照

## 本次新增什么

只训练B0（硬标签BCE）与M0（Mixup硬监督）两个分支。原H0与R5 Mixup＋KD直接复用服务器R5输出，后者在报告中命名Mixup_KD；不重训这两组、不使用旧不同种子的K0冒充新对照。

每个新分支5折内层AP选轮和5折外层重训，共20次训练过程、10个最终分类头。ESMC嵌入、原分组折、标签、初始化种子、优化器、训练上限和早停规则沿用R5。B0/M0损失均不使用教师数值，但复用原数据加载接口仍会读取教师OOF表用于旧输入绑定；这不等于使用它训练。启动测试会核对改变教师数值不改变两分支损失或梯度。

不重编码、不重训教师、不扩展蛋白数据、不改变原模型和R5文件。

## 实施文件与理由

- legacy_v205.py：原输入、模型、优化器等接口，原样复用。
- training.py：沿用R5训练流程，仅将新分支改成B0/M0；元数据明确teacher_supervision_used=false。
- run.py：绑定实际R5的OOF和协议哈希；复用H0/Mixup_KD；训练新分支；计算各自校准指标、交互差值、固定开发选择及选定模型的挑战评价。
- PROTOCOL.json：运行前固定参数与原R5通过规则；不因R5只净增加3条正确预测而降低门槛。
- test_runtime.py：7项服务器CPU测试，包括教师无关性、M0混合目标、梯度、小型训练与checkpoint恢复。
- inputs/：已有R3输入与比较器结果；只在选定新方案后进行挑战预测和评价。

原R5目录与结果必须保留。程序直接读取/home/lvfang/ESMC_halophile/R5_student_training_v1_output/oof_predictions.tsv与PROTOCOL.json，并校验本次收到的真实R5结果对应哈希。

## 关键输出如何解释

supervision_contrasts.tsv和fold_supervision_contrasts.tsv分别报告整体、逐折：
1. M0-B0：没有教师时，Mixup的变化。
2. Mixup_KD-H0：有教师时，Mixup的变化。
3. (M0-B0)-(Mixup_KD-H0)：两种监督方案下Mixup收益的差别。
4. H0-B0：普通训练下加入教师的变化。
5. Mixup_KD-M0：Mixup训练下加入教师的变化。

ACC/MCC/AP/AUROC正差值通常较好；Brier/log loss负差值较好。指标为非线性统计量，差上差只作描述；原教师依赖、选轮差异、损失整体量级与历史H0复用等因素限制因果解释。不能仅凭表中一个差值就证明教师限制性能。

## 筛选与挑战

维持R5的开发替换规则：相对H0，原始与交叉校准AP/AUROC下降均不超过0.001；校准ACC/MCC不下降、Brier/log loss不升高；log loss降低至少0.005或MCC增加至少0.010；至少3折log loss或MCC改善；strict25/clean-core原始AP/AUROC下降不超过0.001。两候选都通过时按log loss、MCC、AP排序，完全相同优先B0。

这个规则是本项目本轮实施约定，不是期刊门槛。所有方案指标和对照差值都保存，即使没有候选通过也能分析原因。选择仅基于开发数据，只有选中B0/M0之一时才复用R3缓存作一次挑战评价；不读取挑战结果来选方案。选中模型的Platt用其自身开发OOF拟合，不套旧系数。原H0、DSP原始及R2校准版都留在挑战比较表。

OOF目标/校准与部署集成分布差异、已知挑战结果的回顾性性质仍保留。无教师方案若更好，论文应按结果调整，不能为了固定蒸馏主线排除它。

## 已完成与未完成

本地已通过Python/shell语法、实际选择函数、对照差值代数、已知Platt和指标检查。本地没有PyTorch，未声称完成真实训练；服务器先执行7项CPU测试，通过后启动GPU训练。测试失败会在主日志直接打印异常，正式训练不会开始。

## Mac上传

```bash
cd ~/Downloads
scp -P 2222 ESMCHalo_R6_supervision_controls_v1.zip lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/
ssh -p 2222 lvfang@10.8.144.45
```

## 服务器启动（使用Python解压）

```bash
(
set -e
cd /home/lvfang/ESMC_halophile
test ! -e R6_supervision_controls_v1_output
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lvfang/miniconda3/envs/esmc_latest/bin/python -s -m zipfile -e ESMCHalo_R6_supervision_controls_v1.zip .
nohup bash R6_supervision_controls_v1/run.sh > R6_supervision_controls_v1_console.log 2>&1 < /dev/null &
echo "R6 PID: $!"
)
```

## 查看进度

```bash
tail -n 25 /home/lvfang/ESMC_halophile/R6_supervision_controls_v1_console.log
```

测试通过后主日志显示RUNTIME TESTS PASSED；各折START/DONE，内层逐轮AP/loss、外层逐轮完成均会打印。正式运行后另可读：

```bash
cat /home/lvfang/ESMC_halophile/R6_supervision_controls_v1_output/STATUS.json
```

不承诺具体运行分钟数，实际取决于选轮和服务器负载。不要重复启动或终止其他GPU任务。故障后已完成折保留，只有确认进程结束并处理错误后才使用run.sh --resume；同一输出有进程锁，不能并行重复训练。

## Mac取回

主日志出现RESULTS后：

```bash
cd ~/Downloads
scp -P 2222 lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/R6_supervision_controls_v1_output_RESULTS.zip .
```

结果包包含新checkpoint、选轮历史、四组OOF和指标、整体与逐折监督对照差值、选择记录；选中时还含原记录挑战预测、五成员logit和比较表。NO_CANDIDATE_SELECTED是本轮筛选结果，非程序错误，也非所有改进空间已经穷尽的结论。FAILED才是需修复的运行错误。此包不运行耗时bootstrap，先交付训练与对照证据。
