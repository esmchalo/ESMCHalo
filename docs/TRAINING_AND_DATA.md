# 训练与数据复现说明

## 三条流程

1. 冻结预测：FASTA → 固定ESMC-600M layer36、重叠窗口池化 → 五个H0头各自StandardScaler → 平均raw logit → 固定Platt → 阈值0.5。
2. 结果复算：已有预测 → 指标、配对统计。通过scripts/reproduce_results.py执行，无需重训。
3. 历史训练：固定开发数据/同源组/外层五折 → Ankh与ESM-2缓存 → 教师MinMaxScaler、PCA与嵌套选择 → 教师OOF → 学生K0—K5 → H0—H3 → 校准与冻结。训练不在自动验收中执行。

## 配置的权威来源

- 教师：`historical_project/v2_01d_execution_package/ESMCHalo_v2_V2_01D_nested_teacher_v1/`。100轮上限、patience25的完整流程与历史RUN_SUMMARY交叉对照；展开其余源码默认参数的命令见下面的打印入口。这是等价参数展开，不伪装成逐字恢复的原shell历史。
- K0—K5：`historical_project/v2_04_execution_package_v1_1_fix/ESMCHalo_v2_V2_04_DeepSaltPro_OOF_KD_v1/V2_04_PROTOCOL.json`，保留全部候选与综合门槛，不重选“最优”。
- H0—H3：`historical_project/v2_05a_execution_package/ESMCHalo_v2_V2_05A_hard_sample_weighting_ablation_v1/V2_05A_PROTOCOL.json`。
- 校准/阈值：`historical_project/v2_06_execution_package/ESMCHalo_v2_V2_06_ensemble_calibration_freeze_v1/V2_06_PROTOCOL.json`。
- 已执行轮数、种子、绑定哈希与选择结果：`historical_project/ESMCHalo_v2_optimization/`各阶段原始JSON/TSV。

## 数据依赖

开发折：`ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/group_folds_v2.tsv`。
训练标签/样本注册：`ESMCHalo_v2_optimization/V2_02_label_audit/V2_02B_audit_lock_v1/V2_02_LOCKED_TRAINING_REGISTRY.tsv`。
教师OOF：`ESMCHalo_v2_optimization/V2_01_baseline_and_teacher_oof/deepsaltpro_nested_oof_full_v1/deepsaltpro_oof_predictions.tsv`。
ESMC缓存：`data/embeddings/esmc600m_canonical_b1/rows.tsv`与`embeddings.npy`。
教师缓存：`work/deepsaltpro_fair_v1/features/full_ankh_window/`与`full_esm2_window/`。
其他依赖以验收产生的`training_dependency_inventory.json`和`historical_absolute_paths.json`为准；该清单属于已识别依赖，不宣称穷尽动态加载依赖。

大型矩阵未随原文本采集附件提供。服务器验收会只读检查明确位置，不会将找到原缓存误写成“包内已经能重建”。从原始数据重新生成特征尚需整理原始序列获取步骤和编码器许可/版本；仅发布行号、序列哈希不能代替可获取的数据。若已上传表格缺少原`v2_development.tsv`，作者已提供该原件；本候选不再收集服务器文件。含原始序列的数据不在公开候选中再分发，需依据来源条件获取。

## 打印历史训练启动命令（现在无需运行训练）

```bash
python scripts/train_stage.py teacher --root /home/lvfang/ESMC_halophile --output /home/lvfang/ESMC_halophile/repro_teacher_new
python scripts/train_stage.py kd --root /home/lvfang/ESMC_halophile --output /home/lvfang/ESMC_halophile/repro_kd_new
python scripts/train_stage.py weighting --root /home/lvfang/ESMC_halophile --output /home/lvfang/ESMC_halophile/repro_weighting_new
python scripts/train_stage.py calibration --root /home/lvfang/ESMC_halophile --output /home/lvfang/ESMC_halophile/repro_calibration_new
```

默认仅打印。只有显式添加`--execute`才执行；拒绝覆盖已有输出。各原程序仍有自己的哈希和数据角色检查，本包没有绕过。注意：后续阶段读取历史固定阶段目录；以上四条命令不是自动串联新输出的端到端重训流水线，也不能据此宣称新模型已产生。新的训练输出与下游绑定文件需经明确整理和核对才能连接。

## 环境记录

公开包的requirements/environment文件原样保留作来源记录，不能等同于已经在新环境成功安装。自动验收使用现有解释器，保存本次版本及硬件。缺少直接绑定教师V2-01D运行的软件快照时如实注明。验收GPU、CPU、内存不能反向证明R1当时配置。

## 验证声明模板的边界

验收通过后可以逐项写“指定版本冻结模型在记录环境通过参考预测检查”“主要结果由公开逐条预测复算”。未执行完整重训之前不写“全流程重训通过”；未做干净环境安装之前不写“已验证一键安装”。公开发行前还需统一README、CITATION、许可范围、数据来源和版本标识。

本候选没有自动服务器收集功能。先前验收日志在本文件中的指代仅为历史背景。实际公开的数据为可见的划分/标签注册和预测记录；原始序列和完整表征缓存不随本候选再分发。完整数据获取重建和第三方训练依赖仍未验收。
