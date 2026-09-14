# R8：已有分类头的固定权重平均验证

## 交付和状态

本包不是新训练包：没有优化器步骤，不重新编码，不重训教师。只读取H0和R5 Mixup＋KD现成模型，在对应折将参数各取一半，得到仍为5个分类头的S0候选。服务器尚未运行。本地语法、选择规则、校准和指标测试已通过；7项torch CPU测试启动时自动执行，通过才进行模型预测。

本次本地分析见 ANALYSIS_CN.md 和 completed_local_analysis/。固定分数平均已实算有收益，但未达原替换门槛；S0权重平均还没有结果。两者绝不能混淆。

## 实现与理由

1. averaging.py：加载同折H0和Mixup权重，检查源文件哈希、折号、参数名称/形状、完全相同的标准化均值/尺度；仅计算0.5θ_H0+0.5θ_Mixup，不训练。
2. run.py：复用R7的数据绑定、校准、选择和挑战推理接口，只将训练步骤替换成权重合并及OOF预测；仅S0参与选择。
3. PROTOCOL.json：固定0.5比例和所有来源模型哈希，无比例网格。沿用原开发保护和最小增益门槛。
4. test_runtime.py：7项CPU测试检查参数平均、相同模型推理恒等、折号/标准化/参数键不匹配拒绝、非有限值拒绝以及标准化计算。
5. run.sh：沿用验证Python和环境隔离。失败打印明确错误，无静默跳过。
6. legacy_v205.py、calibration_core.py和挑战输入：复用原代码和数据。local_analysis.py用于复算已经完成的分数平均与挑战错误诊断，正常服务器启动不再次运行该分析。

## 服务器执行流程

- 读取原10,567条开发ESMC嵌入与绑定标签；不编码。
- 读取冻结H0的5个头和R5 Mixup的5个头，检查逐折兼容性。
- 每个来源头在自身外层留出折重做一次分类头推理，与既有OOF raw logit比较。最大绝对差需<=2e-5，这是数值复现容差，不是性能门槛；不一致报错，不自行放宽。
- 生成5个S0头并得到自身OOF预测；每折只预测对应留出记录。
- 对S0自身做交叉拟合Platt，输出整体、逐折、strict25/clean-core和错误转移结果。来源教师一致性分层为沿用诊断，不是S0筛选条件。
- 沿用原门槛：原始及校准AP/AUROC下降<=0.001；校准ACC/MCC不降、Brier/log loss不升；log loss下降>=0.005或MCC提升>=0.010；至少3折log loss或MCC改善；两个既有子集原始排序损失<=0.001。
- 只有S0通过才进行一次挑战预测；使用已有R3的1675条嵌入，先输出预测再恢复1680条标签记录，分别评价两数据集及历史同源子集。五头raw logit均值后用S0自己的完整开发Platt，阈值0.5。
- 无论选中与否，输出完整结果ZIP。未通过不会继续调平均比例。

原门槛没有因为已看见分数平均收益而改变。分数平均需要10个头，参数平均只需5个头；S0是否保留其收益未知。此流程不是把已观察到的分数平均结果“压缩成5头”的保证。

## 运行命令

需保留服务器原开发文件、冻结模型、R5_student_training_v1_output和R3_existing_challenge_v1_output/esmc。不需要再收集任何附件。

Mac上传：
```bash
scp -P 2222 ~/Downloads/ESMCHalo_R8_head_average_v1.zip lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/
```

服务器启动（不用unzip）：
```bash
(
set -e
cd /home/lvfang/ESMC_halophile
if [ -e R8_head_average_v1_output ]; then
  echo 'R8输出已存在，停止重复启动。'
  exit 1
fi
R8_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R8_PY" -s -m zipfile -e ESMCHalo_R8_head_average_v1.zip .
nohup bash R8_head_average_v1/run.sh > R8_head_average_v1_console.log 2>&1 < /dev/null &
echo "R8后台PID：$!"
)
```

查看：
```bash
tail -n 30 /home/lvfang/ESMC_halophile/R8_head_average_v1_console.log
```

完成后状态在：
```bash
cat /home/lvfang/ESMC_halophile/R8_head_average_v1_output/STATUS.json
```

状态COMPLETED后在Mac取回：
```bash
scp -P 2222 lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/R8_head_average_v1_output_RESULTS.zip ~/Downloads/
```

CPU测试失败在console报错，此时未进入正式主程序。主程序出错写FAILED及traceback并打包已有结果。不要重复启动或覆盖结果。若确定旧进程已结束并修复明确错误，可用run.sh --resume；这是重做已有头推理，不会触发训练。

## 如何解释结果

S0如果未通过，就没有合格新部署版本，不能按已知挑战表现继续调权重。S0若通过开发而挑战未改善，说明收益未迁移。S0若同时改善，则可以进一步复算挑战配对区间并考虑更新最终模型；也不能把回顾性评价改称独立盲测。

这项操作属于已有分类头的组合与部署优化，不新增预测任务或编码器。R1实测耗时仍属于原H0版本；S0结构保持五头不等于已经重新测得相同秒数。它不解决既有教师折间依赖，不构成论文录用保证。
