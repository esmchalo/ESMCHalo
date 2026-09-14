# R7：G0训练标签一致性门控KD

状态：执行包已生成；本地语法、选择规则、校准/指标及诊断函数测试通过。当前环境无torch，7项CPU运行测试由服务器启动脚本自动执行，通过才训练。服务器训练尚未运行。

## 与论文主线的关系

原目标仍为：多表征教师辅助训练、冻结ESMC单编码器部署，用于嗜盐生物相关来源蛋白预测。R5/R6是训练消融和方案排除，放补充材料即可，不将每轮试验拆成新的正文贡献。G0只检验如何使用已有教师监督，不新增任务、编码器、蛋白数据或部署模块。

R6支持保留教师的完整训练方案，但不能完全隔离教师—学生折间依赖。对教师预测与标签不一致的825条，H0比B0多错23条，仅构成门控假设依据，不能视作G0可以追回的确定收益，也不能证明教师错误传播。来源标签不是无噪声的盐耐受真值。

## 固定训练定义

只新增G0。对当前训练批次：m_i=1[(u_i>=0)==(y_i>=0.5)]。

loss=mean(0.5*BCEWithLogits(z,y)+2*m*BCEWithLogits(z/2,sigmoid(u/2)))。

- 教师一致：与H0的T=2、硬/软各0.5一致。
- 教师冲突：软项关闭，硬项仍0.5；按整个批次平均，不对门控记录重新归一化。
- 训练门控只接收当前内层/外层训练分区的标签。留出标签只用于选轮或评价，不参与训练门控。
- 不叠加Mixup、Focal，不改种子、划分、优化器、标准化或网络结构。
- 一个候选，5次内层选轮训练+5个外层最终头；不是只训练5次。
- 门控降低部分记录的有效总损失，结果不能纯归因于删除错误知识。

## 文件及最小改动

training.py：从R6复用训练循环，只替换损失并新增门控计数。
run.py：复用输入绑定、H0参考、校准、筛选与挑战接口；仅训练G0，新增一致/冲突组及错误转移统计。
PROTOCOL.json：锁定上述方案，沿用R5/R6筛选门槛。
test_runtime.py：7项CPU测试，含全一致H0损失/梯度相等、全冲突0.5BCE、混合批次平均、零logit边界、非法输入、有限梯度、实际训练分区检查及微型训练/权重恢复。
run.sh：沿用验证解释器和环境隔离，运行测试后启动训练。
legacy_v205.py、calibration_core.py、inputs：原接口及参考数据直接复用。

## 开发筛选及挑战

G0原始和交叉拟合Platt的AP/AUROC相对H0下降不超过0.001；校准后ACC/MCC不降，Brier/log loss不升；log loss至少降0.005或MCC至少升0.010；至少3/5折log loss或MCC之一改善；strict25和clean-core原始AP/AUROC下降不超过0.001。OR折数不是“分类改善折数”，必须阅读完整逐折表。

未改门槛，不因看过G0结果降低标准。每个版本单独交叉拟合Platt。排序同时检查逐折与合并结果，避免将跨折尺度变化误认为一致排序收益。

选中G0后：全开发OOF拟合部署Platt，五个分类头各自标准化后输出raw logit，取均值再Platt。复用R3的1675条ESMC嵌入，先保存未读取挑战标签的预测，再恢复1680条原记录。分别报告newtest1402、Zhang278及历史strict40/25子集，并列H0冻结系统、DSP原始概率、R2校准DSP。

挑战已有研究暴露，仍为回顾性评价；开发通过不保证真实挑战改善；不能把历史同源标记称为新训练边界的重新比对。此包输出点估计，未加入新的bootstrap或效率循环。真实挑战若有收益，再由现有逐条结果计算配对区间即可，无需重跑模型。

## 运行

保留原开发文件和R3_existing_challenge_v1_output/esmc缓存。无需上传R5/R6结果供本包使用，H0从原验证目录读取并绑定既有哈希。

Mac：
```bash
scp -P 2222 ~/Downloads/ESMCHalo_R7_gated_kd_v1.zip lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/
```

服务器：
```bash
(
set -e
cd /home/lvfang/ESMC_halophile
if [ -e R7_gated_kd_v1_output ]; then
  echo 'R7输出已存在，停止重复启动。'
  exit 1
fi
R7_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R7_PY" -s -m zipfile -e ESMCHalo_R7_gated_kd_v1.zip .
nohup bash R7_gated_kd_v1/run.sh > R7_gated_kd_v1_console.log 2>&1 < /dev/null &
echo "R7后台PID：$!"
)
```

进度（每个训练epoch输出）：
```bash
tail -n 30 /home/lvfang/ESMC_halophile/R7_gated_kd_v1_console.log
```

结束后检查：
```bash
cat /home/lvfang/ESMC_halophile/R7_gated_kd_v1_output/STATUS.json
```

STATUS=COMPLETED后Mac取回：
```bash
scp -P 2222 lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/R7_gated_kd_v1_output_RESULTS.zip ~/Downloads/
```

若CPU测试失败，会在console打印完整错误并停止，此时尚未正式训练；主程序失败会写FAILED及traceback并打包现有结果。不要反复重启。确认原进程已结束且排除具体原因后，才可用run.sh --resume续跑；完整已完成结果拒绝重复执行。

## 结果如何决定下一步

1. 开发门槛未通过：保留H0，查看是否改善冲突组而损害一致组、是否仅概率改变，不能自动启动新候选。
2. 开发通过且挑战收益一致：作为最终训练方案的候选更新，并用现有逐条结果核查配对区间及收益范围。
3. 开发通过但挑战收益不迁移：承认内部优化未解决真实分布差异，不按挑战标签再次改门控。

R7的完成终点是对这一假设作出结论，不是保证期刊录用或保证所有指标超过DSP。模型改进是否继续，需要新的可检验解释；正文始终围绕单编码器部署的判别能力与计算成本，不把失败尝试数量当贡献。
