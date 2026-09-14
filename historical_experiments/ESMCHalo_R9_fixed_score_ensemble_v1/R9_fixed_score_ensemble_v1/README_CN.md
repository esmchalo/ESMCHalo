# R9 固定分数平均：探索性挑战评价

## 本轮定位与明确变更

R8的五头参数平均S0没有选中。本包不修改或部署原H0，不更改原筛选门槛。只完成已在开发OOF看到跨折收益的固定分数平均E0的探索性挑战评价。

E0使用H0五头和R5 Mixup＋KD五头，共10个分类头，共享单ESMC编码器。它不是原五头系统；若最终必须严格保留五头，E0只能是分析对照，不能直接作为最终系统。原R1效率数据不能套到E0。本包不重复效率测试。

E0原开发最低增益门槛没有通过，因此本轮是明确新增的回顾性探索，不是原R5–R8预设方案通过后的确认性验证，也不自动选中模型。不能用本轮较好的挑战结果反向声称它通过了原开发筛选。

## 固定操作

1. 读取已经验证的R5 OOF分数，以0.5*z_H0+0.5*z_Mixup构成开发分数，在完整开发OOF上拟合E0自己的部署Platt。该拟合不是交叉拟合性能评价；开发交叉拟合诊断已完成。
2. 读取R3现成1675条ESMC嵌入，加载已登记的10个现成分类头，各自使用自身标准化参数。只推理，不调用优化器或训练。
3. 先算H0五头logit均值、Mixup五头logit均值，再各取0.5，最后应用E0 Platt。等价于10头raw logit等权均值后Platt；不是概率均值，也不是各校准概率均值。
4. 保存唯一序列预测之后才合并原1680条记录标签。分别报告newtest1402和Zhang278及历史strict40/25子集。
5. 并列E0原始/E0校准、H0原冻结、DSP原始及R2校准DSP。输出各指标差值、纠正/新增错误数和逐条结果。本包没有新增bootstrap；若有实际收益，可直接在输出上补配对区间，无需再跑模型。
6. 原H0预测与R3记录逐条比较；差异超过2e-6报错。该数值是浮点复现容差，不是性能门槛。
7. 输出STATUS明确exploratory=true、automatic_replacement=false、production_model=H0、heads=10。没有调权/调阈值循环，没有根据挑战标签选择另一个版本。

## 文件与验证

run.py：读取现有源模型和嵌入、固定分数聚合、来源复现、原记录恢复及评价。
averaging.py：沿用R8已在服务器测试通过的score和标准化接口；本包不调用其中参数平均函数。
legacy_v205.py、calibration_core.py、inputs：复用。
PROTOCOL.json：锁定现成模型、OOF哈希以及10头探索性质。
LOCAL_CHECKS.json：本地语法与实际原记录恢复函数测试、缺失/重复拒绝、等权聚合恒等式通过；本地无torch，未声称本轮GPU执行通过。

## 上传与启动

保留服务器的冻结H0模型、R5_student_training_v1_output和R3_existing_challenge_v1_output/esmc。

Mac：
```bash
scp -P 2222 ~/Downloads/ESMCHalo_R9_fixed_score_ensemble_v1.zip lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/
```

服务器：
```bash
(
set -e
cd /home/lvfang/ESMC_halophile
if [ -e R9_fixed_score_ensemble_v1_output ]; then
  echo 'R9输出已存在，停止重复启动。'
  exit 1
fi
R9_PY=/home/lvfang/miniconda3/envs/esmc_latest/bin/python
env -u PYTHONPATH PYTHONNOUSERSITE=1 "$R9_PY" -s -m zipfile -e ESMCHalo_R9_fixed_score_ensemble_v1.zip .
nohup bash R9_fixed_score_ensemble_v1/run.sh > R9_fixed_score_ensemble_v1_console.log 2>&1 < /dev/null &
echo "R9后台PID：$!"
)
```

查看进度：
```bash
tail -n 30 /home/lvfang/ESMC_halophile/R9_fixed_score_ensemble_v1_console.log
```

查看状态：
```bash
cat /home/lvfang/ESMC_halophile/R9_fixed_score_ensemble_v1_output/STATUS.json
```

COMPLETED后在Mac取回：
```bash
scp -P 2222 lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/R9_fixed_score_ensemble_v1_output_RESULTS.zip ~/Downloads/
```

## 后续判断

若两份挑战都保持收益，才进一步计算配对区间和评价十头成本是否值得，并明确这是不同部署配置；不自动覆盖原系统。
若一份改善、一份退化，完整报告范围，不按各挑战分别选权重。
若未迁移，说明这项既有模型互补没有解决挑战表现，不再追加相邻混合比例。

本包已生成但尚未在服务器执行。没有承诺超过DSP；它完成的是一个具体、固定、无需训练的迁移检验。
