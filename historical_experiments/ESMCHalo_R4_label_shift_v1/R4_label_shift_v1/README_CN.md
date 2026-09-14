# R4：一次性类别比例适配验证

本包是可执行方案，真实H0开发验证尚未在本地运行。服务器读取既有H0与DeepSaltPro OOF文件；不以旧K2代替H0。输入文件路径及预期哈希由已归档V2-06/V2-05A记录绑定。缺失或不匹配时明确报错，不重新收集序列。

## 执行范围

1. 读取两系统10,567条同记录OOF数据，核对ID/标签/折号/同源组。
2. 每个留出折抽取三个500条模拟批次：阳性150/250/350条，其余为阴性；每个批次内无放回，跨条件可能复用记录。批次只在同一个留出折内部构造，原分组折保持。两系统使用相同记录。
3. 每折仅用其余折拟合既有形式的Platt。这是为避免直接在本轮验证折上拟合校准器，不是再次比较C0/C1/C2。既有OOF预测的训练依赖仍存在，不称完整嵌套独立实验。
4. 来源先验使用来源校准概率均值。EM适配函数仅接收目标分数与来源先验，不接收目标标签。
5. 两系统各15个条件，分别计算修正后减修正前的四指标均值。ACC>0、log loss<0、MCC>=0、Brier<=0才通过；两系统均通过后才运行挑战，避免只适配ESMCHalo。任何先验估计触及边界时停止继续应用。该工程筛选规则不是显著性检验。
6. 挑战预测使用已固定的ESMCHalo Platt概率及DeepSaltPro R2 Platt概率，分别进行同一种适配。DSP基线明确为R2校准版，不冒充原未校准DSP。与原冻结R3的最终比较需同时保留原表。
7. 两份挑战独立估计；先保存不含标签的预测，再读标签计算指标。挑战结果已知，因此仍属于回顾性探索，不是首次独立盲测。

不得从开发模拟通过推断真实挑战只有类别比例变化。OOF单成员与部署五成员分数差异仍存在。适配是批次相关扩展，不修改原冻结模型及论文核心版本。只进行这一轮，不根据失败结果再增候选。

## 最小文件变更

只新增本目录：core.py实现既有Platt形式、EM与指标；run.py负责输入绑定、15条件验证、通过后挑战；PROTOCOL.json固定所有规则；inputs/保存已收到的R3分数和单独标签；test_core.py验证数学与错误分支。项目原文件不修改。

## Mac上传

```bash
cd ~/Downloads
scp -P 2222 ESMCHalo_R4_label_shift_v1.zip lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/
ssh -p 2222 lvfang@10.8.144.45
```

## 服务器解压和启动（无需unzip）

```bash
(
set -e
cd /home/lvfang/ESMC_halophile
test ! -e R4_label_shift_v1_output
env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/lvfang/miniconda3/envs/esmc_latest/bin/python -s -m zipfile -e ESMCHalo_R4_label_shift_v1.zip .
nohup bash R4_label_shift_v1/run.sh > R4_label_shift_v1_console.log 2>&1 < /dev/null &
echo "R4 PID: $!"
)
```

每个折×比例×系统完成后都会打印一行DEV进度，共30行。无需GPU，不跑bootstrap。出现STOP开发未通过是有效研究结果，不是执行故障，不应重跑争取通过。

```bash
tail -n 20 /home/lvfang/ESMC_halophile/R4_label_shift_v1_console.log
cat /home/lvfang/ESMC_halophile/R4_label_shift_v1_output/STATUS.json
```

状态说明：
- COMPLETED + STOP_DEVELOPMENT_GATE_NOT_PASSED：验证完成但不支持继续适配，保留原系统。
- COMPLETED + DEVELOPMENT_PASS_CHALLENGE_EVALUATED：已完成挑战的回顾性适配评价，仍须检查真实得失。
- FAILED：运行/输入错误，具体异常在STATUS与日志，不代表方法评价不通过。

## Mac取回

主日志打印RESULTS后：

```bash
cd ~/Downloads
scp -P 2222 lvfang@10.8.144.45:/home/lvfang/ESMC_halophile/R4_label_shift_v1_output_RESULTS.zip .
```

结果包含实际开发输入、模拟成员、逐条概率、每条件指标、各折校准参数、GATE.json、STATUS.json；只有通过时才包含挑战预测和指标。返回同一个ZIP即可解读。不需要另一轮收集包。

## 交付验证

本地仅用合成数据测试已知先验偏移恢复、不偏移恒等、顺序不变、无效输入/不收敛报错、Platt拟合已知概率和门槛逻辑。不把合成测试当成ESMCHalo真实验证。Python与shell语法检查另行完成，结果见TEST_RESULTS.json。

文献依据：https://proceedings.mlr.press/v119/alexandari20a.html 。本包采用二分类标量Platt与EM先验更新，不声称复现该文全部实验或全部校准变体。
