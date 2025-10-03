# MicroMind


### 🚀 Introduction

### ✨ Features

### 🧪 Experiment 
#### 下载数据集
采用原项目提供的数据集
```
modelscope download --dataset gongjy/minimind_dataset
```

#### 进行预训练

采用默认配置运行预训练
```
python train_pretrain.py
```
会加载pretrain_hq.jsonl中的数据进行预训练

在batch_size为32的情况下，训练用时50min
在batch_size为64的情况下，训练用时会减慢到

#### 进行sft监督微调（学对话方式）

调整batch_size为128，对应的学习率也扩大八倍

参数更新的公式可简化为
```
新参数 = 旧参数 - 学习率 × 梯度
```
大batch_size更新的轮次少，扩大学习率防止收敛过慢

运行代码
```
python train_full_sft.py
```
一个epoch 45min，训练了一个epoch后停止了

### 📊 Results

