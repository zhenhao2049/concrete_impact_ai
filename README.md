# Concrete Impact AI

　　项目地址：[https://github.com/zhenhao2049/concrete_impact_ai](https://github.com/zhenhao2049/concrete_impact_ai)

　　本仓库提供混凝土冲击代理模型的精简佐证代码，主要包括J2率相关粘塑性有限元、周期代表性体积单元（RVE）与直接有限元平方（FE²）计算、能量型循环神经算子（RNO）的训练和材料点部署，以及速度Verlet积分神经校正器（INC）原型。本文件已合并原“说明文档.md”的全部内容，是仓库唯一的项目说明。层合板、Slurm集群生产、阶段0/1交接、历史报告和独立技术文档不属于本仓库。

## 环境

```bash
mamba env create -f environment.yml
conda activate surrogate-model
python -m pip install -e .
python -m concrete_impact.cli.inspect_env
```

## FEM调用

运行J2粘塑性材料点算例：

```bash
concrete-impact-run \
  --config configs/benchmarks/plasticity/plastic_j2_viscoplastic_material_point.yaml
```

生成2000条固定胞元RVE任务清单：

```bash
concrete-impact-generate-rve-data \
  --config configs/experiments/rve_rno_c48_v2_2000.yaml \
  --manifest-only
```

## RNO

使用仓库内置模型执行一次材料点推理：

```bash
python examples/rve_rno_inference.py
```

完整数据就绪后启动RNO训练：

```bash
concrete-impact-train-rve-rno \
  --config configs/nn/rve_rno_c48_v2_k6_gpu_liu_direct.yaml
```

　　`data/rve_rno_c48_v2_2000/`保存2000条正式路径的任务、验收、归一化和划分元数据，并附带一条HDF5响应样例。完整2000条HDF5响应未提交至Git；最终模型及其元数据位于`artifacts/rve_rno/c48_v2_liu_direct/`。

## INC原型

```bash
concrete-impact-generate-inc-data \
  --config configs/nn/velocity_verlet_inc_linear_rod_data.yaml

concrete-impact-train-inc \
  --config configs/nn/velocity_verlet_inc_linear_rod_training.yaml
```

## 主要目录

```text
artifacts/                 最终RVE-RNO模型及元数据
configs/                   FEM、RVE、RNO和INC配置
data/                      数据集元数据与小型样例
examples/                  最小推理示例
src/fem/                   通用有限元底层
src/concrete_impact/       材料、RVE、RNO、INC和PF-CZM代码
tests/                     保留模块的自动化测试
```
