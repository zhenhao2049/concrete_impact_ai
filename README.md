# Concrete Impact AI

　　项目地址：[https://github.com/zhenhao2049/concrete_impact_ai](https://github.com/zhenhao2049/concrete_impact_ai)

　　本仓库提供混凝土冲击代理模型的精简佐证代码，主要包括J2率相关粘塑性有限元、周期代表性体积单元（RVE）与直接有限元平方（FE²）计算、能量型循环神经算子（RNO）的训练和材料点部署，以及速度Verlet积分神经校正器（INC）的原型和二维线弹性多模态实现。本文件已合并原“说明文档.md”的全部内容，是仓库唯一的项目说明。层合板、Slurm作业调度系统的集群启动脚本、阶段0/1交接、历史报告和独立技术文档不属于本仓库。

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

## INC

### 速度Verlet一维杆原型

```bash
concrete-impact-generate-inc-data \
  --config configs/nn/velocity_verlet_inc_linear_rod_data.yaml

concrete-impact-train-inc \
  --config configs/nn/velocity_verlet_inc_linear_rod_training.yaml
```

### 二维线弹性多模态INC

　　本次新增的数据生成程序可以构造固定网格下的粗、细时间步动力学轨迹，并计算速度Verlet积分在步初和步末的广义力残差标签。以下命令生成144条正式轨迹的任务清单和计算分组：

```bash
concrete-impact-generate-elastic-inc-data prepare \
  --config configs/experiments/inc_elastic_p1_official_144.yaml \
  --bundle-count 9 \
  --parallel-paths 16
```

　　正式轨迹的并行计算必须在Slurm作业中执行。本仓库提供与集群无关的任务准备、计算、状态统计和结果归并接口，不包含指定账号、目录和资源配置的Slurm启动脚本。

　　完整数据准备好以后，依次执行运行前检查、训练、响应模型选择、独立测试和计算效率测试：

```bash
concrete-impact-run-elastic-inc preflight \
  --config configs/nn/inc_elastic_multimode_half_sine_spatial_v3.yaml
concrete-impact-run-elastic-inc run \
  --config configs/nn/inc_elastic_multimode_half_sine_spatial_v3.yaml
concrete-impact-run-elastic-inc select-response \
  --config configs/nn/inc_elastic_multimode_half_sine_spatial_v3.yaml
concrete-impact-run-elastic-inc evaluate \
  --config configs/nn/inc_elastic_multimode_half_sine_spatial_v3.yaml
concrete-impact-run-elastic-inc benchmark \
  --config configs/nn/inc_elastic_multimode_half_sine_spatial_v3.yaml
```

　　本仓库不提供144条有限元轨迹、训练权重和运行结果。上述训练、独立测试和测速命令要求将已验收数据放在最终配置所指的`results/dynamics/inc_training_elastic/official/`目录。不使用正式数据时，可运行下列单元测试，检查数据规划、边界载荷积分、特征构造、网络输出和判定条件：

```bash
python -m pytest -q \
  tests/unit/concrete_impact/test_inc_elastic_data.py \
  tests/unit/concrete_impact/test_inc_elastic_multimode.py
```

　　代码阅读顺序建议为：最终训练配置、数据规划与生成模块、数据集与网络模块、训练模块、验证与测速模块，最后结合命令行入口和单元测试核对整体调用关系。当前训练配置仅对应固定网格、线弹性系统和半正弦脉冲载荷，不能用于支持跨网格、跨材料或一般非线性冲击问题的结论。

## 主要目录

```text
artifacts/                 最终RVE-RNO模型及元数据
configs/                   FEM、RVE、RNO和INC配置
data/                      数据集元数据与小型样例
examples/                  最小推理示例
src/fem/                   通用有限元底层
src/concrete_impact/       材料、RVE、RNO、INC数据生成与训练代码
tests/                     保留模块的自动化测试
```
