# 实验配置

　　本目录保存RVE路径设计、固定胞元数据生成和执行性能配置。2000条正式数据使用：

```text
configs/experiments/rve_rno_c48_v2_2000.yaml
```

　　该配置定义100个设计分层、每层20条路径以及1400/300/300生产任务预分配。正式训练验收另按1600/200/200划分，不修改原始任务身份和HDF5响应。

　　生成任务清单：

```bash
concrete-impact-generate-rve-data \
  --config configs/experiments/rve_rno_c48_v2_2000.yaml \
  --manifest-only
```

　　旧版小规模配置仅用于代码接口和回归检查，不应替代正式2000路径配置。
