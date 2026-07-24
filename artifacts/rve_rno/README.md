# RVE-RNO模型制品

　　`c48_v2_liu_direct/`保存固定c48胞元、直接状态率演化的能量型RVE-RNO（循环神经算子）最终检查点及其架构、归一化、训练和性能元数据。`best_model.pt`必须与`artifact_metadata.json`配套加载。

　　最小推理命令为：

```bash
python examples/rve_rno_inference.py
```
