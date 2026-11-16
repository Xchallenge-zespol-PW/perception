# Documentation

Engineering docs for **g1_perception**. Start with the [root README](../README.md) for the project overview; use this index for deep dives.

| Doc | Contents |
|-----|----------|
| [architecture.md](architecture.md) | Pipeline, package layout, Python modules, unit-test map |
| [ros_nodes.md](ros_nodes.md) | `perception_node`, exercise 1 TF/colors contract, `apriltag_node` |
| [config.md](config.md) | `params_*.yaml`, ROI / HSV / labels YAML |
| [workflow.md](workflow.md) | Offline bag → frames → train → infer |
| [tools.md](tools.md) | `tools/` command reference |
| [environment.md](environment.md) | Devcontainer, colcon, NumPy pins, tests, gitignored artifacts |
| [robot.md](robot.md) | Jetson Orin NX deploy (TensorRT + ONNX fallback) |
| [training.md](training.md) | Dataset / train / export / deploy model flow |
| [media/](media/README.md) | Where to drop demo clips and training plots |
