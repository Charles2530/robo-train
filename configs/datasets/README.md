# Dataset Configs

Dataset configs describe embodied dataset families and their supported robot
embodiments independently from any training framework. Framework task profiles
should include one of these files through `defaults` and then override the
specific task path, prompt, save name, and training parameters.

Preferred layout:

```text
configs/datasets/<dataset_family>/<robot_embodiment>.yaml
```

Examples:

```text
configs/datasets/robochallenge_table30v2/aloha_dual_arm.yaml
configs/datasets/robochallenge_table30v2/arx5_single_arm.yaml
configs/datasets/xiaoyu_task_a/agilex_dual_arm.yaml
configs/datasets/libero/generic.yaml
```

Each dataset config should declare:

- `dataset_id`: stable dataset + robot embodiment id.
- `dataset_type`: dataset family, such as `RoboChallenge_Table30v2`, `xiaoyu_Task_A`, or `LIBERO`.
- `dataset_path`: dataset root path; framework task configs may override it to a specific task directory.
- `robot_family`: robot embodiment, such as `aloha_dual_arm`.
- `supported_frameworks`: frameworks that can consume the dataset without extra unsupported conversion.
- `preprocessors`: named preprocessing steps that were already applied or are required.

Unsupported dataset/framework combinations should fail at profile load time.
