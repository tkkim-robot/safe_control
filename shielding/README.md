# Backup-Based Safety Filters

This directory contains the backup-based safety filters used in `safe_control`. The examples in this repository focus on three methods:

- [`gatekeeper`](https://github.com/tkkim-robot/safe_control/blob/main/shielding/gatekeeper.py): a safety filter that searches backward over the nominal horizon and commits to a nominal-plus-backup trajectory when needed. A parallel implementation is available in [`safe_control_jax`](https://github.com/tkkim-robot/safe_control_jax).
- [`mps`](https://github.com/tkkim-robot/safe_control/blob/main/shielding/mps.py): Model Predictive Shielding, which uses a one-step nominal horizon before switching to the backup trajectory.
- [`Backup CBF`](https://github.com/tkkim-robot/safe_control/blob/main/position_control/backup_cbf_qp.py): a backup-policy-based CBF-QP. Its implementation lives in `position_control`, but it is included in the same case studies for comparison.

For more details of the algorithms, please read our review paper.

These methods are demonstrated with two case studies:

- `examples/drift_car/test_drift.py`
- `examples/evade/test_evade.py`

## How to Run Case Studies

You can run the highway driving examples by:

```bash
# High-friction case with two moving obstacles
uv run python examples/drift_car/test_drift.py --test high_friction --algo gatekeeper
```

|     Backup CBF              |        MPS       |    gatekeeper    |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/user-attachments/assets/eca100f6-4f44-4740-8d7f-63d87fece2af"  height="200px"> | <img src="https://github.com/user-attachments/assets/224f1d87-960b-4625-abb2-810ca746dde7"  height="200px"> | <img src="https://github.com/user-attachments/assets/7b45d7a9-8ac0-4f48-8594-ea391e54b001"  height="200px"> |


Change the algorithm by replacing `gatekeeper` with `mps` or `backupcbf`.

```bash
# Ego lane is clear, obstacle remains in the middle lane
uv run python examples/drift_car/test_drift.py --test middle_lane_only --algo gatekeeper
```

You can run the reach-avoid example (motivated by Mario game!) by:

```bash
uv run python examples/evade/test_evade.py --algo gatekeeper
```

|     Backup CBF              |        MPS       |    gatekeeper    |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://github.com/user-attachments/assets/4e75c1be-66b8-4b4a-abda-d06e84fca0d3"  height="200px"> | <img src="https://github.com/user-attachments/assets/f10a6e6d-a4a3-4ec2-81cd-e8fc3b6ad6b4"  height="200px"> | <img src="https://github.com/user-attachments/assets/78e1d266-b052-4eb7-ba7d-bb4eee80e3dc"  height="200px"> |


If you want to export videos, add the `--save` flag.

## Citing

If you find this repository useful, please consider citing our paper:

```
@inproceedings{kim2026backupbased, 
    author    = {Kim, Taekyung and Menon, Aswin D. and Trivedi, Akshunn and Panagou, Dimitra},
    title     = {Backup-Based Safety Filters: A Comparative Review of Backup CBF, Model Predictive Shielding, and gatekeeper}, 
    booktitle = {},
    shorttitle = {Backup-Based Safety Filters},
    year      = {2026}
}
```
