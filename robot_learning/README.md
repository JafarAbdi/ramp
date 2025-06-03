# Instructions

## Generate udev rules

```bash
# Only plug leader board
./generate_udev_rule.bash /dev/ttyACM0 LeRobotLeader | sudo tee /etc/udev/rules.d/99-usb-serial-lerobot-leader.rules
# Only plug follower board
./generate_udev_rule.bash /dev/ttyACM0 LeRobotFollower | sudo tee /etc/udev/rules.d/99-usb-serial-lerobot-follower.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## Data collection

```bash
pixi run python ./robot_learning/data_collection.py --repo_id="$HF_USER/pick_boxes" --single_task="Grasp blue box and put it in the bin" --fps=30 --push_to_hub=false
```

### Resume data collection

```bash
python ./robot_learning/data_collection.py --repo_id="$HF_USER/pick_boxes" --single_task="Pick blue box" --fps=30 --push_to_hub=false --resume=true --root=$HOME/.cache/huggingface/lerobot/JafarUruc/pick_boxes/2025_05_18-20_36_19/
```

## Dataset

```bash
pixi run huggingface-cli repo create pick_boxes --type dataset --organization JafarUruc
```

### Upload episodes

```bash
pixi run huggingface-cli upload JafarUruc/pick_boxes /home/juruc/.cache/huggingface/lerobot/JafarUruc/pick_boxes --repo-type=dataset
```

### Visualize the dataset


```bash
pixi run python ./external/lerobot/lerobot/scripts/visualize_dataset_html.py --root ~/.cache/huggingface/lerobot/JafarUruc/pick_boxes/2025_05_18-20_36_19/ --repo-id JafarUruc/pick_boxes
```

### Replay an episode

```bash
pixi run python ../external/lerobot/lerobot/scripts/control_robot.py --robot.type=so100 --control.type=replay --control.fps=30 --control.repo_id=JafarUruc/pick_boxes --control.episode=0 --control.root=$HOME/.cache/huggingface/lerobot/JafarUruc/pick_boxes/2025_05_18-20_36_19
```
