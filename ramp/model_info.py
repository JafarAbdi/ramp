import pinocchio
import pathlib
import sys


# Why external/mujoco_menagerie/unitree_g1/g1.xml is failing?
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_filename>")
        sys.exit(1)

    model_filename = pathlib.Path(sys.argv[1])

    if not model_filename.exists():
        print(f"Model file {model_filename} does not exist")
        sys.exit(1)

    if model_filename.suffix != ".xml":
        print(f"Only mjcf files are supported! Input file {model_filename}")
        sys.exit(1)

    print(f"Loading model from {model_filename}")
    models: tuple[pinocchio.Model, pinocchio.Model, pinocchio.Model] = (
        pinocchio.shortcuts.buildModelsFromMJCF(
            model_filename,
            verbose=True,
        )
    )
    model, visual_model, collision_model = models
    print(f"Joints: {[name for name in model.names]}")
    print(f"Frames: {[frame.name for frame in model.frames]}")


if __name__ == "__main__":
    main()
