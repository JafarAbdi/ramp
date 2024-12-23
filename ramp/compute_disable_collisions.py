import pinocchio
import pathlib
import sys


class ComputeDisableCollisions:

    def __init__(self, model_filename):
        models: tuple[
            pinocchio.Model, pinocchio.GeometryModel, pinocchio.GeometryModel
        ] = pinocchio.shortcuts.buildModelsFromMJCF(
            model_filename,
            verbose=True,
        )
        self.model = models[0]
        self.visual_model = models[1]
        self.collision_model = models[2]
        self.collision_model.addAllCollisionPairs()

    def check_collision(self, qpos):

        data = self.model.createData()
        collision_data = self.collision_model.createData()
        # print(collision_data.activeCollisionPairs)
        pinocchio.computeCollisions(
            self.model,
            data,
            self.collision_model,
            collision_data,
            qpos,
            stop_at_first_collision=False,
        )
        pairs = {}
        for k in range(len(self.collision_model.collisionPairs)):
            cr = collision_data.collisionResults[k]
            cp = self.collision_model.collisionPairs[k]
            if cr.isCollision():
                frame_name1 = self.model.frames[
                    self.collision_model.geometryObjects[cp.first].parentFrame
                ].name
                frame_name2 = self.model.frames[
                    self.collision_model.geometryObjects[cp.second].parentFrame
                ].name
                pairs[(frame_name1, frame_name2)] = cp
        return pairs


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
    compute_disable_collisions = ComputeDisableCollisions(model_filename)

    # DISABLE "DEFAULT" COLLISIONS
    # Disable all collision checks that occur when the robot is started in its default state
    default_pairs = compute_disable_collisions.check_collision(
        pinocchio.neutral(compute_disable_collisions.model)
    )
    srdf = """
<robot name="disable_collisions">
    {disable_collisions}
</robot>
"""
    disable_collisions = []
    for first, second in default_pairs.keys():
        disable_collisions.append(
            f'<disable_collisions link1="{first}" link2="{second}" reason="Default"/>'
        )
    print("\n".join(disable_collisions))
    print(
        f"Number of collision pairs before removal: {len(compute_disable_collisions.collision_model.collisionPairs)}"
    )
    pinocchio.removeCollisionPairsFromXML(
        compute_disable_collisions.model,
        compute_disable_collisions.collision_model,
        srdf.format(disable_collisions="\n".join(disable_collisions)),
        verbose=True,
    )

    # TODO: Why this is not working???
    # for pair in default_pairs.values():
    #     compute_disable_collisions.collision_model.removeCollisionPair(pair)
    print(
        f"Number of collision pairs after removal: {len(compute_disable_collisions.collision_model.collisionPairs)}"
    )

    # DISABLE ALL "ADJACENT" LINK COLLISIONS
    # "ALWAYS" IN COLLISION: Compute the links that are always in collision
    # "NEVER" IN COLLISION: Get the pairs of links that are never in collision

    print("Done!")


if __name__ == "__main__":
    main()
