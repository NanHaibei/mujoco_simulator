"""兼容入口；实际的 Studio _on_step 位于 MujocoSimulator 中。"""

from mujoco_simulator_python.mujoco_simulator_python import MujocoSimulator, main


if __name__ == "__main__":
    main()
