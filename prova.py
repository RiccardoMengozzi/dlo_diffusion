import os
import genesis as gs
import numpy as np
from env_config import ShapingConfig
from genesis.engine.entities import RigidEntity, MPMEntity
from genesis.engine.entities.rigid_entity import RigidLink
import utils 
from inference import DiffusionInference
from scipy.spatial.transform import Rotation as R
import collections
import time
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))


def main():

    gs.init(
        backend=gs.gpu,
        logging_level="info",
    )
    ########################## create a scene ##########################
    scene: gs.Scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1e-3,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=[1080, 720],
            camera_pos=[0.5, 0.0, 1.4],
            camera_lookat=[0.5, 0.0, 0.0],
            camera_fov=80,
            refresh_rate=30,
            max_FPS=240,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
        show_viewer=True,
    )


    ########################## entities ##########################
    plane = scene.add_entity(
        gs.morphs.Plane(),
    )

    table = scene.add_entity(
        morph=gs.morphs.URDF(
            file=os.path.join(
                PROJECT_FOLDER, "models/SimpleTable/SimpleTable.urdf"
            ),
            fixed=True,
        ),
        material=gs.materials.Rigid(),
        surface=gs.surfaces.Default(),
    )


    ########################## build ##########################
    scene.build()

if __name__ == "__main__":
    main()



