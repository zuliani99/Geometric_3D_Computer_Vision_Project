# Silhouette-Based Space Carving
The goal of this project is to implement technique known as [“space carving"](https://www.cs.toronto.edu/~kyros/pubs/00.ijcv.carve.pdf) to reconstruct the shape of a 
3D object from multiple photographs taken at known but arbitrarily distributed viewpoints. An object is placed 
on top of a rotating plate together with a custom-designed fiducal marker (see the following sections for details). 
The background is made of a uniform-colored material to be easily distinguished from the target object. A calibrated camera is
placed in front of the object capturing the scene throughout an entire rotation.

The volume occupied by the object is represented by a discrete set of voxels distributed on a cube of size N x N x N.
A voxel can be seen as the 3D extension of a “pixel” describing a property of a certain region of space. 
In this case, the property is being either occupied or not by our target object.

At each frame, a set of 3D rays exit the camera starting from the optical center and passing through each pixel of the image. Every ray may intersect some voxels before reaching either the background or the object itself.
If a ray reaches the background without touching the object, all the intersected voxels can be “carved” (ie. removed) as they represent empty space.
On the contrary, if a ray reaches the object, at least one of the intersected voxels is part of the object, so they must not be removed from the set.

For the space-carving technique being effective, two sub-problems must be solved at each frame i:
* Estimate the position of the camera with respect to the object (ie. the camera pose Ri, Ti)
* Find the “silhouette” of the object by clustering the pixels as part of the background or foreground

The complete algorithm works as follows:
* Let V be an initial set of voxels distributed in a N x N x N cube
* For each image:
	* Compute the camera Projection matrix P = K \[Ri Ti\]
  * Project all the voxels onto the image
  * Check if the projection of the voxel is inside or outside the silhouette. In the latter case, remove the voxel from V

When all the frames have been processed, the remaining set of voxels define the volume occupied by the object.

## Video Data
A package containing 4 different video sequences (plus the calibration) can be downloaded  at the following [URL](https://www.dais.unive.it/~bergamasco/teachingfiles/G3DCV2022/data.7z). After having download the compressed folder extract it to the *data* directory.


## Poject Structure
The project structure is divided into three subproject:
* Background foreground detection (*1_background_foreground_detection*)
* Markers Detector (*2_markers_detector*)
* Pose estimation (*3_pose_estimation*)
Each mini project is anassignmet which student cna decide to develop during the course or take the final project that require additional work.

The additional work is mainly summarize by the immplementation of the .ply file in order to export the object mesh built using the voxel structure.

The final project can be examined in the *space_carving* folder

## Requirements
```
pip install numpy
pip install opencv-python
```

## Project Application Start Up
```
cd space_carving
python camera_calibration.py
```
In the final project application you have also to specify the following attribute in the command line:
1. *--hd_laptop*: if you have HD screen resolution
2. *voxel_cube_edge_dim*: which is the dimension of one vexel cube edge
```
python space_carving.py 2
```

## Final Assignments Version Start Up
```
cd pose_estimation
python camera_calibration.py

cd 1_background_foreground_detection
python back_fore_undist_segmentation.py

cd 2_marker_detector/Version3_undist
python marker_detectorV3.py

cd 3_pose_estimation
python pose_estimation.py
```

## Analyse the Results
The Mesh result of each object are stored in the *optput_project* folder, I recomend use [MeshLab](https://www.meshlab.net/#download) as software to deeply analyse the object shape.

## Project Console Output
```
Marker Detector for obj01.mp4...
 DONE
Average FPS is: 0.6086838923118594
Average RMS pixel error is: 0.9412538925287559
Saving PLY file...
 DONE

Marker Detector for obj02.mp4...
 DONE
Average FPS is: 0.47182103639343614
Average RMS pixel error is: 0.896294098953868
Saving PLY file...
 DONE

Marker Detector for obj03.mp4...
 DONE
Average FPS is: 0.2545594350406094
Average RMS pixel error is: 0.9178120944262036
Saving PLY file...
 DONE

Marker Detector for obj04.mp4...
 DONE
Average FPS is: 0.6229241890497109
Average RMS pixel error is: 0.9080787727231838
Saving PLY file...
 DONE
```

