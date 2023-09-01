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
The project structure is divided into three subprojects:
* Background foreground detection (*1_background_foreground_detection*)
* Markers Detector (*2_markers_detector*)
* Pose estimation (*3_pose_estimation*)

Each mini project is a task that the students could decide to develop during the course or to carry out the final project as I did. The additional work in the project is mainly summarized in the implementation of the .ply file in order to export the object mesh.

The final project can be examined in the *space_carving* folder.

## Requirements
```
pip install numpy
pip install opencv-python
```

## Project Application Start Up
The first step in order to start the project is to run the *camera_calibration* program to obtain the camera matrix and distortion parameters 
```
cd space_carving
python camera_calibration.py
```
Then to run the *space_carving* program you have also to specify the following positional argument:
* *voxel_cube_edge_dim*: which is the dimension of one voxel cube edge

And you can add the following option:
* *--hd_laptop*: if you have HD screen resolution

In this console example I run the *space_carving* program using a dimension of a voxel cube edge of 2 wrt the marker reference coordinates dimension
```
python space_carving.py 2
```

## Final Assignments Version Start Up
```
cd 3_pose_estimation
python camera_calibration.py

cd 1_background_foreground_detection
python back_fore_undist_segmentation.py

cd 2_markers_detector
python marker_detector.py

cd 3_pose_estimation
python pose_estimation.py
```

## Analyse the Results
The Mesh results of each object are stored in the *optput_project* folder, I recommend using [MeshLab](https://www.meshlab.net/#download) as software to deeply analyse the object shape.

## Project Console Output
```
Space Carving of obj01.mp4...
 DONE
Average FPS is: 0.6086838923118594
Average Reprojection RMS Pixel Error is: 0.9412538925287559
Saving PLY file...
 DONE

Space Carving of obj02.mp4...
 DONE
Average FPS is: 0.47182103639343614
Average Reprojection RMS Pixel Error is: 0.896294098953868
Saving PLY file...
 DONE

Space Carving of obj03.mp4...
 DONE
Average FPS is: 0.2545594350406094
Average Reprojection RMS Pixel Error is: 0.9178120944262036
Saving PLY file...
 DONE

Space Carving of for obj04.mp4...
 DONE
Average FPS is: 0.6229241890497109
Average Reprojection RMS Pixel Error is: 0.9080787727231838
Saving PLY file...
 DONE
```

