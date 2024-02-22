# Aruco-Cube-Pose-Estimation
## Demo
![](https://github.com/ronsys11/Aruco-Cube-Pose-Estimation/blob/main/WhatsAppVideo2024-02-22at3.03.56PM-ezgif.com-video-to-gif-converter(1).gif)

### Dependencies
* `ROS 1 Noetic`
* `Opencv 4.2.0`
* `tf 1.13.2`
* `numpy 1.17.4`

### Usage
1. Make your catkin workspace and package
   ```bash
    $ mkdir workspace_name/src
    $ cd ~/catkin_ws/
    $ catkin_make
    $ cd src
    $ catkin_create_pkg aruco_package rospy cv_bridge geometry_msgs sensor_msgs visualization_msgs
    ```
2. Place the pose_estimation python file inside your package folder
3. Make your pyhton file an executable
```bash
$ cd workspace/src/package
$ chmod +x pose_estimation.py
```
4. Make workspace again
```bash
$ cd workspace
$ catkin_make
```
5. Source your workspace
```bash
$ echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
$ source ~/.bashrc
```
6. Change the aruco parameter according to your cube in
```bash
self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
```
7. Download the ROS wrapper for Intel RealSense devices from https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy
8. Run the following in a terminal
```bash
$ roslaunch realsense2_camera rs_camera.launch
```
9. Run the following in another terminal
```bash
$ rosrun package_name pose_estimation.py
```
### Ros topics
- Subscribed Topics:
  - `/camera/color/camera_info` (CameraInfo): Camera information topic.
  - `/camera/color/image_raw` (Image): Raw camera image topic.

- Published Topics:
  - `/aruco_cube/pose` (PoseStamped): Pose of the detected cube.
  - `/visualization_marker_real_cube` (Marker): RViz visualization marker for the cube.


  




