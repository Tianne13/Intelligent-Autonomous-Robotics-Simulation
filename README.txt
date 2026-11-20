==========================================================================
1. INSTRUCTIONS FOR RUNNING THE WEBOTS SIMULATION AND PYTHON/MATPLOTLIB SCRIPTS
==========================================================================

One-time Webots setup
  1. Open Webots, go to Tools > Preferences > General.
  2. Set "Python command" to the full path of the Python 3.11 python.exe, for example
         C:\Users\<name>\AppData\Local\Programs\Python\Python311\python.exe
     (no spaces before or after the path), click OK and restart Webots.
  3. An internet connection is needed the first time a world is opened, because
     Webots downloads the robot and arena definitions.

Component A - Python scripts
  The sensor data collected in Webots is already saved in the data folder, so these
  scripts can be run without Webots. Open a terminal in the project folder (the folder
  that contains this README) and run them in this order:

      py -3.11 notebooks\a1_perception.py
      py -3.11 notebooks\a1_fusion.py
      py -3.11 notebooks\a2_grid_map.py
      py -3.11 notebooks\a2_dijkstra.py
      py -3.11 notebooks\a2_astar.py
      py -3.11 notebooks\a2_compare.py
      py -3.11 notebooks\a3_kinematics.py
      py -3.11 notebooks\a3_pid_tracking.py
      py -3.11 notebooks\a3_pid_tuning.py
      py -3.11 notebooks\a4_evaluation.py

  Each script opens Matplotlib figure windows; close each window to let the script
  continue. Figures are saved in the figures folder and tables in the data folder.
  grid_map.py, robot_model.py, pid_controller.py and planner_astar.py are helper
  modules used by the scripts and are not run directly.

Component A - Webots simulation
  a) Data collection (optional): open webots_A\worlds\component_a.wbt, set the
     controller of the E-puck to "data_collector" and press Play. It saves the LiDAR
     scans, robot poses and camera images in the data folder.
  b) Closed loop navigation: open webots_A\worlds\component_a.wbt, set the controller
     of the E-puck to "navigator" and press Play. The robot plans a path and drives to
     the goal, and the log is saved in the data folder. Run notebooks\a4_evaluation.py
     afterwards to make the planned against actual trajectory plot and the table.

Component B - Python scripts
  Run from the project folder, in this order:

      py -3.11 notebooks\b2_consensus.py
      py -3.11 notebooks\b2_pso.py
      py -3.11 notebooks\b2_emergent_eval.py
      py -3.11 notebooks\b3_train_model.py
      py -3.11 notebooks\b3_evaluate.py
      py -3.11 notebooks\b3_architectures_compare.py

  b3_evaluate.py needs the weights saved by b3_train_model.py.

Component B - Webots simulation
  Open webots_B\worlds\component_b.wbt (controller: swarm_controller) and press Play.

==========================================================================
2. WEBOTS VERSION AND PYTHON VERSION USED
==========================================================================
Webots : R2025a
Python : 3.11.9 (64-bit)
System : Windows 11 (64-bit)

==========================================================================
3. LIST OF PYTHON DEPENDENCIES
==========================================================================
numpy 2.4.6
matplotlib 3.11.2
opencv-python 5.0.0
seaborn 0.13.2
pandas 3.0.6
scikit-learn 1.9.1
torch 2.14.1 (CPU build)

Install with:
    py -3.11 -m pip install numpy matplotlib opencv-python seaborn pandas scikit-learn
    py -3.11 -m pip install torch

==========================================================================
4. EXPECTED RUNTIME
==========================================================================
Python scripts (seconds, on the hardware in section 5):
    a1_perception.py ............ 3.0
    a1_fusion.py ................ 36
    a2_grid_map.py .............. 3.0
    a2_dijkstra.py .............. 2.5
    a2_astar.py ................. 3.3
    a2_compare.py ............... 5.3
    a3_kinematics.py ............ 3.6
    a3_pid_tracking.py .......... 3.4
    a3_pid_tuning.py ............ 8.6
    a4_evaluation.py ............ 3.2
    b2_consensus.py ............. 2.4
    b2_pso.py ................... 2.8
    b2_emergent_eval.py ......... 2.5
    b3_train_model.py ........... 97.7
    b3_evaluate.py .............. 62.1
    b3_architectures_compare.py . 2.9
All 16 scripts together: about 4 to 5 minutes.
The first run of a script that imports pandas or PyTorch can be slower while
Windows scans the libraries.
Webots data collection: about 250 seconds.
Webots closed loop navigation: about 32 simulated seconds.
Times exclude the time taken to close the figure windows.

==========================================================================
5. HARDWARE REQUIREMENTS
==========================================================================
Processor : AMD Ryzen 7 7730U with Radeon Graphics (2.00 GHz)
RAM       : 16 GB (15.3 GB usable)
Graphics  : AMD Radeon(TM) Graphics (integrated); no dedicated GPU is needed
            and PyTorch runs on the CPU
Storage   : the project folder is well under 50 MB