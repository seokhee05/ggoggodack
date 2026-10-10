STEP 4 MERGE PACKAGE

Main:
- step4_glove_pose_dashboard_v8_hybrid_top.py
- step4_glove_top_v3.py

Models:
- models/armrest_3class_best.pt
- models/glove_runs/glove_detector_v2/weights/best.pt
- models/glove_runs/glove_detector_v2/weights/last.pt

The glove model was supplied separately and has now been placed at the path
expected by the Step 4 code.

Test video is intentionally excluded from this package because it is large.
Use crop003.mov separately for testing.

Expected Windows working directory:
C:\ggo

Expected glove model path used by the pipeline:
C:\ggo\models\glove_runs\glove_detector_v2\weights\best.pt

If you copy the package contents directly into C:\ggo, adjust the model path
only if the code is configured to use a different base directory.
