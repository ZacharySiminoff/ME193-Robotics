"""Settings shared by characterize.py and deploy.py. Edit the connection card first."""
import legoeducation as le

# ---- the Double Motor's Connection Card (printed on the card) ------------------------------------------
CARD_SERIAL = "6065"                 # e.g. "7552"; keep it a string so leading zeros survive ("0049")
CARD_COLOR = le.LEGO_COLOR_PURPLE    # le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE, _MAGENTA, _AZURE, _ORANGE

# ---- how the computer talks to the motor ---------------------------------------------------------------
CONTROL_DT = 0.05            # s between speed commands (20 Hz). The simulation trains at this same rate.
NOTIFY_MS = 20               # ms between encoder/IMU updates from the motor (15 to 1000)
ACCELERATION = (100, 100)    # the motor's built-in ramp (acceleration, deceleration), 0..100. Same in tests and deploy.

# ---- characterization ----------------------------------------------------------------------------------
ARM_MOTOR = le.MOTOR_LEFT    # the side with the arm the camera watches (le.MOTOR_LEFT or le.MOTOR_RIGHT)
CAMERA_NAME = "FaceTime"     # use the camera whose name contains this: the Mac's own camera. Not the iPhone
                             # (Continuity Camera): it sends video over Wi-Fi, which adds delay. N in the window = next.
TEST_PERCENTS = (10, 20, 30, 40, 50, 60, 80, 100)   # steady-speed sweep (both directions)
STEP_PERCENT = 45            # step-response size; about the speed the robot drives at
STEP_REPEATS = 6             # step-response rounds (each about 5 s): more rounds pin down the timing better
RANDOM_SECONDS = 20          # length of the random-command test, the most useful one for the timing

# ---- the robot (deploy) --------------------------------------------------------------------------------
# movement_move_tank(+, +) is the Double Motor's own "forward". If that drives your robot backwards, set False.
TANK_FORWARD_IS_ROBOT_FORWARD = True
SWAP_LEFT_RIGHT = False      # set True if the motor's LEFT output drives the robot's right wheel
# The gyro's yaw is the turn about the axis through this face of the Double Motor. None = the motor's default
# (it lies flat). If the motor sits on its side or end in the robot, set the face that points UP, e.g.
# le.DEVICE_FACE_LEFT (others: _TOP, _BOTTOM, _FRONT, _BACK, _RIGHT), and redo the gyro test in characterize.py.
IMU_YAW_FACE = None
    