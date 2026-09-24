"""
Install first:
    pip install legoeducation
Then copy lelib.py from the SimpleLE repo into this project's folder.

"""

import time

import legoeducation as le
from lelib import colorSensor, controller, doubleMotor

# --- Bluetooth card info for your hardware -------------------------------
# Fill these in with the color/serial printed on your LEGO connection card.
# Valid values: le.LEGO_COLOR_RED, _YELLOW, _BLUE, _GREEN, _PURPLE,
# _MAGENTA, _AZURE, _ORANGE.
COLOR_SENSOR_CARD_COLOR = le.LEGO_COLOR_ORANGE
COLOR_SENSOR_CARD_SERIAL = 7552

CONTROLLER_CARD_COLOR = le.LEGO_COLOR_ORANGE
CONTROLLER_CARD_SERIAL = 7552

DOUBLE_MOTOR_CARD_COLOR = le.LEGO_COLOR_ORANGE
DOUBLE_MOTOR_CARD_SERIAL = 7552

POLL_DELAY_S = 0.1  # seconds between reads

# --- Sound per detected color ----------------------------------------------
# Each color gets its own beep pitch/pattern so you can tell them apart by ear.
COLOR_SOUNDS = {
    "Red":     dict(frequency=440,  pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "Yellow":  dict(frequency=523,  pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "Blue":    dict(frequency=659,  pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "Teal":    dict(frequency=740,  pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "Green":   dict(frequency=880,  pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "Purple":  dict(frequency=988,  pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "White":   dict(frequency=1175, pattern=le.SOUND_PATTERN_BEEP_SINGLE),
    "Magenta": dict(frequency=330,  pattern=le.SOUND_PATTERN_BEEP_DOUBLE),
    "Orange":  dict(frequency=392,  pattern=le.SOUND_PATTERN_BEEP_DOUBLE),
    "Azure":   dict(frequency=294,  pattern=le.SOUND_PATTERN_BEEP_TRIPLE),
}

dm = None  # doubleMotor, connected in main()



# --- Empty handler functions ----------------------------------------------
# Fill these in with whatever behavior you want.

def play_color_sound(color_name):
    """Beep the Double Motor's speaker with the pitch/pattern for color_name."""
    sound = COLOR_SOUNDS.get(color_name)
    if sound is not None:
        dm.beep(frequency=sound["frequency"], pattern=sound["pattern"], blocking=False)


def DoRed():
    wait(2)
    print("red")
    play_color_sound("Red")



def DoYellow():
    print("yellow")
    play_color_sound("Yellow")



def DoBlue():
    print("blue")
    play_color_sound("Blue")



def DoTeal():
    play_color_sound("Teal")



def DoGreen():
    print('green')
    play_color_sound("Green")



def DoPurple():
    play_color_sound("Purple")



def DoWhite():
    play_color_sound("White")



def DoMagenta():
    play_color_sound("Magenta")



def DoOrange():
    play_color_sound("Orange")



def DoAzure():
    play_color_sound("Azure")



def DoNoColor():
    dm.stop_beep(blocking=False)



def DoUnknownColor():
    dm.beep(pattern=le.SOUND_PATTERN_BEEP_UP_MIDDLE_DOWN, frequency=220, blocking=False)



def DoLeftUp():
    pass



def DoLeftDown():
    pass



def DoLeftReleased():
    pass



def DoRightUp():
    pass



def DoRightDown():
    pass



def DoRightReleased():
    if dm is not None:
        dm.turn_right(360)



# --- Dispatch helpers -------------------------------------------------

def handle_color(color_name):
    """Big switch statement on the color sensor's detected color."""
    match color_name:
        case "Red":
            DoRed()
        case "Yellow":
            DoYellow()
        case "Blue":
            DoBlue()
        case "Teal":
            DoTeal()
        case "Green":
            DoGreen()
        case "Purple":
            DoPurple()
        case "White":
            DoWhite()
        case "Magenta":
            DoMagenta()
        case "Orange":
            DoOrange()
        case "Azure":
            DoAzure()
        case "No color":
            DoNoColor()
        case _:
            DoUnknownColor()



def handle_controller(ctl):
    """Big switch statement on the controller's joystick state."""
    if ctl.left_up():
        left_state = "up"
    elif ctl.left_down():
        left_state = "down"
    else:
        left_state = "released"

    if ctl.right_up():
        right_state = "up"
    elif ctl.right_down():
        right_state = "down"
    else:
        right_state = "released"

    match left_state:
        case "up":
            DoLeftUp()
        case "down":
            DoLeftDown()
        case "released":
            DoLeftReleased()

    match right_state:
        case "up":
            DoRightUp()
        case "down":
            DoRightDown()
        case "released":
            DoRightReleased()



# --- Main loop -------------------------------------------------------------

def main():
    global dm

    sensor = colorSensor()
    sensor.connect(card_serial=COLOR_SENSOR_CARD_SERIAL, card_color=COLOR_SENSOR_CARD_COLOR)

    ctl = controller()
    ctl.connect(card_serial=CONTROLLER_CARD_SERIAL, card_color=CONTROLLER_CARD_COLOR)

    dm = doubleMotor()
    dm.connect(card_serial=DOUBLE_MOTOR_CARD_SERIAL, card_color=DOUBLE_MOTOR_CARD_COLOR)

    last_color = None

    try:
        while True:
            color = sensor.detect_color()
            if color != last_color:
                handle_color(color)
                last_color = color
            handle_controller(ctl)
            time.sleep(POLL_DELAY_S)
    except KeyboardInterrupt:
        pass
    finally:
        dm.stop_beep(blocking=False)
        dm.stop()
        dm.disconnect()



if __name__ == "__main__":
    main()
