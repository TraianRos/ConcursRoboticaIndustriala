import time

from Arm_Lib import Arm_Device

# Get the robotic arm object

Arm = Arm_Device()

time.sleep(.1)

def main():

    while True:
        Arm.Arm_RGB_set(50, 0, 0) #RGB lights up red
        time.sleep(.5)
        Arm.Arm_RGB_set(0, 50, 0) #RGB lights up green
        time.sleep(.5)
        Arm.Arm_RGB_set(0, 0, 50) #RGB lights up blue
        time.sleep(.5)
        print(" END OF LINE! ")


if __name__=="__main__":
    try :
        main()

    except KeyboardInterrupt:
        # Release Arm object
        del Arm
        print(" Program closed! ")
        pass