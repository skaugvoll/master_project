import sys
from utils.ColorPrint import ColorPrinter

def printProgressBar(current, totalOperations, sizeProgressBarInChars, explenation=""):
    colorPrinter = ColorPrinter()
    # try:
    #     import sys, time
    # except Exception as e:
    #     print("Could not import sys and time")

    fraction_completed = current / totalOperations
    filled_bar = round(fraction_completed * sizeProgressBarInChars)

    # \r means start from the beginning of the line
    fillerChars = "#" * filled_bar
    remains = "-" * (sizeProgressBarInChars - filled_bar)

    sys.stdout.write('\r{} {} {} [{:>7.2%}]'.format(
        colorPrinter.colorString(text=explenation, color="blue"),
        fillerChars,
        remains,
        fraction_completed
    ))

    sys.stdout.flush()