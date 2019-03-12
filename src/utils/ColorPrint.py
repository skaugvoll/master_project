
class ColorPrinter:
    def __init__(self):
        self.reset  = "\u001b[0m"  # reset (turn of color)
        self.black  = 30
        self.red    = 31
        self.green  = 32
        self.yellow = 33
        self.blue   = 34
        self.magenta = 35
        self.cyan   = 36



    def get_color(self, color, bright=True):
        clr = None

        if color == 'black':
            clr = self.black

        elif color == 'red':
            clr = self.red

        elif color == 'green':
            clr = self.green

        elif color == 'yellow':
            clr = self.yellow

        elif color == 'blue':
            clr = self.blue

        elif color == 'magenta':
            clr = self.magenta

        elif color == 'cyan':
            clr = self.cyan

        #################

        if not clr is None:
            if bright:
                clr += 60
            return "\033[{}m".format(str(clr))

        else:
            return ""

    def colorString(self, text, color=None, bright=True):
        return self.get_color(color, bright) + text + self.reset
