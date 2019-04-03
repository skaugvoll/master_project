import sys, os

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except:
    print("SAdsadsadhsa;hkldasjkd")

os.system("rm -rf {}".format("../trained_models/"))