import sys
from conformer_utils import checkMissing

if len(sys.argv)<2:
    print("Usage: checkmissing.py <path> [<type>]")
    exit(1)
else:
    path = sys.argv[1]
    if len(sys.argv)>2:
        type = sys.argv[2]
    else:
        type = "sdf"

print (checkMissing(path, type))