import os
import pathlib

p1 = [
    '2021-11-17_10104',
    '2021-11-19_10104',
    '2021-11-22_10104',
    '2021-12-02_10104',
    '2021-12-08_10104',

]
p2 = [
    '2021-11-18 10111',
    '2021-11-18_10111',
    '2021-11-19_10111',
    '2021-11-22_10111',
    '2021-19-10111',

]
p3 = [
    '2021-11-30 10111',
    '2021-12-01 10111',
    '2021-12-03 10111',
    '2021-12-07 10111',

]
filepath = r""
files = [os.path.join(filepath, x) for x in os.listdir(filepath)]
for f in files:
    for p in p1:
        if p in f:
            if os.path.isdir(f) and not os.path.isdir(os.path.join(filepath, "p001_" + pathlib.Path(f).name)):
                os.replace(f, os.path.join(filepath, "p001_" + pathlib.Path(f).name))
    for p in p2:
        if p in f:
            if os.path.isdir(f) and not os.path.isdir(os.path.join(filepath, "p002_" + pathlib.Path(f).name)):
                os.replace(f, os.path.join(filepath, "p002_" + pathlib.Path(f).name))
    for p in p3:
        if p in f:
            if os.path.isdir(f) and not os.path.isdir(os.path.join(filepath, "p003_" + pathlib.Path(f).name)):
                os.replace(f, os.path.join(filepath, "p003_" + pathlib.Path(f).name))

