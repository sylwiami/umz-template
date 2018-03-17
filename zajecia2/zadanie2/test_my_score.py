#!/usr/bin/python3
import os
import math

expected = [x.rstrip() for x in open(
    os.path.join('dev-0', 'expected.tsv'), 'r').readlines()]

out = [x.rstrip() for x in open(
    os.path.join('dev-0', 'out.tsv'), 'r').readlines()]

acc = 0
for x, y in zip(expected, out):
    if x == y:
        acc += 1

acc = math.sqrt(acc / len(out))

min_acceptable = 0.60

print('twój wynik acc to: ', round(acc, 2))
print('żeby zaliczyc zadanie powinieneś mieć conajmniej: ', min_acceptable)

if acc < min_acceptable:
    print('nie zaliczone')
else:
    print('zaliczone')
