#!/usr/bin/python3
import os
import math

expected = [float(x.rstrip()) for x in open(
    os.path.join('dev-0', 'expected.tsv'), 'r').readlines()]

out = [float(x.rstrip()) for x in open(
    os.path.join('dev-0', 'out.tsv'), 'r').readlines()]

rmse = 0
for x, y in zip(expected, out):
    rmse += abs(x - y) ** 2

rmse = math.sqrt(rmse / len(out))

max_acceptable = 48000

print('twój wynik RMSE to: ', round(rmse, 2))
print('żeby zaliczyc zadanie powinieneś mieć conajwyżej: ', max_acceptable)

if rmse > max_acceptable:
    print('nie zaliczone')
else:
    print('zaliczone')
