#!/usr/bin/env python3
import argparse
import re


NUISANCE_TYPES = {
    "shape", "shapeN2", "lnN", "lnU", "gmN", "gmM", "param",
    "rateParam", "flatParam", "discrete", "constr",
}


def natural_key(value):
    return [int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", value)]


def is_nuisance_line(line):
    fields = line.split()
    return len(fields) >= 2 and fields[1] in NUISANCE_TYPES


def sort_datacard(filename):
    with open(filename, encoding="utf-8") as input_file:
        lines = input_file.readlines()

    nuisance_indices = [
        index for index, line in enumerate(lines) if is_nuisance_line(line)
    ]
    nuisance_lines = sorted(
        (lines[index] for index in nuisance_indices),
        key=lambda line: natural_key(line.split()[0]),
    )

    for index, nuisance_line in zip(nuisance_indices, nuisance_lines):
        lines[index] = nuisance_line

    with open(filename, "w", encoding="utf-8") as output_file:
        output_file.writelines(lines)


parser = argparse.ArgumentParser(
    description="Sort datacard nuisance parameters in natural numerical order."
)
parser.add_argument("datacard", help="Datacard text file to update in place")
args = parser.parse_args()

sort_datacard(args.datacard)
