import numpy as np
import util
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import json
import vaex

filename = "/scratch/visual/bzfharth/in-silico-install-dir/project_src/in_silico_framework/getting_started/linked-views-example-data/backend_data_2023-06-22/simulation_samples.csv"
df = vaex.from_csv(filename, copy_index=False)

columns = ['ephys.CaDynamics_E2_v2.apic.decay','ephys.NaTa_t.axon.gNaTa_tbar']
col1 = columns[0]
col2 = columns[1]


df.select_nothing(name="foobar")
limits = [(150.43976544267906, 192.23353845624592),(1.625505788407902, 1.6641847312004356)]
df.select_rectangle(df[col1], df[col2], limits, name="foobar", mode="or")
limits = [(130.43976544267906, 192.23353845624592),(1.625505788407902, 1.6641847312004356)]
df.select_rectangle(df[col1], df[col2], limits, name="foobar", mode="or")

print(df.evaluate(df[col1], selection="foobar"))