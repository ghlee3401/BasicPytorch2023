import torch
import pandas as pd 
import numpy as np 


stock = pd.read_csv(filepath_or_buffer="kospi_kosdak.csv", encoding='utf-8')

kospi = stock["Kospi"].to_numpy().astype(float)
kosdak = stock["Kosdak"].to_numpy().astype(float)

torch.manual_seed(777)
np.random.seed(777)