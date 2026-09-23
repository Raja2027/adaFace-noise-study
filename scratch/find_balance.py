import pandas as pd
df = pd.read_csv('data/metadata/record_label_map.csv')
counts = df['y_true'].value_counts()
for N in range(50, 201):
  if 100000 % N == 0:
    num_id = 100000 // N
    valid = (counts >= (N+4)).sum()
    if valid >= num_id:
      print(f'N={N}, num_id={num_id}, valid={valid}')
