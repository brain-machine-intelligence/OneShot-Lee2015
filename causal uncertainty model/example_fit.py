from causal_uncertainty import CausalUncertaintyModel
import pandas as pd

# Simulation example
model = CausalUncertaintyModel(rounds=40, lr_base=0.1, lr_temp=3.0, primacy=0.36, recency=0.36)
print("Parameters:", model.get_params())
model.simulate()
df = model.output_dataframe()
print(df[['round_idx', 'stimulus', 'outcome', 'mean', 'tag']].head())

# Behavioral fitting example (using dummy sample data)
subject_data = pd.DataFrame({
    'round_idx': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'stimulus': [0, 1, 2, 0, 1, 0, 2, 1, 0, 2],
    'outcome': ['non-novel', 'non-novel', 'novel', 'non-novel', 'non-novel',
                'novel', 'non-novel', 'non-novel', 'non-novel', 'novel'],
    'round_type': ['type1','type1','type1','type1','type1',
                   'type2','type2','type2','type2','type2'],
    'subject_rating': [0.35, 0.22, 0.85, 0.20, 0.25, 0.81, 0.22, 0.21, 0.24, 0.88]
})
fit_result = model.fit_to_behavior(subject_data)
print("Best-fit parameters:", fit_result['params'])
print("Loss (MSE):", fit_result['loss'])
