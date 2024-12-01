import pickle

def load(filename: str):
	with open(filename, 'rb') as f_in:
		return pickle.load(f_in)

dv = load('dv.bin')
model = load('model1.bin')

client = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([client])
y_prod = model.predict_proba(X)[:, 1]

print(y_prod)