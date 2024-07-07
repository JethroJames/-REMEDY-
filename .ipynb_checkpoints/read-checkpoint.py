import pickle


file_path = 'model_weights/2017_uncertainty_fusion-tic/data_epoch1.pkl'


with open(file_path, 'rb') as file:

    data = pickle.load(file)

print(data)
