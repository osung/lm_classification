import pickle

# pickle 파일에서 리스트 읽어오기
with open('summary_data_1_7999.pkl', 'rb') as file:
    loaded_list = pickle.load(file)

print("로드된 리스트:", loaded_list)

print(len(loaded_list))
