from pandas import read_csv, read_excel
# load the dataset
def load_smallECAI(full_path='../data/smaller_dataset.csv'):
    dataset = read_csv(full_path)
    dataset = dataset.drop(['Unnamed: 0', 'X1', 'nace', 'ratio036', 'ratio037', 'ratio039', 'ratio040'], axis=1)
    return dataset

def load_fullECAI(full_path='../data/final_dataset_smes.xlsx'):
    dataset = read_excel(full_path, sheet_name=0)
    dataset = dataset.drop(['Unnamed: 0', 'V_16', 'V_17', 'V_18', 'V_19', 'V_24'], axis=1)
    return dataset