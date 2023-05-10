import pickle
import xgboost as xgb
import sys
import pandas as pd
import os

def read_data(filepath,datatable):
    "Reusable function to read PSV data from source directory"
    count = 0
    rows = 0
    for filename in os.listdir(filepath):
        if filename.endswith(".psv"): 
            with open(filepath + filename) as openfile:
                patient = filename.split(".")[0]

                file = pd.read_csv(openfile,sep = "|")
                file['Patient_ID'] = patient
                
                file = file.reset_index()
                file = file.rename(columns={"index": "Hour"})
                
                if 'SepsisLabel' in file.columns:
                  index = file.loc[file['SepsisLabel'] == 1].index.min()
                  # create new dataframe with data up to and including the row with SepsisLabel = 1
                  file = file.loc[:index]

                datatable = pd.concat([datatable, file])
                rows += file.size
                count += 1
        # Print progress after 10k files
        if count % 1000 == 0:
            print("Progress || Files: {} || Number of items: {}".format(count,rows))
    print("Done ||| Files: {} || Number of items: {}".format(count,rows))
    return(datatable)
    
test_path = sys.argv[1] +'/'
df = pd.DataFrame()
data = read_data(test_path,df)



# Create dictionary to specify the aggregation:

wide_features = [ 'HR' ,'O2Sat' ,'Temp' ,'SBP' ,'MAP' ,'DBP' ,'Resp' ,'EtCO2' ,'BaseExcess' ,'HCO3' ,'FiO2' ,'pH' ,'PaCO2' ,'SaO2' ,'AST' ,'BUN' ,'Alkalinephos' ,'Calcium' ,'Chloride' ,'Creatinine' ,'Bilirubin_direct' ,'Glucose' ,'Lactate' ,'Magnesium' ,'Phosphate' ,'Potassium' ,'Bilirubin_total' ,'TroponinI' ,'Hct' ,'Hgb' ,'PTT' ,'WBC' ,'Fibrinogen' ,'Platelets']
wide_metrics = ['max', 'min', 'mean', 'count', 'first', 'last', "std"]

narrow_features = ['Age', 'Gender', 'Unit1', 'Unit2', 'HospAdmTime', 'ICULOS', 'Hour']

if 'SepsisLabel' in data.columns:
  narrow_features.append('SepsisLabel') 

agg_dict = {k: wide_metrics for k in wide_features}
narrow_dict = {k: 'last' for k in narrow_features}
agg_dict.update(narrow_dict)


data = data.groupby('Patient_ID').agg(agg_dict)

def one_out(gdf):
  # onehot encoding the gender
  one_hot = pd.get_dummies(gdf[('Gender','last')])

  # create a new hierarchical index
  new_columns = pd.MultiIndex.from_arrays([['Gender','Gender'], one_hot.columns])
  # assign the new index to the DataFrame columns
  one_hot.columns = new_columns


  #merge back to the df
  gdf = gdf.merge(one_hot, on='Patient_ID')
  gdf = gdf.drop(('Gender','last'), axis=1)
  return gdf

data = one_out(data)


for i in wide_features:
  data[(i, "range")] = data[(i, "max")] -  data[(i, "min")]

if 'SepsisLabel' in data.columns:
  X = data.drop(('SepsisLabel','last'), axis=1).values
  y = data[('SepsisLabel','last')].values
else:
  X = data.values

# Load the XGBoost model from the pickle file
with open('model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# Evaluate the model
preds = xgb_model.predict(X)


# Convert the predicted probabilities into binary classes
threshold = 0.5
binary_preds = [1 if p > threshold else 0 for p in preds]

# from sklearn.metrics import f1_score
# print(f1_score(binary_preds, y))



data['prediction'] = binary_preds
data = data.rename_axis('id')

# Write the combined DataFrame to a CSV file
data['prediction'].to_csv('prediction.csv', index=True, header=True)
