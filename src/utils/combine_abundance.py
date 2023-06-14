import pandas as pd

def read_abundance(dataset, pred_dataset, threshold):
  path = "../data/" + dataset
  data = pd.read_csv(path + '/abundance.tsv', index_col=0, sep='\t', header=None)
  to_drop = data.loc[(data < threshold).all(axis=1)]
  data = data.drop(to_drop.index)

  path2 = "../data/" + pred_dataset
  data2 = pd.read_csv(path2 + '/abundance.tsv', index_col=0, sep='\t', header=None)
  to_drop2 = data2.loc[(data2 < threshold).all(axis=1)]
  data2 = data2.drop(to_drop.index)

  return data, data2

def main(dataset, pred_dataset, threshold):
  training_data, pred_data = read_abundance(dataset, pred_dataset, threshold)
  #training_data = training_data.set_index('OTU', inplace=True)
  #pred_data = pred_data.set_index('OTU', inplace=True)

  training_data = training_data.groupby(training_data.index).sum()
  pred_data = pred_data.groupby(pred_data.index).sum()






  #get OTUs of each dataset
  training_data_list = training_data.index.tolist()
  pred_data_list = pred_data.index.tolist()

  #get lengths
  training_data_otus = len(training_data_list)
  pred_data_otus = len(pred_data_list)

  training_data_samples = training_data.shape[1]
  pred_data_samples = pred_data.shape[1]

  #get the uncommon OTUs of each dataset
  pred_otu_not_in_training = [element for element in pred_data_list if element not in training_data_list]
  training_otu_not_in_pred = [element for element in training_data_list if element not in pred_data_list]

  #add new otus
  for otu in pred_otu_not_in_training:
    training_data.loc[otu] = [0]*training_data_samples

  for otu in training_otu_not_in_pred:
    pred_data.loc[otu] = [0]*pred_data_samples

  # for otu in pred_otu_not_in_training:
  #   print(otu)
  print(len(pred_otu_not_in_training))
  print(len(training_otu_not_in_pred))
  # print(training_data)
  # print(pred_data)
  # print(pred_data.sum().sum())
  # for otu in training_otu_not_in_pred:
  
  # #merge tables
  # combined_df = training_data.copy()
  # for i in range(pred_data_samples):
  #   combined_df[i+training_data_samples+1] = [0]*training_data_otus

  # for otu in training_data_list:
  #   if otu in pred_data_list:
  #     pred_data
  
  # print(combined_df)
  non_unique_items = []

  for item in training_data_list:
    if training_data_list.count(item) > 1 and item not in non_unique_items:
        non_unique_items.append(item)

  print(non_unique_items)
  


  # print(new_otu_pred_list)
  # print(len(new_otu_pred_list))

  # print(training_data)

  # df_merged = pd.merge(training_data, pred_data, left_index=True, right_index=True, how='outer')
  # print(df_merged.shape)




  return training_data, pred_data

main('skg-wt-t1', 'skg2-wt-t14', 0)