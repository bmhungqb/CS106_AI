import Knapsack
import os
import sys
def main():
    file_path_data = "assets/kplib"
    groupTestCases =  os.listdir(r'{}'.format(file_path_data))
    groupSizeItem = os.listdir(r'{}'.format(file_path_data+'/'+groupTestCases[0]))
    groupSizeR = ['R01000','R10000']
    items = ['s000']
    for groupCase in groupTestCases:
        for sizeItem in groupSizeItem:
            for item in items:
                path_data = './assets/kplib/{}/{}/R01000/{}.kp'.format(groupCase,sizeItem,item)
                path_result = "assets/Result/{}/{}_R01000_{}.txt".format(groupCase,sizeItem,item)
                path_res = "assets/Result/{}".format(groupCase)
                data = {
                    "values":[],
                    "weights":[[]],
                    "capacities":[],
                    "size":[]
                }
                with open(path_data,'r') as file_data:
                    file = file_data.read().splitlines()
                    data["size"].append(int(file[1]))
                    data["capacities"].append(int(file[2]))
                    for i in range(4,len(file)):
                        data["values"].append(int(file[i].split(' ')[0]))
                        data["weights"][0].append(int(file[i].split(' ')[1]))
                result = Knapsack.knapsack(data,180)
                if not os.path.exists(path_res):
                    os.makedirs(path_res)
                with open(path_result, 'w+') as f:
                    f.write("Total value: %f\n" %result["total_value"])
                    f.write("Total weight: %f\n" %result["total_weight"])
                    f.write("Runtime: %f\n" %result["runtime"])
                    f.write("Optimal: %s\n" %result["isOptimal"])
                    f.write(">>>Packed items: %s\n"%str(result["packed_items"]))
                    f.write(">>>Packed weights: %s\n" %str(result["packed_weights"]))

if __name__ == "__main__":
    main()