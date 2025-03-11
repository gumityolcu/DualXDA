import os

root_dir="/home/fe/yolcu/Documents/Code/DualView-wip/test_output/surrogate_evaluation"
ds_name="AWA"
row_names=["Cosine similarity of weight matrices", "Correlation of logits", "Correlation of prediction", "Kendall tau-rank correlation of logits"]
short_row_names=["Weight Cos Similarity", "Logit Correlation", "Prediction Correlation", "Logit Kendall Tau"]
root_dir=os.path.join(root_dir, ds_name, "std")
final_rows=[]
final_rows.append("XAI Method,"+",".join(short_row_names))
for xai_name in sorted(os.listdir(root_dir), key=lambda x: 10000 if x=="representer" else float(x.split("_")[1])):
    if not os.path.isdir(os.path.join(root_dir, xai_name)):
        continue
    with open(os.path.join(root_dir, xai_name, f"{ds_name}_std_surrogate_evaluation.csv"), "r") as f:
       lines=f.readlines()
    assert len(lines)==5
    lines=lines[1:]
    for i in range(len(lines)):
        lines[i]=lines[i].split(",")
        lines[i][1]=float(lines[i][1].replace("\n", ""))
        assert lines[i][0]==row_names[i]
    rowstr=xai_name+","
    for i in range(len(lines)):
        rowstr+="{val:.5f}".format(val=lines[i][1])+","
    rowstr=rowstr[:-1]
    final_rows.append(rowstr)
for l in final_rows:
    print(l)
        