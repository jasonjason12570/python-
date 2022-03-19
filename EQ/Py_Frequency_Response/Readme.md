---
title: Py_Frequency_Response
tags: jurytest, python, EQ validation
description: for jurytest EQ validation
---
# Py_Frequency_Response

1. 讀取耳機頻響，將其繪製成頻域圖
1. 虛擬耳機技術，將Raw耳機轉換為Target頻響參數，經正規化與增益限制
        
## Prerequisites

- [Python](https://www.python.org/downloads/), which, at this writing, is the lastest 3.6.x or greater.

- csv需要按照以下格式:
    - 含有Freq欄位名稱，用以標註頻率(Frequency)
    - 含有Gain欄位名稱，用以標註震幅(Gain)

|frequency|raw|
|----|----|
|20|3.0|
|30|2.5|
|40|3.0|

## Usage

參數說明：
| -console | code variable | 說明                         |
| -------- | ------------- | ---------------------------- |
| -raw     | Raw_csv       | 原始耳機頻響csv(required)    |
| -output  | output_name   | 輸出地址與檔名               |
| -target  | Target_csv    | 目標耳機頻響csv              |
| -v       | Virtualize    | 是否進行虛擬化耳機           |
| -l       | Limit         | 目標耳機增益限制，預設無限制 |
| -x       | xfig          | Plot匯出X軸大小，預設6       |
| -y       | yfig          | Plot匯出Y軸大小，預設4       |


  
### 1.Draw the plot by Raw.csv
繪出單一頻響圖
```python=
python Py_Frequency_Response_Plot.py -raw="./Raw.csv" -outputdir="./Result/Raw" -x=20 -y=10
```
輸出：
> ------Raw Plot-----
Have save fig into ./Result/Raw_FR.png
------Raw Plot-----
![](https://i.imgur.com/COOYBVp.png)

### 2.Draw the plot with Raw.csv and Target.csv
同時繪出兩個頻響圖並且匯出兩者誤差
```python=
python Py_Frequency_Response_Plot.py -raw="./Raw.csv" -outputdir="./Result/Raw_Target" -target="./Target.csv" -x=20 -y=10
```
輸出：
> ------Compare Plot-----
Have save fig into ./Result/Raw_Target_FR.png
------Compare Plot-----
![](https://i.imgur.com/touIySN.png)

### 3.Equalization the plot with Raw.csv and Target.csv
- 虛擬耳機技術
- 產出Raw_Target.csv
- 產出Raw_Target_Equalization.png
```python=
python Py_Frequency_Response_Plot.py -raw="./Raw.csv" -output="./Result/Raw_Target" -target="./Target.csv" -v=True -l=5 -x=20 -y=10
```
輸出：
> 
> ------Virtualize Script Start-----
Have save fig into ./Result/Raw_Target_Equalization.png
Have save csv into ./Result/Raw_Target.csv
------Virtualize Script End-----
![](https://i.imgur.com/WDzPs3r.png)

#### 圖型標記說明：
| 圖內簡標          | 詳細說明              |
| ----------------- | --------------------- |
| Raw               | 原有頻響值            |
| Error             | Raw - Target          |
| LimitEqualization | 耳機增益限制之頻響    |
| Equalization      | Target - Raw          |
| Equalized         | Raw+LimitEqualization |
| Target            | 目標頻響值            |



## Reference
  * [run-jupyter-notebook-script-from-terminal](https://deeplearning.lipingyang.org/2018/03/28/run-jupyter-notebook-script-from-terminal/)