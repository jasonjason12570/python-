---
title: Py_Equalization_Audacity
tags: jurytest, python, EQ validation
description: for jurytest EQ validation
---
# Py_Frequency_Response

- 讀取頻響值，匯出Audacity的EQ設定檔
        
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
```md
-input = --Raw_csv = 原始耳機頻響csv
-output = --output_name = 輸出地址與檔名
-xml = --xml = 是否輸出xml檔案
```
  
### Export only txt

```python=
python Py_Equalization_Audacity.py -input="../Result/MI_MH750.csv" -output="./MI_MH750"
```
輸出：
> ------Export-----
Have save txt into ./MI_MH750_Audacity_config.txt
------Export-----
![](https://i.imgur.com/aymsk0Q.png)


### Export txt and xml

```python=
python Py_Equalization_Audacity.py -input="../Result/MI_MH750.csv" -output="./MI_MH750" -xml=True
```
輸出：
> ------Export-----
Have save txt into ./MI_MH750_Audacity_config.txt
Have save xml into ./MI_MH750_Audacity_config.xml
------Export-----
![](https://i.imgur.com/yBRv9JJ.png)



## Reference
  * [run-jupyter-notebook-script-from-terminal](https://deeplearning.lipingyang.org/2018/03/28/run-jupyter-notebook-script-from-terminal/)