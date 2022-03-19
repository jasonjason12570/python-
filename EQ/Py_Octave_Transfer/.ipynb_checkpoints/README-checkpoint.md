---
title: Py_Octave_Transfer
tags: jurytest, python, EQ validation
description: for jurytest EQ validation
---
# Py_Octave_Transfer
- 轉換AutoEQ原有的frequency為標準之1/12 Octave

## Prerequisites

- [Python](https://www.python.org/downloads/), which, at this writing, is the lastest 3.6.x or greater.

- csv需要按照以下格式:
    - 含有Frequency欄位名稱，用以標註頻率(Frequency)
    - 含有raw欄位名稱，用以標註震幅(raw)

| frequency | raw |
| --------- | --- |
| 20        | 3.0 |
| 30        | 2.5 |
| 40        | 3.0 |

## Usage

參數說明：
```md
-input = --Raw_csv = 原始耳機頻響csv
-output = --output_name = 輸出地址與檔名(預設為./octave_result.csv)
```
  
### Export Transfer csv

```python=
python Py_Octave_Transfer.py -input="Xiaomi Piston 2.csv" -output="./octave_MI.csv"
```
經圖形化輸出：
Raw為AutoEQ，Target為經轉換為1/12 Octave之曲線
![](https://i.imgur.com/TqjwfJo.png)

而經Octave轉化之虛擬耳機Filter為以下：
* Raw為AutoEQ以Xiaomi Piston 2模擬Sony MH750
* Target為經Octave轉換以Xiaomi Piston 2模擬Sony MH750
![](https://i.imgur.com/TGHNlfH.png)


## Reference
  * [run-jupyter-notebook-script-from-terminal](https://deeplearning.lipingyang.org/2018/03/28/run-jupyter-notebook-script-from-terminal/)

