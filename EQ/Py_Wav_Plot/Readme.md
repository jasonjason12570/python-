---
title: Py_Wav_Plot
tags: jurytest, python, EQ validation
description: for jurytest EQ validation
---
# Py_Wav_Plot

1. 讀取音頻/具有濾波器的音頻
2. 匯出頻域圖/時域圖

        
## Prerequisites

- [Python](https://www.python.org/downloads/), which, at this writing, is the lastest 3.6.x or greater.

## Usage

參數說明：
| console | code variable    | 說明                      |
| ------- | ---------------- | ------------------------- |
| -raw    | Raw_wav          | 原始耳機頻響wav(required) |
| -output | output_name      | 輸出地址與檔名            |
| -target | Target_wav       | 目標耳機頻響wav           |
| -TD     | Time_Domain      | 時域圖顯示，預設True      |
| -FD     | Frequency_Domain | 頻域圖顯示，預設True      |
| -x      | xfig             | Plot匯出X軸大小，預設6    |
| -y      | yfig             | Plot匯出Y軸大小，預設4    |


  
### 1.Draw the plot by Raw.wav
繪出單一音檔之時域圖和頻域圖
```python=
python Py_Wav_Plot.py -raw="../Test/oBRD.wav" -output="BRD_AutoEQ"
```
輸出：
> ------Raw Plot-----
Have save fig into ./Result/Raw_FR.png
------Raw Plot-----
![](https://i.imgur.com/5gyHjVI.png)
![](https://i.imgur.com/ZdQ4Udm.png)


### 2.Draw the plot with Raw.wav and Target.wav
同時繪出兩個檔案之對比時域圖和頻域圖
```python=
python Py_Wav_Plot.py -raw="../Test/oBRD.wav" -target="../Test/re_audio.wav" -output="BRD_AutoEQ" -x=25 -y=10
```
輸出：
> ------Compare mode-----
number of samplerate = 44100
number of channels = 1
length = 18.62047619047619s
number of samplerate = 44100
number of channels = 1
length = 18.62047619047619s
Have save fig into BRD_AutoEQ_Wav_TD_twice.png
Have save fig into BRD_AutoEQ_Wav_FD_twice.png
------Compare mode-----
![](https://i.imgur.com/8bDI0SJ.png)

![](https://i.imgur.com/28p2ggM.png)



#### 圖型標記說明：
| 圖內簡標 | 詳細說明   |
| -------- | ---------- |
| Raw      | 原有頻響值 |
| Target   | 目標頻響值 |



## Reference
  * [run-jupyter-notebook-script-from-terminal](https://deeplearning.lipingyang.org/2018/03/28/run-jupyter-notebook-script-from-terminal/)