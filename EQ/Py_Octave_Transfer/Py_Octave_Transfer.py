import pandas as pd
import numpy as np
import argparse
octave_24 = [20.0,20.6,21.2,21.8,22.4,23.0,23.6,24.3,25.0,25.8,26.5,27.2,28.0,29.0,30.0,30.7,31.5,32.5,33.5,34.5,35.5,36.5,37.5,38.7,40.0,41.2,42.5,43.7,45.0,46.2,47.5,48.7,50.0,51.5,53.0,54.5,56.0,58.0,60.0,61.5,63.0,65.0,67.0,69.0,71.0,73.0,75.0,77.5,80.0,82.5,85.0,87.5,90.0,92.5,95.0,97.5,100.0,103.0,106.0,109.0,112.0,115.0,118.0,122.0,125.0,128.0,132.0,136.0,140.0,145.0,150.0,155.0,160.0,165.0,170.0,175.0,180.0,185.0,190.0,195.0,200.0,206.0,212.0,218.0,224.0,230.0,236.0,243.0,250.0,258.0,265.0,272.0,280.0,290.0,300.0,307.0,315.0,325.0,335.0,345.0,355.0,365.0,375.0,387.0,400.0,412.0,425.0,437.0,450.0,462.0,475.0,487.0,500.0,515.0,530.0,545.0,560.0,580.0,600.0,615.0,630.0,650.0,670.0,690.0,710.0,730.0,750.0,775.0,800.0,825.0,850.0,875.0,900.0,925.0,950.0,975.0,1000.0,1030.0,1060.0,1090.0,1120.0,1150.0,1180.0,1220.0,1250.0,1280.0,1320.0,1360.0,1400.0,1450.0,1500.0,1550.0,1600.0,1650.0,1700.0,1750.0,1800.0,1850.0,1900.0,1950.0,2000.0,2060.0,2120.0,2180.0,2240.0,2300.0,2360.0,2430.0,2500.0,2580.0,2650.0,2720.0,2800.0,2900.0,3000.0,3070.0,3150.0,3250.0,3350.0,3450.0,3550.0,3650.0,3750.0,3870.0,4000.0,4120.0,4250.0,4370.0,4500.0,4620.0,4750.0,4870.0,5000.0,5150.0,5300.0,5450.0,5600.0,5800.0,6000.0,6150.0,6300.0,6500.0,6700.0,6900.0,7100.0,7300.0,7500.0,7750.0,8000.0,8250.0,8500.0,8750.0,9000.0,9250.0,9500.0,9750.0,10000.0,10300.0,10600.0,10900.0,11200.0,11500.0,11800.0,12200.0,12500.0,12800.0,13200.0,13600.0,14000.0,14500.0,15000.0,15500.0,16000.0,16500.0,17000.0,17500.0,18000.0,18500.0,19000.0,19500.0,20000.0]
octave_12 = [20.0,21.2,22.4,23.6,25.0,26.5,28.0,30.0,31.5,33.5,35.5,37.5,40.0,42.5,45.0,47.5,50.0,53.0,56.0,60.0,63.0,67.0,71.0,75.0,80.0,85.0,90.0,95.0,100.0,106.0,112.0,118.0,125.0,132.0,140.0,150.0,160.0,170.0,180.0,190.0,200.0,212.0,224.0,236.0,250.0,265.0,280.0,300.0,315.0,335.0,355.0,375.0,400.0,425.0,450.0,475.0,500.0,530.0,560.0,600.0,630.0,670.0,710.0,750.0,800.0,850.0,900.0,950.0,1000.0,1060.0,1120.0,1180.0,1250.0,1320.0,1400.0,1500.0,1600.0,1700.0,1800.0,1900.0,2000.0,2120.0,2240.0,2360.0,2500.0,2650.0,2800.0,3000.0,3150.0,3350.0,3550.0,3750.0,4000.0,4250.0,4500.0,4750.0,5000.0,5300.0,5600.0,6000.0,6300.0,6700.0,7100.0,7500.0,8000.0,8500.0,9000.0,9500.0,10000.0,10600.0,11200.0,11800.0,12500.0,13200.0,14000.0,15000.0,16000.0,17000.0,18000.0,19000.0,20000.0]
octave_06 = [20.0,22.4,25.0,28.0,31.5,35.5,40.0,45.0,50.0,56.0,63.0,71.0,80.0,90.0,100.0,112.0,125.0,140.0,160.0,180.0,200.0,224.0,250.0,280.0,315.0,355.0,400.0,450.0,500.0,560.0,630.0,710.0,800.0,900.0,1000.0,1120.0,1250.0,1400.0,1600.0,1800.0,2000.0,2240.0,2500.0,2800.0,3150.0,3550.0,4000.0,4500.0,5000.0,5600.0,6300.0,7100.0,8000.0,9000.0,10000.0,11200.0,12500.0,14000.0,16000.0,18000.0,20000.0]
def readcsv(path):
    df = pd.read_csv(path)
    return df

def round_v2(num, decimal):
    num = np.round(num, decimal)
    num = float(num)
    return num

def _octave24(df,octave_24):
    df_freq = []
    df_gain = []
    for i in range (len(octave_24)):
        tmp = []
        tmpn = 0
        for j in range (len(df)):
            if(i==0):
                if(df.frequency[j]==octave_24[i]):
                    tmp.append(df.raw[j])
            elif(i>0 and i<len(octave_24)-1):
                if(df.frequency[j]<octave_24[i+1] and df.frequency[j]>octave_24[i-1]):
                    tmp.append(df.raw[j])
            elif(i==len(octave_24)-1):
                if(df.frequency[j]>octave_24[i-1]):
                    tmp.append(df.raw[j])
        
        for n in range (len(tmp)):
            tmpn = tmpn+tmp[n]
        if(tmpn!=0):
            mean = tmpn/len(tmp)
        else:
            mean = 0

        df_freq.append(octave_24[i])
        df_gain.append(mean)
        dfnew = pd.DataFrame({'frequency':df_freq,'raw':df_gain})
    return dfnew
        

def _octave12(df,octave):
    df_freq = []
    df_gain = []
    for i in range (len(octave)):
        tmp = []
        tmpn = 0
        for j in range (len(df)):
            if(i==0):
                if(df.frequency[j]==octave[i]):
                    tmp.append(df.raw[j])
            elif(i>0 and i<len(octave)-1):
                if(df.frequency[j]<octave[i+1] and df.frequency[j]>octave[i-1]):
                    tmp.append(df.raw[j])
            elif(i==len(octave)-1):
                if(df.frequency[j]>octave[i-1]):
                    tmp.append(df.raw[j])
        
        for n in range (len(tmp)):
            tmpn = tmpn+tmp[n]
        if(tmpn!=0):
            mean = tmpn/len(tmp)
            mean = round_v2(mean,2)
        else:
            mean = 0

        df_freq.append(octave[i])
        df_gain.append(mean)
        dfnew = pd.DataFrame({'frequency':df_freq,'raw':df_gain})
    return dfnew


def batch_processing(Raw_csv,output_name="./output_octave_24.csv"):
    df = readcsv(Raw_csv)
    df = _octave24(df,octave_24)
    df = _octave12(df,octave_12)
    df.to_csv(output_name, index=False)


def cli_args():
    """Parses command line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-input','--Raw_csv', type=str, required=True,
                            help='Path to input CSV files.')
    arg_parser.add_argument('-output','--output_name', type=str, default=argparse.SUPPRESS,
                            help='Path to results directory and file name.')

    args = vars(arg_parser.parse_args())
    print(args)
    return args


if __name__ == '__main__':
    batch_processing(**cli_args())