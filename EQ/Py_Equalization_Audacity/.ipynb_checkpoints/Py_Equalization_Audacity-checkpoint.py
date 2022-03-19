import pandas as pd
import xml.etree.ElementTree as gfg 
from lxml import etree
import argparse

def readcsv(path):
    df = pd.read_csv(path)
    return df

# XML maker and prettier
def prettyPrintXml(xmlFilePathToPrettyPrint):
    assert xmlFilePathToPrettyPrint is not None
    parser = etree.XMLParser(resolve_entities=False, strip_cdata=False)
    document = etree.parse(xmlFilePathToPrettyPrint, parser)
    document.write(xmlFilePathToPrettyPrint, pretty_print=True, encoding='utf-8')
def GenerateXML(fileName,xml_f,xml_g) :
    root = gfg.Element("equalizationeffect")
    m1 = gfg.Element("curve", name="xba-h3(16k)20210914")
    root.append (m1)
    #b1 = gfg.SubElement(m1, "curve", name="xba-h3(16k)20210914")
    for i in range(len(xml_f)):
        gfg.SubElement(m1, "point", f=xml_f[i],d=xml_g[i])
    tree = gfg.ElementTree(root)
    with open (fileName, "wb") as files :
        tree.write(files)
    print("Have save xml into "+fileName)

# TXT maker
def GemerateTXT(dicf,dicv,path):
    freqLine = ""
    valueLine = ""
    for i in range(len(dicf)):
        for j in range(len(dicf[i])):
            freqLine=freqLine+dicf[i][j]
            valueLine=valueLine+dicv[i][j]
    audacityLine = '''FilterCurve: {freqLine}FilterLength="8191" InterpolateLin="0" InterpolationMethod="B-spline" {valueLine}'''.format(freqLine=freqLine,valueLine=valueLine)
    #print(audacityLine)
    path = path+"_Audacity_config.txt"
    f = open(path, 'w')
    f.writelines(audacityLine)
    f.close()
    print("Have save txt into "+path)


def batch_processing(Raw_csv,output_name="./output",xml=False):
    df = readcsv(Raw_csv)
    tmpF = ""
    tmpV = ""
    classification =[1,2,3,4,5,6,7,8,9,0]
    dicf={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],0:[]}
    dicv={1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],0:[]}
    num = 0 
    xml_f=[]
    xml_g=[]

    # 將依照equalization之欄位值去存取eq調整數值
    for i in range(df.shape[0]):
        tmpF = 'f'+str(num)+'="'+str(df['frequency'][i])+'" '
        tmpV = 'v'+str(num)+'="'+str(df['raw'][i])+'" '
        for j in range (len(classification)):
            if(str(num)[0]==str(classification[j])):
                dicf[classification[j]].append(tmpF)
                dicv[classification[j]].append(tmpV)
                # XML(舊版本)
                xml_f.append(str(df['frequency'][i]))
                xml_g.append(str(df['raw'][i]))
                num+=1
                break
    GemerateTXT(dicf,dicv,output_name)
    
    if(xml):
        GenerateXML(output_name+"_Audacity_config.xml",xml_f,xml_g)
        prettyPrintXml(output_name+"_Audacity_config.xml")
    


def cli_args():
    """Parses command line arguments."""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-input','--Raw_csv', type=str, required=True,
                            help='Path to input CSV files.')
    arg_parser.add_argument('-output','--output_name', type=str, default=argparse.SUPPRESS,
                            help='Path to results directory and file name.')
    arg_parser.add_argument('-xml','--xml', type=str, default=argparse.SUPPRESS,
                            help='Boolean to export xml file for old system.')

    args = vars(arg_parser.parse_args())
    print(args)
    return args


if __name__ == '__main__':
    batch_processing(**cli_args())
